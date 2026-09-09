from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Generator
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from presine.game.config import GameConfig
from presine.game.observation import InformationState
from presine.game.state import Phase, RoundState
from presine.policies.neural import NeuralPolicy

from ..checkpoint import load_training_payload, save_policy_checkpoint, save_training_checkpoint
from ..config import TrainingConfig
from ..multiplayer import multiplayer_encoding
from .encoding import (
    ACTION_DIM,
    ENCODING_VERSION,
    STATE_DIM,
    action_id,
    action_probabilities_from_logits,
    encode_information_state,
    legal_action_mask,
)
from .memory import MemoryBatch, ReservoirBuffer
from .network import StrategyNetwork


@dataclass(frozen=True, slots=True)
class IterationMetrics:
    iteration: int
    elapsed_seconds: float
    nodes_visited: int
    advantage_memory: tuple[int, int]
    policy_memory: tuple[int, int]
    advantage_loss: tuple[float, float]
    policy_loss: tuple[float, float] | None
    advantage_validation_loss: tuple[float, float]
    policy_validation_loss: tuple[float, float] | None
    timing_seconds: dict[str, float]
    coverage: dict[str, object]
    uniform_fallback: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "iteration": self.iteration,
            "elapsed_seconds": self.elapsed_seconds,
            "nodes_visited": self.nodes_visited,
            "nodes_per_second": self.nodes_visited / self.elapsed_seconds
            if self.elapsed_seconds
            else 0.0,
            "advantage_memory": self.advantage_memory,
            "policy_memory": self.policy_memory,
            "advantage_loss": self.advantage_loss,
            "policy_loss": self.policy_loss,
            "advantage_validation_loss": self.advantage_validation_loss,
            "policy_validation_loss": self.policy_validation_loss,
            "timing_seconds": self.timing_seconds,
            "coverage": self.coverage,
            "uniform_fallback": self.uniform_fallback,
        }


@dataclass(frozen=True, slots=True)
class _InferenceRequest:
    info: InformationState
    actor: int


_Traversal = Generator[_InferenceRequest, tuple[float, ...], float]


@dataclass(frozen=True, slots=True)
class FitMetrics:
    train_loss: float
    validation_loss: float


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


class DeepCFRTrainer:
    """Model-based Deep CFR with exact actions and external sampling."""

    def __init__(
        self,
        game_config: GameConfig,
        training_config: TrainingConfig,
    ) -> None:
        if not 2 <= game_config.players <= 6:
            raise ValueError("Deep CFR supports players in 2..6")
        self.game_config = game_config
        self.training_config = training_config
        self.device = resolve_device(training_config.device)
        if game_config.players == 2:
            self.state_dim = STATE_DIM
            self.encoding_version = ENCODING_VERSION
            self._encode = encode_information_state
            self._legal_mask = legal_action_mask
            self._depth_index = 10
        else:
            self.state_dim = multiplayer_encoding.state_dim(game_config.players)
            self.encoding_version = multiplayer_encoding.encoding_version(game_config.players)
            self._encode = multiplayer_encoding.encode_information_state
            self._legal_mask = multiplayer_encoding.legal_action_mask
            self._depth_index = multiplayer_encoding.trick_index_offset(game_config.players)
        self.rng = random.Random(training_config.seed)
        torch.manual_seed(training_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(training_config.seed)
        if training_config.deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
        self.advantage_networks = self._new_networks()
        self.policy_networks = self._new_networks()
        self.advantage_memories = tuple(
            ReservoirBuffer(
                training_config.advantage_memory,
                seed=training_config.seed + 101 + player,
                state_dim=self.state_dim,
                action_dim=ACTION_DIM,
                depth_index=self._depth_index,
            )
            for player in range(game_config.players)
        )
        self.policy_memories = tuple(
            ReservoirBuffer(
                training_config.policy_memory,
                seed=training_config.seed + 201 + player,
                state_dim=self.state_dim,
                action_dim=ACTION_DIM,
                depth_index=self._depth_index,
            )
            for player in range(game_config.players)
        )
        self.iteration = 0
        self.nodes_visited = 0
        self.regret_queries = [0] * game_config.players
        self.uniform_fallbacks = [0] * game_config.players
        self._inference_seconds = 0.0
        self.final_policy_fit: tuple[FitMetrics, ...] | None = None
        self.final_policy_fit_seconds = 0.0
        self.stop_requested = False
        self.stop_signal: int | None = None

    def request_stop(self, signal_number: int | None = None) -> None:
        """Request a checkpoint at the next complete iteration boundary."""
        self.stop_requested = True
        self.stop_signal = signal_number

    def _new_networks(self) -> tuple[StrategyNetwork, ...]:
        config = self.training_config.network
        return tuple(
            StrategyNetwork(
                config.hidden_sizes,
                dropout=config.dropout,
                input_dim=self.state_dim,
                output_dim=ACTION_DIM,
            ).to(self.device)
            for _ in range(self.game_config.players)
        )

    def current_policy(self) -> NeuralPolicy:
        return NeuralPolicy(self.advantage_networks, device=str(self.device), regret_matching=True)

    def average_policy(self) -> NeuralPolicy:
        return NeuralPolicy(self.policy_networks, device=str(self.device), regret_matching=False)

    def train(self, output_dir: Path) -> list[IterationMetrics]:
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / ".internal" / "metrics.jsonl"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        reports: list[IterationMetrics] = []
        target = self.training_config.iterations
        while self.iteration < target:
            report = self.train_iteration()
            reports.append(report)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(report.to_dict(), sort_keys=True) + "\n")
            if report.iteration % self.training_config.checkpoint_every == 0:
                self.save_training(output_dir)
            if self.stop_requested:
                # Never serialize a half-fitted network or a half-collected
                # iteration. The next SLURM slice resumes an equivalent state.
                self.save_training(output_dir)
                return reports
        final_policy_started = time.perf_counter()
        self.final_policy_fit = self._train_policy_networks()
        self.final_policy_fit_seconds = time.perf_counter() - final_policy_started
        # A signal received during the final refit is harmless once the refit
        # and both atomic saves can complete.  Do not make a successor repeat
        # the final refit forever when the target iteration was already met.
        self.stop_requested = False
        self.stop_signal = None
        self.save(output_dir)
        return reports

    def train_iteration(self) -> IterationMetrics:
        started = time.perf_counter()
        self.iteration += 1
        before_nodes = self.nodes_visited
        before_queries = tuple(self.regret_queries)
        before_fallbacks = tuple(self.uniform_fallbacks)
        self._inference_seconds = 0.0
        players = range(self.game_config.players)
        losses = [math.nan] * self.game_config.players
        validation_losses = [math.nan] * self.game_config.players
        traversal_seconds = 0.0
        advantage_fit_seconds = 0.0
        for traverser in players:
            traversal_started = time.perf_counter()
            self._collect_traversals(traverser)
            traversal_seconds += time.perf_counter() - traversal_started
            if self.iteration % self.training_config.train_every == 0:
                fit_started = time.perf_counter()
                fit = self._fit_network(
                    self.advantage_networks[traverser],
                    self.advantage_memories[traverser],
                    self.training_config.advantage_steps,
                    reset=True,
                )
                advantage_fit_seconds += time.perf_counter() - fit_started
                losses[traverser] = fit.train_loss
                validation_losses[traverser] = fit.validation_loss
        policy_losses: tuple[float, ...] | None = None
        policy_validation_losses: tuple[float, ...] | None = None
        policy_fit_seconds = 0.0
        if (
            self.training_config.policy_train_every
            and self.iteration % self.training_config.policy_train_every == 0
        ):
            policy_started = time.perf_counter()
            policy_fit = self._train_policy_networks()
            policy_fit_seconds = time.perf_counter() - policy_started
            policy_losses = tuple(result.train_loss for result in policy_fit)
            policy_validation_losses = tuple(result.validation_loss for result in policy_fit)
        elapsed = time.perf_counter() - started
        query_delta = tuple(
            self.regret_queries[player] - before_queries[player] for player in players
        )
        fallback_delta = tuple(
            self.uniform_fallbacks[player] - before_fallbacks[player] for player in players
        )
        total_queries = sum(query_delta)
        total_fallbacks = sum(fallback_delta)
        return IterationMetrics(
            iteration=self.iteration,
            elapsed_seconds=elapsed,
            nodes_visited=self.nodes_visited - before_nodes,
            advantage_memory=tuple(len(memory) for memory in self.advantage_memories),
            policy_memory=tuple(len(memory) for memory in self.policy_memories),
            advantage_loss=tuple(float(loss) for loss in losses),
            policy_loss=policy_losses,
            advantage_validation_loss=tuple(float(loss) for loss in validation_losses),
            policy_validation_loss=policy_validation_losses,
            timing_seconds={
                "total": elapsed,
                "traversal": traversal_seconds,
                "inference": self._inference_seconds,
                "advantage_fit": advantage_fit_seconds,
                "policy_fit": policy_fit_seconds,
            },
            coverage={
                "advantage": tuple(memory.coverage() for memory in self.advantage_memories),
                "policy": tuple(memory.coverage() for memory in self.policy_memories),
            },
            uniform_fallback={
                "queries": total_queries,
                "fallbacks": total_fallbacks,
                "rate": total_fallbacks / total_queries if total_queries else 0.0,
                "queries_by_player": query_delta,
                "fallbacks_by_player": fallback_delta,
                "rate_by_player": tuple(
                    fallback_delta[player] / query_delta[player] if query_delta[player] else 0.0
                    for player in players
                ),
                "cumulative_queries_by_player": tuple(self.regret_queries),
                "cumulative_fallbacks_by_player": tuple(self.uniform_fallbacks),
            },
        )

    def _iteration_game_config(self) -> GameConfig:
        """Return the role-balanced rules for the current iteration."""
        if not self.training_config.alternate_starting_player:
            return self.game_config
        offset = (self.iteration - 1) % self.game_config.players
        return replace(
            self.game_config,
            starting_player=(self.game_config.starting_player + offset) % self.game_config.players,
        )

    def _collect_traversals(self, traverser: int) -> None:
        """Run independent traversals while batching their network requests."""
        remaining = self.training_config.traversals_per_player
        limit = min(self.training_config.inference_batch_size, remaining)
        active: list[_Traversal] = []
        requests: list[_InferenceRequest] = []
        for network in self.advantage_networks:
            network.eval()

        def fill() -> None:
            nonlocal remaining
            while remaining and len(active) < limit:
                traversal = self._traverse_generator(
                    RoundState(self._iteration_game_config()), traverser
                )
                remaining -= 1
                try:
                    request = next(traversal)
                except StopIteration:
                    continue
                active.append(traversal)
                requests.append(request)

        fill()
        while active:
            strategies = self._regret_strategies(requests)
            next_active: list[_Traversal] = []
            next_requests: list[_InferenceRequest] = []
            for traversal, strategy in zip(active, strategies):
                try:
                    request = traversal.send(strategy)
                except StopIteration:
                    continue
                next_active.append(traversal)
                next_requests.append(request)
            active = next_active
            requests = next_requests
            fill()

    def _traverse(self, state: RoundState, traverser: int) -> float:
        """Run one traversal in legacy request order (mainly for tests/tools)."""
        traversal = self._traverse_generator(state, traverser)
        try:
            request = next(traversal)
        except StopIteration as completed:
            return completed.value
        while True:
            strategy = self._regret_strategies((request,))[0]
            try:
                request = traversal.send(strategy)
            except StopIteration as completed:
                return completed.value

    def _traverse_generator(self, state: RoundState, traverser: int) -> _Traversal:
        self.nodes_visited += 1
        if state.phase == Phase.TERMINAL:
            return state.utilities()[traverser]
        if state.phase == Phase.CHANCE:
            # Questa mano serve solo per questo giro: le carte sono già corrette
            # e non vale la pena conservare una copia da ripristinare dopo.
            state.apply_chance_fast_unchecked(state.sample_chance(self.rng))
            return (yield from self._traverse_generator(state, traverser))

        actor = state.current_player
        info = state.information_state()
        strategy = yield _InferenceRequest(info, actor)
        if actor == traverser:
            action_values: list[float] = []
            for action in info.legal_actions:
                if self.game_config.hand_size > 1:
                    token = state.apply_action_fast(action)
                    action_values.append((yield from self._traverse_generator(state, traverser)))
                    state.undo_fast(token)
                else:
                    token = state.apply_action(action)
                    action_values.append((yield from self._traverse_generator(state, traverser)))
                    state.undo(token)
            node_value = sum(
                probability * value for probability, value in zip(strategy, action_values)
            )
            target = np.zeros(ACTION_DIM, dtype=np.float32)
            for action, value in zip(info.legal_actions, action_values):
                target[action_id(info, action)] = value - node_value
            self.advantage_memories[actor].add(
                self._encode(info),
                target,
                self._legal_mask(info),
                float(self.iteration),
            )
            return node_value

        target = np.zeros(ACTION_DIM, dtype=np.float32)
        for action, probability in zip(info.legal_actions, strategy):
            target[action_id(info, action)] = probability
        self.policy_memories[actor].add(
            self._encode(info),
            target,
            self._legal_mask(info),
            float(self.iteration),
        )
        action = self.rng.choices(info.legal_actions, weights=strategy, k=1)[0]
        if self.game_config.hand_size > 1:
            token = state.apply_action_fast(action)
            value = yield from self._traverse_generator(state, traverser)
            state.undo_fast(token)
        else:
            token = state.apply_action(action)
            value = yield from self._traverse_generator(state, traverser)
            state.undo(token)
        return value

    @torch.inference_mode()
    def _regret_strategies(
        self, requests: tuple[_InferenceRequest, ...] | list[_InferenceRequest]
    ) -> list[tuple[float, ...]]:
        started = time.perf_counter()
        encoded = [self._encode(request.info) for request in requests]
        logits: list[np.ndarray | None] = [None] * len(requests)
        for actor in range(self.game_config.players):
            indices = [index for index, request in enumerate(requests) if request.actor == actor]
            if not indices:
                continue
            states = torch.from_numpy(np.stack([encoded[index] for index in indices])).to(
                self.device
            )
            outputs = self.advantage_networks[actor](states).cpu().numpy()
            for index, output in zip(indices, outputs):
                logits[index] = output
        strategies: list[tuple[float, ...]] = []
        for index, request in enumerate(requests):
            output = logits[index]
            if output is None:
                raise AssertionError(f"no inference output for actor {request.actor}")
            self.regret_queries[request.actor] += 1
            positive_total = sum(
                max(float(output[action_id(request.info, action)]), 0.0)
                for action in request.info.legal_actions
            )
            if positive_total <= 1e-12:
                self.uniform_fallbacks[request.actor] += 1
            strategies.append(
                action_probabilities_from_logits(
                    request.info,
                    output,
                    regret_matching=True,
                )
            )
        self._inference_seconds += time.perf_counter() - started
        return strategies

    def _train_policy_networks(self, *, steps: int | None = None) -> tuple[FitMetrics, ...]:
        resolved_steps = self.training_config.policy_steps if steps is None else steps
        return tuple(
            self._fit_network(
                self.policy_networks[player],
                self.policy_memories[player],
                resolved_steps,
                reset=True,
                classification=True,
            )
            for player in range(self.game_config.players)
        )

    def refit_policy_networks(self, *, steps: int, seed: int) -> tuple[FitMetrics, ...]:
        if steps <= 0:
            raise ValueError("policy refit steps must be positive")
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return self._train_policy_networks(steps=steps)

    def _fit_network(
        self,
        network: StrategyNetwork,
        memory: ReservoirBuffer,
        steps: int,
        *,
        reset: bool,
        classification: bool = False,
    ) -> FitMetrics:
        if not len(memory) or not memory.partition_size("train"):
            return FitMetrics(math.nan, math.nan)
        if reset:
            network.reset_parameters()
        network.to(self.device).train()
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        total = 0.0
        for _ in range(steps):
            batch = memory.sample(self.training_config.batch_size, self.rng, partition="train")
            loss = self._batch_loss(network, batch, classification=classification)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 10.0)
            optimizer.step()
            total += float(loss.detach().cpu())
        network.eval()
        validation_batch = memory.validation_batch(self.training_config.validation_samples)
        if validation_batch is None:
            validation_loss = math.nan
        else:
            with torch.no_grad():
                validation_loss = float(
                    self._batch_loss(network, validation_batch, classification=classification).cpu()
                )
        return FitMetrics(total / steps, validation_loss)

    def _batch_loss(
        self,
        network: StrategyNetwork,
        batch: MemoryBatch,
        *,
        classification: bool,
    ) -> torch.Tensor:
        states = torch.from_numpy(batch.states).to(self.device)
        targets = torch.from_numpy(batch.targets).to(self.device)
        masks = torch.from_numpy(batch.masks).to(self.device)
        weights = torch.from_numpy(batch.weights).to(self.device)
        weights = weights / weights.mean().clamp_min(1e-8)
        predictions = network(states)
        if classification:
            masked_logits = predictions.masked_fill(~masks, -1e9)
            log_probabilities = torch.log_softmax(masked_logits, dim=1)
            sample_loss = -(targets * log_probabilities).sum(dim=1)
        else:
            squared = (predictions - targets).square()
            sample_loss = (squared * masks).sum(dim=1) / masks.sum(dim=1).clamp_min(1)
        return (sample_loss * weights).mean()

    def save(self, output_dir: Path) -> None:
        self.save_training(output_dir)
        self.save_policy(output_dir)

    def save_training(self, output_dir: Path) -> None:
        save_training_checkpoint(self, output_dir / "training.pt")

    def save_policy(self, output_dir: Path) -> None:
        save_policy_checkpoint(self, output_dir / "policy.pt")

    @classmethod
    def restore(cls, path: Path, *, device_override: str | None = None) -> DeepCFRTrainer:
        payload = load_training_payload(path, device="cpu")
        game = GameConfig.from_dict(payload["game"])  # type: ignore[arg-type]
        training_values = dict(payload["training"])  # type: ignore[arg-type]
        if device_override is not None:
            training_values["device"] = device_override
        training = TrainingConfig.from_dict(training_values)
        trainer = cls(game, training)
        trainer.iteration = int(payload["iteration"])
        trainer.nodes_visited = int(payload["nodes_visited"])
        diagnostics = payload.get("diagnostics", {})
        trainer.regret_queries = list(diagnostics.get("regret_queries", (0, 0)))  # type: ignore[union-attr]
        trainer.uniform_fallbacks = list(diagnostics.get("uniform_fallbacks", (0, 0)))  # type: ignore[union-attr]
        trainer.rng.setstate(payload["rng_state"])  # type: ignore[arg-type]
        for network, state in zip(trainer.advantage_networks, payload["advantage_networks"]):  # type: ignore[arg-type]
            network.load_state_dict(state)
        for network, state in zip(trainer.policy_networks, payload["policy_networks"]):  # type: ignore[arg-type]
            network.load_state_dict(state)
        trainer.advantage_memories = tuple(
            ReservoirBuffer.from_state_dict(values)
            for values in payload["advantage_memories"]  # type: ignore[union-attr]
        )  # type: ignore[assignment]
        trainer.policy_memories = tuple(
            ReservoirBuffer.from_state_dict(values)
            for values in payload["policy_memories"]  # type: ignore[union-attr]
        )  # type: ignore[assignment]
        torch.set_rng_state(payload["torch_rng_state"])  # type: ignore[arg-type]
        cuda_states = payload.get("cuda_rng_states")
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_states)  # type: ignore[arg-type]
        return trainer
