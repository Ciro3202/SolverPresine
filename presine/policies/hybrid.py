from __future__ import annotations

import torch

from presine.game.observation import InformationState
from presine.learning.deep_cfr.encoding import (
    action_probabilities_from_logits,
    encode_information_state,
)
from presine.learning.deep_cfr.network import StrategyNetwork


class HybridPolicy:
    """Exact sparse table with an optional neural fallback for missing states."""

    def __init__(
        self,
        entries,
        networks: tuple[StrategyNetwork, StrategyNetwork] | None = None,
        *,
        device: str = "cpu",
        network_fitted: bool = False,
        mode: str = "auto",
        neural_fallback_phases: tuple[int, ...] | None = None,
    ) -> None:
        self.entries = entries
        self.networks = networks
        self.network_fitted = network_fitted
        if mode not in {"auto", "exact", "nearest", "nearest-blend", "neural", "uniform"}:
            raise ValueError("invalid hybrid policy mode")
        self.mode = mode
        self.neural_fallback_phases = (
            None
            if neural_fallback_phases is None
            else frozenset(int(phase) for phase in neural_fallback_phases)
        )
        self.device = torch.device(device)
        # Lightweight inference diagnostics.  These counters deliberately
        # live on the policy instance (and are not part of checkpoints), so
        # evaluation can report how often a trajectory used the exact sparse
        # table versus the neural/uniform fallback without changing behavior.
        self._lookup_counts = {"exact": 0, "nearest": 0, "neural": 0, "uniform": 0}
        self._lookup_by_player = {
            player: {"exact": 0, "nearest": 0, "neural": 0, "uniform": 0} for player in (0, 1)
        }
        self._lookup_by_seat_phase: dict[str, dict[str, int]] = {}
        # Building this index requires a full scan and duplicates many keys.
        # The normal exact+neural policy never needs it, especially for the
        # multi-hundred-million-state round-three checkpoint.
        self._nearest_buckets = (
            self._build_nearest_buckets(entries) if mode in {"nearest", "nearest-blend"} else None
        )
        if self.networks:
            for network in self.networks:
                network.to(self.device).eval()

    def reset_stats(self) -> None:
        """Reset inference hit counters before a new evaluation."""
        for mode in self._lookup_counts:
            self._lookup_counts[mode] = 0
        for counters in self._lookup_by_player.values():
            for mode in counters:
                counters[mode] = 0
        self._lookup_by_seat_phase.clear()

    def stats(self) -> dict[str, object]:
        """Return JSON-serializable exact/neural/uniform lookup statistics."""
        total = sum(self._lookup_counts.values())

        def ratios(counts: dict[str, int]) -> dict[str, float]:
            denominator = sum(counts.values())
            return {
                f"{mode}_rate": counts[mode] / denominator if denominator else 0.0
                for mode in ("exact", "nearest", "neural", "uniform")
            }

        return {
            "mode": self.mode,
            "lookups": total,
            "lookup_counts": dict(self._lookup_counts),
            "lookup_rates": ratios(self._lookup_counts),
            "by_player": {
                str(player): {
                    "lookup_counts": dict(counts),
                    "lookup_rates": ratios(counts),
                }
                for player, counts in self._lookup_by_player.items()
            },
            "by_seat_phase": {
                label: {
                    "lookup_counts": dict(counts),
                    "lookup_rates": ratios(counts),
                }
                for label, counts in self._lookup_by_seat_phase.items()
            },
        }

    def _record(self, info: InformationState, mode: str) -> None:
        self._lookup_counts[mode] += 1
        self._lookup_by_player[info.player][mode] += 1
        label = f"seat_{info.player}/phase_{info.phase}"
        counts = self._lookup_by_seat_phase.setdefault(
            label, {name: 0 for name in self._lookup_counts}
        )
        counts[mode] += 1

    @staticmethod
    def _public_bucket(key: tuple[object, ...]) -> tuple[object, ...]:
        """Bucket states with the same public decision situation.

        Private observations and the current legal action set are intentionally
        excluded: they are the features on which nearest-neighbour selection
        operates.  Candidate probabilities are aligned to the current action
        set at lookup time.
        """
        return (
            key[0],
            key[1],
            key[2],
            key[3],
            key[4],
            key[5],
            key[6],
            key[7],
            key[10],
            key[11],
        )

    @classmethod
    def _build_nearest_buckets(cls, entries):
        buckets: dict[tuple[object, ...], list[tuple[object, ...]]] = {}
        for key in entries:
            buckets.setdefault(cls._public_bucket(key), []).append(key)
        return buckets

    @staticmethod
    def _distance(left: tuple[object, ...], right: tuple[object, ...]) -> float:
        """Distance between private observations in the same public bucket."""
        left_hand = set(left[8])
        right_hand = set(right[8])
        left_initial = set(left[9])
        right_initial = set(right[9])
        # Current hand cards affect legal actions directly; initial private
        # cards affect beliefs.  Both are more informative than public fields,
        # which are equal by construction of the bucket.
        return float(
            2 * len(left_hand.symmetric_difference(right_hand))
            + len(left_initial.symmetric_difference(right_initial))
        )

    def _nearest(self, info: InformationState):
        if self._nearest_buckets is None:
            return None
        candidates = self._nearest_buckets.get(self._public_bucket(info.key()), ())
        if not candidates:
            return None
        scored = sorted(
            ((self._distance(key, info.key()), key) for key in candidates),
            key=lambda item: item[0],
        )[:8]
        if not scored:
            return None
        if scored[0][0] > 8.0:
            return None
        probabilities = [0.0] * len(info.legal_actions)
        total_weight = 0.0
        for distance, key in scored:
            actions, values = self.entries[key]
            weight = 1.0 / (1.0 + distance)
            total_weight += weight
            candidate_values = dict(zip(actions, values))
            default = 1.0 / len(info.legal_actions)
            for index, action in enumerate(info.legal_actions):
                probabilities[index] += weight * candidate_values.get(action, default)
        if total_weight <= 0.0:
            return None
        return tuple(value / total_weight for value in probabilities), scored[0][0]

    def _neural_probabilities(self, info: InformationState) -> tuple[float, ...]:
        state = torch.from_numpy(encode_information_state(info)).to(self.device).unsqueeze(0)
        logits = self.networks[info.player](state)[0].cpu().numpy()
        return action_probabilities_from_logits(info, logits, regret_matching=False)

    @torch.no_grad()
    def probabilities(self, info: InformationState) -> tuple[float, ...]:
        entry = None if self.mode in {"neural", "uniform"} else self.entries.get(info.key())
        if entry is not None:
            self._record(info, "exact")
            actions, probabilities = entry
            if actions != info.legal_actions:
                raise ValueError("hybrid policy action order differs from the game")
            return probabilities
        if self.mode in {"nearest", "nearest-blend"}:
            nearest_result = self._nearest(info)
            if nearest_result is not None:
                nearest, distance = nearest_result
                self._record(info, "nearest")
                if self.mode == "nearest" or self.networks is None or not self.network_fitted:
                    return nearest
                # A similar state is a useful prior, not a replacement for
                # the learned fallback.  Give it at most 25% weight; distant
                # neighbours receive less.
                neural = self._neural_probabilities(info)
                alpha = min(0.25, 1.0 / (1.0 + distance))
                return tuple(
                    alpha * near + (1.0 - alpha) * learned for near, learned in zip(nearest, neural)
                )
        neural_phase_enabled = (
            self.neural_fallback_phases is None or info.phase in self.neural_fallback_phases
        )
        if (
            self.mode in {"auto", "neural"}
            and self.networks is not None
            and self.network_fitted
            and neural_phase_enabled
        ):
            self._record(info, "neural")
            return self._neural_probabilities(info)
        self._record(info, "uniform")
        return (1.0 / len(info.legal_actions),) * len(info.legal_actions)
