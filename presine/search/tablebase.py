from __future__ import annotations

"""Exact perfect-information endgame solver for the final Presine tricks."""

from presine.game.state import Phase, RoundState


class EndgameTablebase:
    """Memoized Max-N minimax on a determinized state.

    In heads-up the utilities are zero-sum, so this is ordinary minimax.  The
    class deliberately accepts only play/ace-choice states with a bounded
    number of remaining tricks: bidding and hidden-card inference belong to
    the blueprint/resolver layer, not to the tablebase.
    """

    def __init__(self, max_tricks: int = 2) -> None:
        if max_tricks not in (1, 2):
            raise ValueError("endgame tablebase supports one or two tricks")
        self.max_tricks = max_tricks
        self._cache: dict[tuple[object, ...], tuple[float, ...]] = {}
        self.hits = 0
        self.misses = 0

    def eligible(self, state: RoundState) -> bool:
        return (
            state.config.players == 2
            and state.phase in (Phase.PLAY, Phase.ACE_CHOICE, Phase.TERMINAL)
            and state.config.hand_size - state.trick_index <= self.max_tricks
        )

    def solve(self, state: RoundState) -> tuple[float, ...]:
        if not self.eligible(state):
            raise ValueError("state is outside the configured endgame tablebase")
        if state.is_terminal:
            return state.utilities()
        key = state.state_key()
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        actor = state.current_player
        best: tuple[float, ...] | None = None
        for action in state.legal_actions():
            token = state.apply_action_solver(action)
            value = self.solve(state)
            state.undo_solver(token)
            if best is None or value[actor] > best[actor]:
                best = value
        if best is None:
            raise RuntimeError("non-terminal tablebase state has no legal actions")
        self._cache[key] = best
        return best

    def action_values(self, state: RoundState) -> tuple[tuple[int, ...], tuple[float, ...]]:
        if not self.eligible(state) or state.is_terminal:
            raise ValueError("action values require an eligible decision state")
        actor = state.current_player
        actions = state.legal_actions()
        values: list[float] = []
        for action in actions:
            token = state.apply_action_solver(action)
            values.append(self.solve(state)[actor])
            state.undo_solver(token)
        return actions, tuple(values)

    def stats(self) -> dict[str, int]:
        return {"states": len(self._cache), "hits": self.hits, "misses": self.misses}
