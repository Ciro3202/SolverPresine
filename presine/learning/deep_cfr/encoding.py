from __future__ import annotations

import numpy as np

from presine.game.cards import NUM_CARDS
from presine.game.observation import EventKind, InformationState
from presine.game.state import Phase

MAX_EVENTS = 18
ENCODING_VERSION = "exact-cards-perfect-recall-v1"
EVENT_FEATURES = 4 + 2 + NUM_CARDS
SCALAR_FEATURES = 3 + 2 + 2 + 2 + 1 + 1 + 14 + 12
CARD_FEATURES = NUM_CARDS * 4
STATE_DIM = SCALAR_FEATURES + CARD_FEATURES + MAX_EVENTS * EVENT_FEATURES

BID_OFFSET = 0
CARD_OFFSET = 6
ACE_OFFSET = 46
ACTION_DIM = 48


def action_id(info: InformationState, action: int) -> int:
    phase = Phase(info.phase)
    if phase == Phase.BID:
        return BID_OFFSET + action
    if phase == Phase.PLAY:
        return CARD_OFFSET + action
    if phase == Phase.ACE_CHOICE:
        return ACE_OFFSET + action
    raise ValueError("actions exist only at decision phases")


def legal_action_mask(info: InformationState) -> np.ndarray:
    mask = np.zeros(ACTION_DIM, dtype=np.bool_)
    for action in info.legal_actions:
        mask[action_id(info, action)] = True
    return mask


def encode_information_state(info: InformationState) -> np.ndarray:
    """Lossless fixed-width encoding for the implemented rules.

    Exact card identities and the ordered public history are retained. No card
    bucket or action merging occurs anywhere in the learning pipeline.
    """

    vector = np.zeros(STATE_DIM, dtype=np.float32)
    cursor = 0

    phase_index = {
        Phase.BID: 0,
        Phase.PLAY: 1,
        Phase.ACE_CHOICE: 2,
    }[Phase(info.phase)]
    vector[cursor + phase_index] = 1.0
    cursor += 3
    vector[cursor + info.player] = 1.0
    cursor += 2
    vector[cursor + info.starting_player] = 1.0
    cursor += 2
    vector[cursor + info.leader] = 1.0
    cursor += 2
    vector[cursor] = info.hand_size / 5.0
    cursor += 1
    vector[cursor] = info.trick_index / 5.0
    cursor += 1

    for player in range(2):
        bid = info.bids[player]
        vector[cursor + (0 if bid < 0 else bid + 1)] = 1.0
        cursor += 7
    for player in range(2):
        vector[cursor + info.catches[player]] = 1.0
        cursor += 6

    for card in info.visible_cards:
        vector[cursor + card] = 1.0
    cursor += NUM_CARDS
    for card in info.initial_private_observation:
        vector[cursor + card] = 1.0
    cursor += NUM_CARDS
    for actor, card in info.current_trick:
        relative = int(actor != info.player)
        vector[cursor + relative * NUM_CARDS + card] = 1.0
    cursor += 2 * NUM_CARDS

    if len(info.public_history) > MAX_EVENTS:
        raise ValueError("public history exceeds the exact encoder capacity")
    for index, event in enumerate(info.public_history):
        start = cursor + index * EVENT_FEATURES
        vector[start + int(event.kind)] = 1.0
        relative = int(event.actor != info.player)
        vector[start + 4 + relative] = 1.0
        value = event.value
        if event.kind == EventKind.BID:
            value = event.value
        elif event.kind == EventKind.TRICK_WINNER:
            value = relative
        if not 0 <= value < NUM_CARDS:
            raise ValueError("event value cannot be encoded")
        vector[start + 6 + value] = 1.0
    return vector


def action_probabilities_from_logits(
    info: InformationState,
    logits: np.ndarray,
    *,
    regret_matching: bool,
) -> tuple[float, ...]:
    ids = [action_id(info, action) for action in info.legal_actions]
    values = np.asarray([logits[index] for index in ids], dtype=np.float64)
    if regret_matching:
        values = np.maximum(values, 0.0)
        total = float(values.sum())
        if total <= 1e-12:
            return (1.0 / len(ids),) * len(ids)
        return tuple(float(value / total) for value in values)
    values -= float(values.max())
    probabilities = np.exp(values)
    probabilities /= probabilities.sum()
    return tuple(float(value) for value in probabilities)
