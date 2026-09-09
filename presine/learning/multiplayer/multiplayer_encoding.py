from __future__ import annotations

import numpy as np

from presine.game.cards import NUM_CARDS
from presine.game.observation import EventKind, InformationState
from presine.game.state import Phase

from ..deep_cfr.encoding import ACTION_DIM, action_id

PLAYERS = 4
# A complete hand can expose one BID and one PLAY event per player/trick,
# plus one TRICK_WINNER event per trick (and an occasional ACE_CHOICE).  Six
# players therefore need 42+ slots; keep headroom without storing state rows.
MAX_EVENTS = 48
ENCODING_VERSION = "relative-exact-cards-perfect-recall-4p-v3"


def event_features(players: int) -> int:
    return 4 + players + NUM_CARDS


def state_dim(players: int) -> int:
    scalar_features = 3 + players * 3 + 2 + players * 7 + players * 6
    card_features = NUM_CARDS * (1 + players + players)
    return scalar_features + card_features + MAX_EVENTS * event_features(players)


def trick_index_offset(players: int) -> int:
    return 3 + players * 3 + 1


def encoding_version(players: int) -> str:
    return (
        ENCODING_VERSION
        if players == PLAYERS
        else f"relative-exact-cards-perfect-recall-{players}p-v3"
    )


EVENT_FEATURES = event_features(PLAYERS)
STATE_DIM = state_dim(PLAYERS)
TRICK_INDEX_OFFSET = trick_index_offset(PLAYERS)


def legal_action_mask(info: InformationState) -> np.ndarray:
    mask = np.zeros(ACTION_DIM, dtype=np.bool_)
    for action in info.legal_actions:
        mask[action_id(info, action)] = True
    return mask


def encode_information_state(info: InformationState) -> np.ndarray:
    players = len(info.bids)
    if not 3 <= players <= 6:
        raise ValueError("the multiplayer encoder requires 3..6 players")
    vector = np.zeros(state_dim(players), dtype=np.float32)
    cursor = 0
    phase_index = {Phase.BID: 0, Phase.PLAY: 1, Phase.ACE_CHOICE: 2}[Phase(info.phase)]
    vector[cursor + phase_index] = 1.0
    cursor += 3
    # Every seat channel is relative to the acting player.  This makes a
    # clockwise rotation of an otherwise identical position encode exactly
    # the same way, so a single network can learn all seats without wasting
    # capacity on absolute labels.
    vector[cursor] = 1.0
    cursor += players
    vector[cursor + (info.starting_player - info.player) % players] = 1.0
    cursor += players
    vector[cursor + (info.leader - info.player) % players] = 1.0
    cursor += players
    vector[cursor] = info.hand_size / 5.0
    cursor += 1
    vector[cursor] = info.trick_index / 5.0
    cursor += 1

    for relative in range(players):
        player = (info.player + relative) % players
        bid = info.bids[player]
        vector[cursor + (0 if bid < 0 else bid + 1)] = 1.0
        cursor += 7
    for relative in range(players):
        player = (info.player + relative) % players
        vector[cursor + info.catches[player]] = 1.0
        cursor += 6

    for card in info.visible_cards:
        vector[cursor + card] = 1.0
    cursor += NUM_CARDS
    if info.hand_size == 1:
        # In the blind round the flattened observation is ordered by relative
        # opponent seat.  Separate channels retain card ownership exactly.
        for offset, card in enumerate(info.initial_private_observation, start=1):
            vector[cursor + offset * NUM_CARDS + card] = 1.0
    else:
        for card in info.initial_private_observation:
            vector[cursor + card] = 1.0
    cursor += players * NUM_CARDS
    for actor, card in info.current_trick:
        relative = (actor - info.player) % players
        vector[cursor + relative * NUM_CARDS + card] = 1.0
    cursor += players * NUM_CARDS

    if len(info.public_history) > MAX_EVENTS:
        raise ValueError("public history exceeds the multiplayer encoder capacity")
    for index, event in enumerate(info.public_history):
        start = cursor + index * event_features(players)
        vector[start + int(event.kind)] = 1.0
        relative = (event.actor - info.player) % players
        vector[start + 4 + relative] = 1.0
        value = relative if event.kind == EventKind.TRICK_WINNER else event.value
        if not 0 <= value < NUM_CARDS:
            raise ValueError("event value cannot be encoded")
        vector[start + 4 + players + value] = 1.0
    return vector
