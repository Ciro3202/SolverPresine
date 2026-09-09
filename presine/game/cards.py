from __future__ import annotations

NUM_CARDS = 40
ACE_OF_DENARI = 30
ACE_LOW = 0
ACE_HIGH = 1


def card_name(card: int) -> str:
    if not 0 <= card < NUM_CARDS:
        raise ValueError("card must be in 0..39")
    suits = ("Bastoni", "Spade", "Coppe", "Denari")
    return f"{card % 10 + 1} di {suits[card // 10]}"


def card_strength(card: int, ace_high: bool | None) -> int:
    if card == ACE_OF_DENARI:
        if ace_high is None:
            raise ValueError("the Ace of Denari needs an explicit high/low choice")
        return NUM_CARDS if ace_high else -1
    return card
