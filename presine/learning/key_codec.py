from __future__ import annotations

"""Lossless codec for solver information keys.

Information keys contain only integers and nested tuples.  A tiny dedicated
codec is substantially smaller than pickle, deterministic across Python
versions, and can be decoded without executing a general object deserializer.
"""

from presine.policies.tabular import InformationKey

INT = 0
TUPLE = 1
PACKED_KEY_CODEC = "presine-key-v1"


def _put_varint(target: bytearray, value: int) -> None:
    while value >= 0x80:
        target.append((value & 0x7F) | 0x80)
        value >>= 7
    target.append(value)


def _get_varint(data: memoryview, cursor: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if cursor >= len(data) or shift > 63:
            raise ValueError("invalid packed information key")
        byte = int(data[cursor])
        cursor += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, cursor
        shift += 7


def _encode(value: object, target: bytearray) -> None:
    if isinstance(value, int):
        target.append(INT)
        # Zig-zag keeps the common -1 sentinel and small non-negative values
        # to one byte.
        zigzag = value * 2 if value >= 0 else -value * 2 - 1
        _put_varint(target, zigzag)
        return
    if isinstance(value, tuple):
        target.append(TUPLE)
        _put_varint(target, len(value))
        for item in value:
            _encode(item, target)
        return
    raise TypeError(f"unsupported information-key component: {type(value)!r}")


def encode_information_key(key: InformationKey) -> bytes:
    target = bytearray()
    _encode(key, target)
    return bytes(target)


def _decode(data: memoryview, cursor: int) -> tuple[object, int]:
    if cursor >= len(data):
        raise ValueError("truncated packed information key")
    kind = int(data[cursor])
    cursor += 1
    if kind == INT:
        zigzag, cursor = _get_varint(data, cursor)
        return (zigzag // 2 if zigzag % 2 == 0 else -(zigzag // 2) - 1), cursor
    if kind == TUPLE:
        length, cursor = _get_varint(data, cursor)
        values: list[object] = []
        for _ in range(length):
            value, cursor = _decode(data, cursor)
            values.append(value)
        return tuple(values), cursor
    raise ValueError(f"unknown packed information-key tag {kind}")


def decode_information_key(payload: bytes | memoryview) -> InformationKey:
    data = memoryview(payload)
    value, cursor = _decode(data, 0)
    if cursor != len(data) or not isinstance(value, tuple):
        raise ValueError("invalid packed information key")
    return value
