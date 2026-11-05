"""Deterministic identities used by sampling and evaluation."""

from __future__ import annotations

import hashlib
from typing import Any


def stable_seed(*parts: Any) -> int:
    payload = repr(parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def source_observation_seed(identity_index: int, point_id: int, base: int = 20261102) -> int:
    """Return the frozen source-view seed for a synthetic material point."""
    if identity_index < 0 or point_id < 0:
        raise ValueError("identity_index and point_id must be non-negative")
    return int(base + (identity_index * 32 + point_id) * 7919)
