"""Deterministic per-iteration seed derivation.

A single run uses one master seed; per-iteration RNGs are derived from
(master_seed, domain, *identifiers) via SHA-256 truncated to 64 bits.
The `domain` tag ("sample" vs "shuffle") keeps independent random streams
uncorrelated even when they share identifiers.
"""
import hashlib


def derive_seed(master_seed: int, domain: str, *identifiers) -> int:
    payload = "|".join([str(master_seed), domain, *(str(i) for i in identifiers)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")
