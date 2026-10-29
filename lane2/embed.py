"""Deterministic token-hash embedder (lane2 shared utility).

Bag-of-words hashed into a fixed-dim L2-normalized vector. This is
the embedder the v2 design's cosine signals (guard Part D ent/cos,
arbiter D2) are computed with in this build: no model download, no
runtime dependency, identical vectors on every machine. Cosine
between a sentence and a job label reflects shared content tokens —
weaker than a neural embedder, and the guard thresholds (spec
defaults) are reported against this reality in the Step 6 bench.
"""
import hashlib
import math
import re

_STOP = frozenset(
    "the a an and or of to in on for with is are was were it its "
    "that this your you i me my we our they them he she".split())
_TOK = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    def __init__(self, dim=256):
        self.dim = dim

    def __call__(self, text):
        vec = [0.0] * self.dim
        for tok in _TOK.findall((text or "").lower()):
            if tok in _STOP or len(tok) < 3:
                continue
            h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
