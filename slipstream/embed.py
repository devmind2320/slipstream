"""Embeddings with an offline-first default.

Backend ``hash`` (default) builds a deterministic 256-d vector from hashed word
unigrams, bigrams and character 4-grams, L2-normalised. It is not a semantic
model — it is a cheap lexical-similarity signal that is stable across machines
and needs no download. That is what INV-5 requires of the default path.

Backend ``st`` swaps in ``sentence-transformers/all-MiniLM-L6-v2``. Same
interface, same call sites; the only difference is vector quality. Which backend
ran is recorded on every eval report so no number is ever ambiguous.
"""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache

import numpy as np

from . import settings
from .textutil import normalise, tokens

DIM = 256

_st_model = None


def _hash_vector(text: str) -> np.ndarray:
    vec = np.zeros(DIM, dtype=np.float32)
    toks = tokens(normalise(text))
    if not toks:
        return vec

    def bump(feature: str, weight: float) -> None:
        h = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % DIM
        sign = 1.0 if h[4] & 1 else -1.0
        vec[idx] += sign * weight

    for t in toks:
        bump("w:" + t, 1.0)
    for a, b in zip(toks, toks[1:]):
        bump(f"b:{a}_{b}", 0.7)
    flat = " ".join(toks)
    for i in range(len(flat) - 3):
        bump("c:" + flat[i : i + 4], 0.25)

    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def _load_st():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer  # lazy, optional

        _st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _st_model


@lru_cache(maxsize=8192)
def embed(text: str) -> tuple[float, ...]:
    """Cached because the estimator embeds the same seed corpora on every call."""
    if settings.EMBED_BACKEND == "st":
        v = _load_st().encode(text, normalize_embeddings=True)
        return tuple(float(x) for x in v)
    return tuple(float(x) for x in _hash_vector(text))


def embed_array(text: str) -> np.ndarray:
    return np.asarray(embed(text), dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class NearestNeighbours:
    """Tiny brute-force kNN index. Corpora here are tens of items, not millions."""

    def __init__(self, corpus: list[str]) -> None:
        self.corpus = list(corpus)
        self.matrix = (
            np.stack([embed_array(c) for c in self.corpus])
            if self.corpus
            else np.zeros((0, DIM), dtype=np.float32)
        )

    def max_similarity(self, text: str) -> tuple[float, str | None]:
        if self.matrix.shape[0] == 0:
            return 0.0, None
        q = embed_array(text)
        nq = float(np.linalg.norm(q))
        if nq == 0.0:
            return 0.0, None
        sims = self.matrix @ (q / nq)
        norms = np.linalg.norm(self.matrix, axis=1)
        norms[norms == 0.0] = 1.0
        sims = sims / norms
        i = int(np.argmax(sims))
        return float(sims[i]), self.corpus[i]

    def top_k_mean(self, text: str, k: int = 3) -> float:
        """Mean of the top-k similarities. Less jumpy than a single max."""
        if self.matrix.shape[0] == 0:
            return 0.0
        q = embed_array(text)
        nq = float(np.linalg.norm(q))
        if nq == 0.0:
            return 0.0
        norms = np.linalg.norm(self.matrix, axis=1)
        norms[norms == 0.0] = 1.0
        sims = (self.matrix @ (q / nq)) / norms
        k = min(k, sims.shape[0])
        top = np.sort(sims)[-k:]
        return float(np.mean(top))


def semantic_entropy(samples: list[str]) -> float:
    """Cluster samples by pairwise similarity; entropy over cluster sizes.

    Used by the consistency detector when the provider can multi-sample. High
    entropy == the model tells a different story each time.
    """
    if len(samples) < 2:
        return 0.0
    vecs = [embed_array(s) for s in samples]
    clusters: list[list[int]] = []
    for i, v in enumerate(vecs):
        for c in clusters:
            if cosine(v, vecs[c[0]]) >= 0.82:
                c.append(i)
                break
        else:
            clusters.append([i])
    n = len(samples)
    h = -sum((len(c) / n) * math.log(len(c) / n) for c in clusters)
    max_h = math.log(n)
    return float(h / max_h) if max_h > 0 else 0.0
