#!/usr/bin/env python3
"""
Text embeddings for the semantic schema retriever (Roadmap Part 4.2, Retriever 2).

Uses fastembed (ONNX, no PyTorch). The model (~130 MB) is downloaded from
Hugging Face the first time and cached; after that it runs offline on CPU.

    from embedder import get_embedder
    emb = get_embedder()                     # BAAI/bge-small-en-v1.5
    emb.embed_query("films after 2000")      # -> unit vector (list of floats)
"""

import hashlib
import math
import re

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def _unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class Embedder:
    def __init__(self, model_name=DEFAULT_MODEL):
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise SystemExit("fastembed is not installed; run `uv sync`") from e
        self.model_name = model_name
        self._model = TextEmbedding(model_name)
        self._cache = {}

    def embed_query(self, text):
        key = ("q", text)
        if key not in self._cache:
            v = next(iter(self._model.query_embed([text])))
            self._cache[key] = _unit([float(x) for x in v])
        return self._cache[key]

    def embed_passages(self, texts):
        missing = [t for t in texts if ("p", t) not in self._cache]
        if missing:
            for t, v in zip(missing, self._model.passage_embed(missing)):
                self._cache[("p", t)] = _unit([float(x) for x in v])
        return [self._cache[("p", t)] for t in texts]


class HashEmbedder:
    """
    Dependency-free stand-in (bag of hashed word stems). Used by the tests and
    for a quick dry run; it is NOT a semantic model.
    """
    model_name = "hash-bow"

    def __init__(self, dim=256):
        self.dim = dim

    def _vec(self, text):
        from schema_retriever import tokens
        v = [0.0] * self.dim
        for t in tokens(re.sub(r"[^\w\s]", " ", text), drop_stopwords=True):
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % self.dim] += 1.0
        return _unit(v)

    def embed_query(self, text):
        return self._vec(text)

    def embed_passages(self, texts):
        return [self._vec(t) for t in texts]


_INSTANCES = {}


def get_embedder(model_name=DEFAULT_MODEL):
    """Shared instance per model ('hash' gives the HashEmbedder stand-in)."""
    if model_name not in _INSTANCES:
        _INSTANCES[model_name] = HashEmbedder() if model_name == "hash" else Embedder(model_name)
    return _INSTANCES[model_name]
