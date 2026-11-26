#!/usr/bin/env python3
"""
Ranked schema retrievers. (Roadmap Part 4.2 and 4.4)

Every retriever turns a question + schema into one ranked list of schema
elements, so they can be compared with Recall@k and MRR:

    label      "User"
    rel type   "WATCHED"
    property   "User.state"  /  "WATCHED.watchCount"

Retrievers:
    bm25      Retriever 1, lexical. BM25 over the element names.
    semantic  Retriever 2, embeddings. Each element gets a short text
              description (name, endpoints, owner, type) that is embedded
              with sentence-transformers and ranked by cosine similarity.
              Needs:  uv sync --extra semantic
    hybrid    Reciprocal-rank fusion of bm25 and semantic.

Retriever 3 (topology: keep the smallest connected subgraph) works on top
of any of these; see schema_retriever.retrieve(..., retriever=...).

    from retrievers import get_ranker
    ranked = get_ranker("bm25").rank(question, schema)   # [(element, score), ...]
"""

import json
import math
from collections import Counter

# Reuse the tokeniser from the first retriever so both behave the same way.
from schema_retriever import STOPWORDS, BM25, _split_ident, tokens

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ------------------------------------------------------------
# Schema -> elements
# ------------------------------------------------------------

def parse_schema(schema):
    return json.loads(schema) if isinstance(schema, str) else schema


def schema_elements(schema):
    """
    Return a list of dicts: {"id", "kind", "tokens", "text"}.

    "tokens" feed BM25, "text" is the natural-language description that
    the semantic retriever embeds.
    """
    s = parse_schema(schema)
    node_props = s.get("node_props", {})
    rel_props = s.get("rel_props", {})
    rels = s.get("relationships", [])

    labels = sorted(set(node_props) | {r["start"] for r in rels} | {r["end"] for r in rels})
    ends = {}
    for r in rels:
        ends.setdefault(r["type"], set()).add((r["start"], r["end"]))

    def words(name):
        return " ".join(_split_ident(name))

    out = []
    for L in labels:
        props = [p["property"] for p in node_props.get(L, [])]
        text = f"{words(L)}: a {words(L)} node"
        if props:
            text += " with " + ", ".join(words(p) for p in props[:15])
        out.append({"id": L, "kind": "label", "tokens": tokens(L) * 3, "text": text})

    for T in sorted(ends):
        pairs = sorted(ends[T])
        text = f"{words(T)}: relationship " + "; ".join(
            f"from {words(a)} to {words(b)}" for a, b in pairs[:5])
        props = [p["property"] for p in rel_props.get(T, [])]
        if props:
            text += ", with " + ", ".join(words(p) for p in props)
        endpoint_toks = [t for a, b in pairs for t in tokens(a) + tokens(b)]
        out.append({"id": T, "kind": "rel", "tokens": tokens(T) * 3 + endpoint_toks,
                    "text": text})

    for owners, what in ((node_props, "node"), (rel_props, "relationship")):
        for owner in sorted(owners):
            for p in owners[owner]:
                name = p["property"]
                dtype = str(p.get("datatype", "")).lower()
                text = f"{words(name)} of {words(owner)} ({what} property, {dtype})"
                out.append({"id": f"{owner}.{name}", "kind": "prop",
                            "tokens": tokens(name) * 2 + tokens(owner), "text": text})
    return out


def question_tokens(question):
    return [t for t in tokens(question) if t not in STOPWORDS]


# ------------------------------------------------------------
# Rankers
# ------------------------------------------------------------

class BM25Ranker:
    name = "bm25"

    def rank(self, question, schema):
        els = schema_elements(schema)
        if not els:
            return []
        q = question_tokens(question)
        bm = BM25([e["tokens"] for e in els])
        scored = [(e["id"], bm.score(q, i)) for i, e in enumerate(els)]
        # stable order: score desc, then labels before rels before props
        order = {e["id"]: i for i, e in enumerate(els)}
        return sorted(scored, key=lambda x: (-x[1], order[x[0]]))


class SemanticRanker:
    name = "semantic"

    def __init__(self, model_name=DEFAULT_EMBED_MODEL, encoder=None):
        """`encoder` (list[str] -> 2-D array of unit vectors) is only for tests."""
        self._encode = encoder
        self.model_name = model_name
        self._cache = {}

    def _encoder(self):
        if self._encode is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise SystemExit(
                    "The semantic retriever needs sentence-transformers.\n"
                    "Install it with:  uv sync --extra semantic") from e
            model = SentenceTransformer(self.model_name)
            self._encode = lambda texts: model.encode(
                texts, batch_size=64, normalize_embeddings=True,
                show_progress_bar=False)
        return self._encode

    def embed(self, texts):
        """Embed with a cache, so repeated schema elements are encoded once."""
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            for t, v in zip(missing, self._encoder()(missing)):
                self._cache[t] = [float(x) for x in v]
        return [self._cache[t] for t in texts]

    def rank(self, question, schema):
        els = schema_elements(schema)
        if not els:
            return []
        qv = self.embed([question])[0]
        evs = self.embed([e["text"] for e in els])
        scored = [(e["id"], sum(a * b for a, b in zip(qv, v))) for e, v in zip(els, evs)]
        order = {e["id"]: i for i, e in enumerate(els)}
        return sorted(scored, key=lambda x: (-x[1], order[x[0]]))


class HybridRanker:
    """Reciprocal-rank fusion: score = sum over rankers of 1 / (k + rank)."""
    name = "hybrid"

    def __init__(self, rankers, k=60):
        self.rankers, self.k = rankers, k

    def rank(self, question, schema):
        fused = Counter()
        first_seen = {}
        for r in self.rankers:
            for pos, (eid, _) in enumerate(r.rank(question, schema), start=1):
                fused[eid] += 1.0 / (self.k + pos)
                first_seen.setdefault(eid, len(first_seen))
        return sorted(fused.items(), key=lambda x: (-x[1], first_seen[x[0]]))


RETRIEVERS = ["bm25", "semantic", "hybrid"]
_shared = {}


def get_ranker(name, model_name=DEFAULT_EMBED_MODEL):
    """One shared instance per name, so the embedding cache is reused."""
    key = (name, model_name)
    if key not in _shared:
        if name == "bm25":
            _shared[key] = BM25Ranker()
        elif name == "semantic":
            _shared[key] = SemanticRanker(model_name)
        elif name == "hybrid":
            _shared[key] = HybridRanker([get_ranker("bm25"), get_ranker("semantic", model_name)])
        else:
            raise ValueError(f"unknown retriever {name!r}; choose from {RETRIEVERS}")
    return _shared[key]


# ------------------------------------------------------------
# Ranking metrics (Part 4.4)
# ------------------------------------------------------------

KS = (1, 3, 5, 10)


def ranking_metrics(ranked_ids, gold_ids, ks=KS):
    """
    Recall@k = share of the gold elements found in the top k.
    MRR      = 1 / rank of the first gold element (0 if none is ranked).
    """
    gold = set(gold_ids)
    if not gold:
        return None
    out = {}
    for k in ks:
        out[f"R@{k}"] = len(gold & set(ranked_ids[:k])) / len(gold)
    rr = 0.0
    for i, eid in enumerate(ranked_ids, start=1):
        if eid in gold:
            rr = 1.0 / i
            break
    out["MRR"] = rr
    # rank needed to recover the whole gold set (useful for picking k)
    pos = {eid: i for i, eid in enumerate(ranked_ids, start=1)}
    out["rank_all"] = max(pos.get(g, math.inf) for g in gold)
    return out
