#!/usr/bin/env python3
"""
Schema retriever: give the LLM only the part of the schema that the
question needs. (Roadmap Part 4, System 2 / "retrieved_schema")

Works on the MTQ schema JSON format:
    {"node_props": {Label: [{"property", "datatype"}, ...]},
     "rel_props":  {TYPE:  [{"property", "datatype"}, ...]},
     "relationships": [{"start", "type", "end"}, ...]}

Algorithm (first version, no embeddings needed):
    1. Lexical scoring (BM25) of every label and relationship type against
       the question. A label's "document" is its name plus its property
       names; a relationship's document is its type plus its endpoints.
    2. Keep the top-k labels and top-k relationship types (score > 0),
       plus the endpoints of the kept relationships.
    3. Topology step: connect the kept labels with shortest paths in the
       schema graph, adding any labels/relationships on those paths.
    4. Return a pruned schema JSON (same format) with only those elements.
       If nothing matched, fall back to the full schema.
"""

import json
import math
import re
from collections import Counter, deque


# ------------------------------------------------------------
# Tokenisation
# ------------------------------------------------------------

def _split_ident(name):
    """'BankAccount' -> ['bank', 'account'];  'HAS_GENRE' -> ['has', 'genre']."""
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name)
    return [t for t in re.split(r"[^A-Za-z0-9]+", name.lower()) if t]


def _stem(tok):
    """
    Tiny stemmer: maps singular/plural/verb forms to one key, e.g.
    movie/movies -> movi, genre/genres -> genr, category/categories -> categori,
    watch/watched/watches/watching -> watch, address/addresses -> address.
    """
    t = tok
    for suf in ("ing", "ed"):
        if len(t) > len(suf) + 2 and t.endswith(suf):
            t = t[: -len(suf)]
            break
    else:
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
    if len(t) > 3 and t.endswith("ie"):
        t = t[:-1]
    elif len(t) > 3 and t.endswith("y"):
        t = t[:-1] + "i"
    if len(t) > 3 and t.endswith("e"):
        t = t[:-1]
    return t


def tokens(text, drop_stopwords=False):
    words = _split_ident(text)
    if drop_stopwords:
        words = [w for w in words if w not in STOPWORDS]
    return [_stem(w) for w in words]


STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "by", "with",
    "what", "which", "who", "how", "many", "much", "are", "is", "was", "were",
    "do", "does", "did", "have", "has", "that", "this", "each", "all", "any",
    "list", "show", "find", "give", "return", "number", "count", "top", "most",
    "from", "at", "be", "their", "there", "than", "more", "less", "name",
}


# ------------------------------------------------------------
# BM25
# ------------------------------------------------------------

class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.docs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avg = (sum(self.lens) / len(self.lens)) if docs else 0
        self.k1, self.b = k1, b
        n = len(docs)
        df = Counter(t for d in self.docs for t in d)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query, i):
        d, L = self.docs[i], self.lens[i]
        s = 0.0
        for t in query:
            if t not in d:
                continue
            tf = d[t]
            s += self.idf[t] * tf * (self.k1 + 1) / (
                tf + self.k1 * (1 - self.b + self.b * L / (self.avg or 1)))
        return s


# ------------------------------------------------------------
# Retriever
# ------------------------------------------------------------

def _parse(schema):
    return json.loads(schema) if isinstance(schema, str) else schema


def _neighbours(rels):
    adj = {}
    for r in rels:
        adj.setdefault(r["start"], []).append((r["end"], r["type"]))
        adj.setdefault(r["end"], []).append((r["start"], r["type"]))
    return adj


def _shortest_path(adj, src, dst, max_hops=3):
    """BFS in the undirected schema graph. Returns list of (label, rel) steps."""
    if src == dst:
        return []
    prev = {src: None}
    dq = deque([(src, 0)])
    while dq:
        node, d = dq.popleft()
        if d >= max_hops:
            continue
        for nb, rtype in adj.get(node, ()):
            if nb in prev:
                continue
            prev[nb] = (node, rtype)
            if nb == dst:
                path = []
                cur = dst
                while prev[cur] is not None:
                    p, rt = prev[cur]
                    path.append((cur, rt))
                    cur = p
                return path[::-1]
            dq.append((nb, d + 1))
    return None


def _sem_tokens(semantics, section, name, use_descriptions=False):
    """Alias (and optionally description) tokens from the semantic layer."""
    if not semantics:
        return [], []
    entry = semantics.get(section, {}).get(name) or {}
    alias_toks = [t for a in entry.get("aliases", [])
                  for t in tokens(a, drop_stopwords=True)]
    desc_toks = (tokens(entry.get("description", ""), drop_stopwords=True)
                 if use_descriptions else [])
    return alias_toks, desc_toks


def retrieve(question, schema, k_labels=3, k_rels=3, prop_weight=0.5,
             semantics=None, use_descriptions=False):
    """
    Return (pruned_schema_dict, info) where info lists what was selected.

    semantics: optional dict from semantic_layer.py (aliases/descriptions);
               aliases count almost like the element's own name.
    """
    s = _parse(schema)
    node_props = s.get("node_props", {})
    rel_props = s.get("rel_props", {})
    rels = s.get("relationships", [])

    q = tokens(question, drop_stopwords=True)

    # ---- 1. score labels ----
    labels = list(node_props.keys() | {r["start"] for r in rels} | {r["end"] for r in rels})
    label_docs = []
    label_name_toks = {}
    for L in labels:
        name_toks = tokens(L)
        alias_toks, desc_toks = _sem_tokens(semantics, "labels", L, use_descriptions)
        prop_toks = []
        for p in node_props.get(L, []):
            prop_toks += tokens(p["property"])
            prop_toks += _sem_tokens(semantics, "properties", f"{L}.{p['property']}")[0]
        label_name_toks[L] = set(name_toks) | set(alias_toks)
        # repeat name tokens so a name hit outweighs a property hit
        label_docs.append(name_toks * 3 + alias_toks * 2 + desc_toks + prop_toks)
    bm_l = BM25(label_docs)
    lab_scores = {L: bm_l.score(q, i) for i, L in enumerate(labels)}
    # property hits only add a damped bonus (an alias hit counts as a name hit)
    for i, L in enumerate(labels):
        name_hit = any(t in q for t in label_name_toks[L])
        if not name_hit:
            lab_scores[L] *= prop_weight

    # ---- 2. score relationship types ----
    rtypes = sorted({r["type"] for r in rels})
    rel_docs = []
    for T in rtypes:
        alias_toks, desc_toks = _sem_tokens(semantics, "relationships", T, use_descriptions)
        toks = tokens(T) * 3 + alias_toks * 2 + desc_toks
        toks += [t for p in rel_props.get(T, []) for t in tokens(p["property"])]
        rel_docs.append(toks)
    bm_r = BM25(rel_docs) if rel_docs else None
    rel_scores = {T: bm_r.score(q, i) for i, T in enumerate(rtypes)} if bm_r else {}

    top_l = [L for L, sc in sorted(lab_scores.items(), key=lambda x: -x[1]) if sc > 0][:k_labels]
    top_r = [T for T, sc in sorted(rel_scores.items(), key=lambda x: -x[1]) if sc > 0][:k_rels]

    if not top_l and not top_r:
        return s, {"fallback": True, "labels": sorted(labels), "rels": rtypes,
                   "semantics": bool(semantics)}

    keep_l, keep_r = set(top_l), set(top_r)
    for r in rels:
        if r["type"] in keep_r:
            keep_l |= {r["start"], r["end"]}

    # ---- 3. topology: connect the kept labels ----
    adj = _neighbours(rels)
    seeds = sorted(keep_l)
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            path = _shortest_path(adj, seeds[i], seeds[j])
            if path:
                for lab, rt in path:
                    keep_l.add(lab)
                    keep_r.add(rt)

    # A single isolated label: include its direct relationships as context.
    if len(keep_l) == 1:
        (only,) = keep_l
        for nb, rt in adj.get(only, ()):
            keep_l.add(nb)
            keep_r.add(rt)

    # ---- 4. build pruned schema ----
    pruned = {
        "node_props": {L: v for L, v in node_props.items() if L in keep_l},
        "rel_props": {T: v for T, v in rel_props.items() if T in keep_r},
        "relationships": [
            r for r in rels
            if r["type"] in keep_r and r["start"] in keep_l and r["end"] in keep_l
        ],
    }
    return pruned, {
        "fallback": False,
        "semantics": bool(semantics),
        "labels": sorted(keep_l),
        "rels": sorted(keep_r),
        "label_scores": {L: round(lab_scores[L], 3) for L in top_l},
        "rel_scores": {T: round(rel_scores[T], 3) for T in top_r},
    }


def retrieve_json(question, schema, **kw):
    pruned, _ = retrieve(question, schema, **kw)
    return json.dumps(pruned)


if __name__ == "__main__":
    import argparse
    import pandas as pd

    p = argparse.ArgumentParser(description="Show what the retriever picks for a question")
    p.add_argument("--parquet", default="Benchmark/MTQ-Auto/test.parquet")
    p.add_argument("--data-source", default="er")
    p.add_argument("--row", type=int, default=0)
    p.add_argument("--question", help="override the question text")
    a = p.parse_args()

    df = pd.read_parquet(a.parquet)
    row = df[df["data_source"] == a.data_source].iloc[a.row]
    question = a.question or row["question"]
    pruned, info = retrieve(question, row["schema"])
    print("Question:", question)
    print("Gold    :", row["cypher"])
    print("Picked  :", json.dumps(info, indent=2))
    print(f"Context : {len(json.dumps(pruned))} chars vs full {len(row['schema'])} chars")
