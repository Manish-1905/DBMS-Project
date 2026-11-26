#!/usr/bin/env python3
"""
Distractor experiment for the schema retriever. (Roadmap Part 4.4 + 4.5)

For every noise level (share of the schema that is fake) and every retrieval
method, over all questions:

    gold_kept      % questions where every label/rel/property the gold query
                   uses survives retrieval   (higher = better)
    label_R@1/@3   share of gold labels among the top-1 / top-3 ranked labels
    label_MRR      mean reciprocal rank of the first gold label
    rel_R@3, rel_MRR   same for relationship types
    leak           % of retrieved labels/rels that are distractors (lower = better)
    ctx_chars      average schema characters sent to the LLM
    vs_full        ctx_chars as % of the full noisy schema

No LLM and no Neo4j needed. Embedding methods download the model once.

    uv run eval_distractors.py                                   # bm25, bm25+sem
    uv run eval_distractors.py --methods bm25 bm25+sem embed hybrid
    uv run eval_distractors.py --embed-model hash                # quick, no download
    uv run eval_distractors.py --out results/distractors_er.csv
"""

import argparse
import csv
import json
from collections import defaultdict

import pandas as pd

from distractors import LEVELS, add_distractors
from eval_retrieval import ranked_metrics, schema_sets
from gold_schema import gold_schema_set
from schema_retriever import rank_elements, retrieve
from semantic_layer import load_semantics, merge, rule_semantics

ALL_METHODS = ("bm25", "bm25+sem", "embed", "embed+sem", "hybrid", "hybrid+sem")


def parse_method(m):
    base, _, sem = m.partition("+")
    if base not in ("bm25", "embed", "hybrid") or sem not in ("", "sem"):
        raise SystemExit(f"unknown method {m!r}; choose from {ALL_METHODS}")
    return base, bool(sem)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="Benchmark/MTQ-Auto/test-split.parquet")
    p.add_argument("--data-source", default="er")
    p.add_argument("--levels", type=float, nargs="+", default=list(LEVELS))
    p.add_argument("--methods", nargs="+", default=["bm25", "bm25+sem"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k-labels", type=int, default=3)
    p.add_argument("--k-rels", type=int, default=3)
    p.add_argument("--semantics-dir", default="semantics")
    p.add_argument("--embed-margin", type=float, default=0.1,
                   help="embed/hybrid: keep elements within this cosine of the best")
    p.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5",
                   help="'hash' = dependency-free stand-in (not semantic)")
    p.add_argument("--limit", type=int, help="only the first N questions")
    p.add_argument("--out", help="write the summary table as CSV")
    a = p.parse_args()

    methods = [parse_method(m) for m in a.methods]
    df = pd.read_parquet(a.parquet)
    df = df[df["data_source"] == a.data_source].reset_index(drop=True)
    if a.limit:
        df = df.head(a.limit)
    base_schema = json.loads(df["schema"].iloc[0])
    real_sem = load_semantics(a.data_source, a.semantics_dir)
    if any(sem for _, sem in methods) and not real_sem:
        print(f"note: no {a.semantics_dir}/{a.data_source}.json; '+sem' methods = plain")

    embedder = None
    if any(b in ("embed", "hybrid") for b, _ in methods):
        from embedder import get_embedder
        embedder = get_embedder(a.embed_model)
        print(f"embedding model: {embedder.model_name}")

    golds = []
    full_sets = schema_sets(base_schema)
    for _, r in df.iterrows():
        g = gold_schema_set(r["cypher"])
        golds.append({k: g[k] & full_sets[k] for k in g})

    rows = []
    for share in a.levels:
        noisy, info = add_distractors(base_schema, share, a.seed)
        distract = set(info["labels"]) | set(info["rels"])
        full_chars = len(json.dumps(noisy))
        # distractors get the same rule-based aliases as real elements; LLM
        # aliases exist only for the real ones (see README caveat)
        sem = merge(rule_semantics(noisy), real_sem) if real_sem else None

        for base, use_sem in methods:
            name = base + ("+sem" if use_sem else "")
            acc = defaultdict(float)
            for (_, r), gold in zip(df.iterrows(), golds):
                kw = dict(method=base, embedder=embedder,
                          semantics=sem if use_sem else None)
                pruned, rinfo = retrieve(r["question"], noisy, k_labels=a.k_labels,
                                         k_rels=a.k_rels, embed_margin=a.embed_margin, **kw)
                got = schema_sets(pruned)
                acc["gold_kept"] += all(gold[k] <= got[k] for k in gold)
                picked = set(rinfo["labels"]) | set(rinfo["rels"])
                acc["leak"] += len(picked & distract) / max(len(picked), 1)
                acc["ctx"] += len(json.dumps(pruned))
                rl, rr = rank_elements(r["question"], noisy, **kw)
                for k, v in ranked_metrics(gold["labels"], rl, "label").items():
                    acc[k] += v
                for k, v in ranked_metrics(gold["rels"], rr, "rel").items():
                    acc[k] += v
            n = len(df)
            rows.append({
                "share": share, "n_distractors": info["n_distractors"], "method": name,
                "gold_kept": acc["gold_kept"] / n,
                "label_R@1": acc["label_R@1"] / n, "label_R@3": acc["label_R@3"] / n,
                "label_MRR": acc["label_MRR"] / n,
                "rel_R@3": acc["rel_R@3"] / n, "rel_MRR": acc["rel_MRR"] / n,
                "leak": acc["leak"] / n,
                "ctx_chars": acc["ctx"] / n, "full_chars": full_chars,
            })
            print(f"  share={share:<4} {name:<11} done", flush=True)

    pct = lambda x: f"{x:.1%}"
    header = ["share", "distractors", "method", "gold_kept", "label_R@1", "label_R@3",
              "label_MRR", "rel_R@3", "rel_MRR", "leak", "ctx_chars", "vs_full"]
    print()
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for x in rows:
        print("| " + " | ".join([
            f"{x['share']:.0%}", str(x["n_distractors"]), x["method"],
            pct(x["gold_kept"]), pct(x["label_R@1"]), pct(x["label_R@3"]),
            f"{x['label_MRR']:.3f}", pct(x["rel_R@3"]), f"{x['rel_MRR']:.3f}",
            pct(x["leak"]), f"{x['ctx_chars']:.0f}",
            pct(x["ctx_chars"] / x["full_chars"]),
        ]) + " |")

    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
