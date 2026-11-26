#!/usr/bin/env python3
"""
Benchmark the schema retriever on its own, before any LLM is involved.
(Roadmap Part 4.4)

For every question:
    gold set      = labels / rel types / properties used by the gold Cypher
    retrieved set = labels / rel types / properties in the pruned schema
and we report recall (did we keep everything needed?) and how much
smaller the context got.

    uv run eval_retrieval.py --data-source er
    uv run eval_retrieval.py                       # all graphs
    uv run eval_retrieval.py --k-labels 5 --csv retrieval_rows.csv
"""

import argparse
import csv
import json
from collections import defaultdict

import pandas as pd

from gold_schema import gold_schema_set
from schema_retriever import retrieve


def schema_sets(schema):
    s = json.loads(schema) if isinstance(schema, str) else schema
    labels = set(s.get("node_props", {})) | {r["start"] for r in s.get("relationships", [])} \
        | {r["end"] for r in s.get("relationships", [])}
    rels = {r["type"] for r in s.get("relationships", [])} | set(s.get("rel_props", {}))
    props = {f"{L}.{p['property']}" for L, ps in s.get("node_props", {}).items() for p in ps}
    props |= {f"{T}.{p['property']}" for T, ps in s.get("rel_props", {}).items() for p in ps}
    return {"labels": labels, "rels": rels, "props": props}


def recall(gold, got):
    return 1.0 if not gold else len(gold & got) / len(gold)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="Benchmark/MTQ-Auto/test.parquet")
    p.add_argument("--data-source", default=None)
    p.add_argument("--k-labels", type=int, default=3)
    p.add_argument("--k-rels", type=int, default=3)
    p.add_argument("--csv", help="write per-question rows here")
    a = p.parse_args()

    df = pd.read_parquet(a.parquet)
    if a.data_source:
        df = df[df["data_source"] == a.data_source]

    rows = []
    for _, r in df.iterrows():
        full = schema_sets(r["schema"])
        gold = gold_schema_set(r["cypher"])
        # only score elements that really exist in the schema (drops parser noise)
        gold = {k: gold[k] & full[k] for k in gold}
        pruned, info = retrieve(r["question"], r["schema"],
                                k_labels=a.k_labels, k_rels=a.k_rels)
        got = schema_sets(pruned)
        rec = {k: recall(gold[k], got[k]) for k in gold}
        rows.append({
            "data_source": r["data_source"],
            "query_type": r.get("query_type"),
            "question": r["question"],
            "label_recall": rec["labels"],
            "rel_recall": rec["rels"],
            "prop_recall": rec["props"],
            "all_needed": all(v == 1.0 for v in rec.values()),
            "fallback": info["fallback"],
            "ctx_chars": len(json.dumps(pruned)),
            "full_chars": len(r["schema"]),
            "missing": "; ".join(sorted(
                (gold["labels"] - got["labels"]) | (gold["rels"] - got["rels"])
                | (gold["props"] - got["props"]))),
        })

    def summarise(group):
        n = len(group)
        m = lambda k: sum(x[k] for x in group) / n
        reduction = 1 - sum(x["ctx_chars"] for x in group) / sum(x["full_chars"] for x in group)
        return [str(n), f"{m('label_recall'):.1%}", f"{m('rel_recall'):.1%}",
                f"{m('prop_recall'):.1%}", f"{m('all_needed'):.1%}",
                f"{m('fallback'):.1%}", f"{reduction:.1%}"]

    header = ["group", "n", "label_recall", "rel_recall", "prop_recall",
              "full_gold_recall", "fallback", "context_reduction"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    by_ds, by_type = defaultdict(list), defaultdict(list)
    for x in rows:
        by_ds[x["data_source"]].append(x)
        by_type[x["query_type"]].append(x)
    for k, g in sorted(by_ds.items()):
        print("| " + " | ".join([f"graph={k}"] + summarise(g)) + " |")
    for k, g in sorted(by_type.items(), key=lambda kv: str(kv[0])):
        print("| " + " | ".join([f"type={k}"] + summarise(g)) + " |")
    print("| " + " | ".join(["ALL"] + summarise(rows)) + " |")

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nPer-question rows written to {a.csv}")


if __name__ == "__main__":
    main()
