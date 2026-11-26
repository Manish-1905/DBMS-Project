#!/usr/bin/env python3
"""
Benchmark the schema retrievers on their own, before any LLM is involved.
(Roadmap Part 4.4 and 4.5)

For every question:
    gold set      = labels / rel types / properties used by the gold Cypher
    ranked list   = the retriever's ranking of all schema elements
    retrieved set = labels / rel types / properties in the pruned schema
                    (ranking + topology step, what the LLM would see)

Two tables are printed per retriever:
    1. Pruned-schema metrics: recall per element kind, "all needed elements
       kept", fallback rate, context reduction.
    2. Ranking metrics: Recall@1/3/5/10 and MRR over the ranked elements.

With --distractors the same questions are re-run on schemas where a fixed
share of the elements are distractors (see distractors.py), giving the
"recall vs context size" curve of Part 4.5.

    uv run eval_retrieval.py --data-source er
    uv run eval_retrieval.py                                   # all graphs
    uv run eval_retrieval.py --retriever bm25 semantic hybrid  # compare retrievers
    uv run eval_retrieval.py --data-source er --distractors 0.4 0.6 0.8 0.95
    uv run eval_retrieval.py --parquet Benchmark/Text2Cypher-2024/test.parquet
"""

import argparse
import csv
import json
from collections import defaultdict

import pandas as pd

from distractors import build_pool, count_elements, irrelevant_share, schema_sets, with_distractors
from gold_schema import gold_schema_set
from retrievers import KS, RETRIEVERS, get_ranker, ranking_metrics
from schema_retriever import retrieve


def recall(gold, got):
    return 1.0 if not gold else len(gold & got) / len(gold)


def score_one(question, schema, gold, retriever, k_labels, k_rels, k_elements=6):
    """All metrics for one question on one schema."""
    full = schema_sets(schema)
    gold = {k: set(gold[k]) & full[k] for k in gold}  # drop parser noise
    pruned, info = retrieve(question, schema, k_labels=k_labels, k_rels=k_rels,
                            retriever=retriever, k_elements=k_elements)
    got = schema_sets(pruned)
    rec = {k: recall(gold[k], got[k]) for k in gold}
    full_json = schema if isinstance(schema, str) else json.dumps(schema)
    row = {
        "label_recall": rec["labels"],
        "rel_recall": rec["rels"],
        "prop_recall": rec["props"],
        "all_needed": all(v == 1.0 for v in rec.values()),
        "fallback": info["fallback"],
        "ctx_chars": len(json.dumps(pruned)),
        "full_chars": len(full_json),
        "n_elements": count_elements(schema),
        "missing": "; ".join(sorted(
            (gold["labels"] - got["labels"]) | (gold["rels"] - got["rels"])
            | (gold["props"] - got["props"]))),
    }
    ranked = [eid for eid, _ in get_ranker(retriever).rank(question, schema)]
    rm = ranking_metrics(ranked, gold["labels"] | gold["rels"] | gold["props"])
    row["has_gold"] = rm is not None
    for k in KS:
        row[f"R@{k}"] = rm[f"R@{k}"] if rm else None
    row["MRR"] = rm["MRR"] if rm else None
    row["rank_all"] = rm["rank_all"] if rm else None
    return row


def summarise_set(group):
    n = len(group)
    m = lambda k: sum(x[k] for x in group) / n
    reduction = 1 - sum(x["ctx_chars"] for x in group) / sum(x["full_chars"] for x in group)
    return [str(n), f"{m('label_recall'):.1%}", f"{m('rel_recall'):.1%}",
            f"{m('prop_recall'):.1%}", f"{m('all_needed'):.1%}",
            f"{m('fallback'):.1%}", f"{reduction:.1%}"]


def summarise_rank(group):
    g = [x for x in group if x["has_gold"]]
    if not g:
        return [str(len(group))] + ["-"] * (len(KS) + 1)
    m = lambda k: sum(x[k] for x in g) / len(g)
    return [str(len(g))] + [f"{m(f'R@{k}'):.1%}" for k in KS] + [f"{m('MRR'):.3f}"]


def print_table(title, header, rows):
    print(f"\n### {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(r) + " |")


SET_HEADER = ["group", "n", "label_recall", "rel_recall", "prop_recall",
              "full_gold_recall", "fallback", "context_reduction"]
RANK_HEADER = ["group", "n"] + [f"R@{k}" for k in KS] + ["MRR"]


def grouped(rows, fn, by_type=True):
    out = []
    by_ds, by_type_ = defaultdict(list), defaultdict(list)
    for x in rows:
        by_ds[x["data_source"]].append(x)
        by_type_[x["query_type"]].append(x)
    for k, g in sorted(by_ds.items()):
        out.append([f"graph={k}"] + fn(g))
    if by_type:
        for k, g in sorted(by_type_.items(), key=lambda kv: str(kv[0])):
            out.append([f"type={k}"] + fn(g))
    out.append(["ALL"] + fn(rows))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="Benchmark/MTQ-Auto/test.parquet")
    p.add_argument("--data-source", default=None)
    p.add_argument("--retriever", nargs="+", default=["bm25"],
                   choices=RETRIEVERS + ["all"],
                   help="one or more of bm25, semantic, hybrid (or 'all')")
    p.add_argument("--k-labels", type=int, default=3)
    p.add_argument("--k-rels", type=int, default=3)
    p.add_argument("--k-elements", type=int, default=6,
                   help="semantic/hybrid: how many top-ranked elements seed the topology step")
    p.add_argument("--distractors", nargs="+", type=float, default=None,
                   help="distractor shares to test, e.g. 0.4 0.6 0.8 0.95")
    p.add_argument("--pool-parquet", default="Benchmark/MTQ-Auto/test.parquet",
                   help="where distractor labels/relationships are borrowed from")
    p.add_argument("--sample", type=int, default=None,
                   help="use a fixed random sample of N questions (faster)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", help="write per-question rows here")
    a = p.parse_args()
    retrievers = RETRIEVERS if "all" in a.retriever else list(dict.fromkeys(a.retriever))

    df = pd.read_parquet(a.parquet)
    if a.data_source:
        df = df[df["data_source"] == a.data_source]
    if a.sample and a.sample < len(df):
        df = df.sample(n=a.sample, random_state=a.seed)
    if "query_type" not in df.columns:
        df = df.assign(query_type=None)

    pools = {}
    if a.distractors:
        pool_df = pd.read_parquet(a.pool_parquet)
        all_schemas = pool_df.drop_duplicates("data_source")[["data_source", "schema"]]

        def pool_for(ds):
            if ds not in pools:
                pools[ds] = build_pool(all_schemas[all_schemas.data_source != ds].schema)
            return pools[ds]

    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        gold = gold_schema_set(r["cypher"])
        variants = [("real", r["schema"], irrelevant_share(r["schema"], gold))]
        for level in a.distractors or []:
            s, got = with_distractors(r["schema"], gold, level, pool_for(r["data_source"]),
                                      seed=a.seed * 100003 + i)
            variants.append((f"{level:.0%}", s, got))
        for name in retrievers:
            for level, schema, achieved in variants:
                row = score_one(r["question"], schema, gold, name, a.k_labels, a.k_rels,
                                a.k_elements)
                row.update({"retriever": name, "distractors": level,
                            "irrelevant_share": achieved,
                            "data_source": r["data_source"],
                            "query_type": r["query_type"], "question": r["question"]})
                rows.append(row)

    for name in retrievers:
        real = [x for x in rows if x["retriever"] == name and x["distractors"] == "real"]
        print_table(f"{name}: pruned schema ("
                    + (f"k_labels={a.k_labels}, k_rels={a.k_rels}" if name == "bm25"
                       else f"k_elements={a.k_elements}") + ")",
                    SET_HEADER, grouped(real, summarise_set))
        print_table(f"{name}: ranking", RANK_HEADER, grouped(real, summarise_rank))

    if len(retrievers) > 1:
        comp = []
        for name in retrievers:
            g = [x for x in rows if x["retriever"] == name and x["distractors"] == "real"]
            s, rk = summarise_set(g), summarise_rank(g)
            comp.append([name, s[0], s[4], s[6]] + rk[1:])
        print_table("Retriever comparison (all questions)",
                    ["retriever", "n", "full_gold_recall", "context_reduction"]
                    + [f"R@{k}" for k in KS] + ["MRR"], comp)

    if a.distractors:
        table = []
        for name in retrievers:
            for level in ["real"] + [f"{x:.0%}" for x in a.distractors]:
                g = [x for x in rows if x["retriever"] == name and x["distractors"] == level]
                if not g:
                    continue
                shares = [x["irrelevant_share"] for x in g if x["irrelevant_share"] is not None]
                achieved = f"{sum(shares) / len(shares):.1%}" if shares else "-"
                elems = sum(x["n_elements"] for x in g) / len(g)
                s, rk = summarise_set(g), summarise_rank(g)
                table.append([name, level, achieved, f"{elems:.0f}", s[4], s[6]]
                             + rk[1:])
        print_table("Controlled distractors (Part 4.5)",
                    ["retriever", "target", "achieved_irrelevant", "avg_elements",
                     "full_gold_recall", "context_reduction"]
                    + [f"R@{k}" for k in KS] + ["MRR"], table)

    if a.csv and rows:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            fields = ["retriever", "distractors", "irrelevant_share", "data_source",
                      "query_type", "question"]
            fields += [k for k in rows[0] if k not in fields]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nPer-question rows written to {a.csv}")


if __name__ == "__main__":
    main()
