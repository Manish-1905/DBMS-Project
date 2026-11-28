import argparse
import csv
import json
from collections import defaultdict

import pandas as pd

from gold_schema import gold_schema_set
from schema_retriever import METHODS, rank_elements, retrieve
from semantic_layer import load_semantics


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


def ranked_metrics(gold, ranked, prefix, ks=(1, 3, 5, 10)):
    out = {}
    for k in ks:
        out[f"{prefix}_R@{k}"] = 1.0 if not gold else len(gold & set(ranked[:k])) / len(gold)
    if not gold:
        out[f"{prefix}_MRR"] = 1.0
    else:
        first = next((i for i, x in enumerate(ranked, 1) if x in gold), None)
        out[f"{prefix}_MRR"] = 1.0 / first if first else 0.0
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="Benchmark/MTQ-Auto/test.parquet")
    p.add_argument("--data-source", default=None)
    p.add_argument("--k-labels", type=int, default=3)
    p.add_argument("--k-rels", type=int, default=3)
    p.add_argument("--semantics-dir", default=None,
                   help="use semantics/<data_source>.json aliases (e.g. semantics)")
    p.add_argument("--method", choices=METHODS, default="bm25",
                   help="bm25 (lexical), embed (embeddings) or hybrid (both)")
    p.add_argument("--embed-margin", type=float, default=0.1,
                   help="embed/hybrid: keep elements within this cosine of the best")
    p.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5",
                   help="embedding model for embed/hybrid ('hash' = offline stand-in)")
    p.add_argument("--use-descriptions", action="store_true",
                   help="also match words from the descriptions")
    p.add_argument("--csv", help="write per-question rows here")
    a = p.parse_args()

    df = pd.read_parquet(a.parquet)
    if a.data_source:
        df = df[df["data_source"] == a.data_source]

    embedder = None
    if a.method != "bm25":
        from embedder import get_embedder
        embedder = get_embedder(a.embed_model)
    sem_cache = {}
    rows = []
    for _, r in df.iterrows():
        ds = r["data_source"]
        if ds not in sem_cache:
            sem_cache[ds] = load_semantics(ds, a.semantics_dir)
        full = schema_sets(r["schema"])
        gold = gold_schema_set(r["cypher"])
        gold = {k: gold[k] & full[k] for k in gold}
        pruned, info = retrieve(r["question"], r["schema"],
                                k_labels=a.k_labels, k_rels=a.k_rels,
                                semantics=sem_cache[ds],
                                use_descriptions=a.use_descriptions,
                                method=a.method, embedder=embedder,
                                embed_margin=a.embed_margin)
        rl, rr = rank_elements(r["question"], r["schema"], method=a.method,
                               embedder=embedder, semantics=sem_cache[ds],
                               use_descriptions=a.use_descriptions)
        ranked = {**ranked_metrics(gold["labels"], rl, "label"),
                  **ranked_metrics(gold["rels"], rr, "rel")}
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
            "label_R@1": ranked["label_R@1"], "label_R@3": ranked["label_R@3"],
            "label_MRR": ranked["label_MRR"], "rel_R@1": ranked["rel_R@1"],
            "rel_MRR": ranked["rel_MRR"],
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
                f"{m('fallback'):.1%}", f"{reduction:.1%}",
                f"{m('label_R@1'):.1%}", f"{m('label_R@3'):.1%}", f"{m('label_MRR'):.3f}",
                f"{m('rel_R@1'):.1%}", f"{m('rel_MRR'):.3f}"]

    header = ["group", "n", "label_recall", "rel_recall", "prop_recall",
              "full_gold_recall", "fallback", "context_reduction",
              "label_R@1", "label_R@3", "label_MRR", "rel_R@1", "rel_MRR"]
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
    used = sorted(ds for ds, sem in sem_cache.items() if sem)
    print(f"\nSemantic layer used for: {', '.join(used) if used else 'none'}")

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nPer-question rows written to {a.csv}")


if __name__ == "__main__":
    main()
