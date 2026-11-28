import argparse
import csv
from collections import defaultdict

import pandas as pd

from gold_schema import gold_literals
from grounding import Grounder
from schema_retriever import retrieve
from semantic_layer import load_semantics
from value_index import load_catalog


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--values", default="values/er.json", help="catalog from value_index.py")
    p.add_argument("--parquet", default="Benchmark/MTQ-Auto/value_variants_er.parquet")
    p.add_argument("--data-source", default="er")
    p.add_argument("--semantics-dir", default="semantics")
    p.add_argument("--no-retrieval", action="store_true",
                   help="don't use the retrieved schema as context for grounding")
    p.add_argument("--csv", help="write per-question rows here")
    a = p.parse_args()

    cat = load_catalog(a.values)
    grounder = Grounder(cat)
    known = {(k, v) for k, pv in cat["properties"].items() for v in pv.get("values", {})}
    df = pd.read_parquet(a.parquet)
    df = df[df["data_source"] == a.data_source].reset_index(drop=True)
    if "variant" not in df.columns:
        df["variant"] = "original"
    sem = load_semantics(a.data_source, a.semantics_dir)
    print(f"catalog state_version {cat.get('state_version')}; {len(df)} questions")

    rows = []
    for _, r in df.iterrows():
        gold = gold_literals(r["cypher"])
        if "target_property" in df.columns:
            target = (r["target_property"], r["target_value"])
        else:
            target = next(iter(sorted(gold)), None)
        if not target:
            continue
        labels = None
        if not a.no_retrieval:
            labels = set(retrieve(r["question"], r["schema"], semantics=sem)[1]["labels"])
        res = grounder.ground(r["question"], schema_labels=labels)
        got = [(g["property"], g["value"]) for g in res]
        same_prop = [x for x in got if x[0] == target[0]]
        rows.append({
            "variant": r["variant"],
            "target_property": target[0],
            "question": r["question"],
            "target_value": target[1],
            "in_catalog": target in known,
            "verbatim": target[1] in r["question"],
            "target_R@1": bool(same_prop) and same_prop[0] == target,
            "target_R@5": target in got[:5],
            "all_resolved": gold <= set(got),
            "wrong_value": bool(same_prop) and target not in same_prop,
            "extra": len(set(got) - gold),
            "grounded": "; ".join(f"{p}={v!r}" for p, v in got),
        })

    def table(key):
        groups = defaultdict(list)
        for x in rows:
            groups[x[key]].append(x)
        header = [key, "n", "in_catalog", "verbatim", "target_R@1", "target_R@5",
                  "all_resolved", "wrong_value", "extra"]
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        order = ["original", "lowercase", "unquoted", "typo", "partial", "alias"]
        for k in sorted(groups, key=lambda k: (order.index(k) if k in order else 99, str(k))):
            g = groups[k]
            m = lambda c: sum(x[c] for x in g) / len(g)
            print("| " + " | ".join([str(k), str(len(g))] + [
                f"{m(c):.1%}" for c in ("in_catalog", "verbatim", "target_R@1",
                                        "target_R@5", "all_resolved", "wrong_value")]
                + [f"{m('extra'):.2f}"]) + " |")

    print()
    table("variant")
    print()
    table("target_property")
    missing = sum(not x["in_catalog"] for x in rows)
    if missing:
        print(f"\nWARNING: {missing} target values are not in the catalog. Is the "
              f"catalog from the same database the gold queries run on?")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nper-question rows written to {a.csv}")


if __name__ == "__main__":
    main()
