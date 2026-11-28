import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict

import pandas as pd

from cypher_guard import TAXONOMY, classify_db_error, policy_check, schema_check, value_check
from evaluate import evaluate_row, load
from gold_schema import gold_literals, gold_schema_set
from query_ir import SchemaInfo

ORDER = [f"F{i}" for i in range(1, 15)]
_AGG = re.compile(r"\b(count|sum|avg|min|max|collect)\s*\(\s*(distinct\s+)?", re.I)
_TEMPORAL = re.compile(r"\b(date|datetime|localdatetime|duration|time)\s*\(|"
                       r"\.(year|month|day)\b|\byear\b", re.I)
_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")


def labels_for(row, sch, catalog):
    cy, gold = row["llm_cypher"] or "", row["gold_cypher"] or ""
    codes = []
    if not cy.strip():
        return ["F1"], "no query generated"
    if row["llm_error"]:
        codes.append(classify_db_error(row["llm_error"]))
    codes += [f["code"] for f in policy_check(cy) + schema_check(cy, sch)]
    if catalog:
        codes += [f["code"] for f in value_check(cy, catalog)]
    g_vals, c_vals = gold_literals(gold), gold_literals(cy)
    g_by, c_by = defaultdict(set), defaultdict(set)
    for p, v in g_vals:
        g_by[p].add(v)
    for p, v in c_vals:
        c_by[p].add(v)
    for p in g_by:
        if p in c_by and c_by[p] != g_by[p] and not any(c in ("F7", "F8") for c in codes):
            high = p.endswith(("title", "Name", "name", "email", "address", "phone"))
            codes.append("F7" if high and not p.endswith("Genre.name") else "F8")
    if row["runnable"]:
        g, c = gold_schema_set(gold), gold_schema_set(cy)
        if g["labels"] != c["labels"] or g["rels"] != c["rels"]:
            codes.append("F9")
        gagg = sorted((m.group(1).lower(), bool(m.group(2))) for m in _AGG.finditer(gold))
        cagg = sorted((m.group(1).lower(), bool(m.group(2))) for m in _AGG.finditer(cy))
        if gagg != cagg:
            codes.append("F10")
        if _TEMPORAL.search(gold):
            gn = sorted(_NUM.findall(_TEMPORAL.sub(" ", gold)))
            cn = sorted(_NUM.findall(_TEMPORAL.sub(" ", cy)))
            if not _TEMPORAL.search(cy) or gn != cn:
                codes.append("F11")
        if not codes:
            codes.append("F14")
    codes = sorted(set(codes), key=ORDER.index)
    return codes, ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sqlite", required=True)
    p.add_argument("--parquet", default="Benchmark/MTQ-Auto/test-split.parquet")
    p.add_argument("--values", default="values/er.json")
    p.add_argument("--csv", help="write one row per wrong answer")
    a = p.parse_args()

    schemas = pd.read_parquet(a.parquet).groupby("data_source")["schema"].first().to_dict()
    sch_cache = {}
    catalog = json.load(open(a.values)) if a.values and os.path.exists(a.values) else None
    if not catalog:
        print(f"(no value catalog at {a.values}: F7/F8 come only from differences "
              f"with the gold values)")
    rows = [evaluate_row(r) for r in load(a.sqlite)]
    per_run = defaultdict(list)
    out_rows = []
    for r in rows:
        if r["result_correct"] or not r["gold_ok"]:
            continue
        if r["dataset"] not in sch_cache:
            sch_cache[r["dataset"]] = SchemaInfo(schemas[r["dataset"]])
        codes, note = labels_for(r, sch_cache[r["dataset"]], catalog)
        per_run[r["run_id"]].append(codes)
        out_rows.append({"run_id": r["run_id"], "query_type": r["query_type"],
                         "primary": codes[0], "codes": " ".join(codes),
                         "question": r["question"], "llm_cypher": r["llm_cypher"],
                         "gold_cypher": r["gold_cypher"], "llm_error": r["llm_error"]})

    totals = Counter(r["run_id"] for r in rows if r["gold_ok"])
    for run, lists in sorted(per_run.items()):
        n_wrong = len(lists)
        any_c = Counter(c for codes in lists for c in codes)
        prim = Counter(codes[0] for codes in lists)
        print(f"\n## {run}: {n_wrong} wrong of {totals[run]}")
        print("| code | failure | primary | % of wrong | any label | % of wrong |")
        print("|---|---|---|---|---|---|")
        for c in ORDER:
            if any_c[c]:
                print(f"| {c} | {TAXONOMY[c]} | {prim[c]} | {prim[c] / n_wrong:.0%} "
                      f"| {any_c[c]} | {any_c[c] / n_wrong:.0%} |")
    if a.csv and out_rows:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nwritten to {a.csv}")


if __name__ == "__main__":
    main()
