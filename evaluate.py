#!/usr/bin/env python3
"""
Compute benchmark metrics from the SQLite file written by main.py.

Usage:
    python evaluate.py --sqlite cypher_benchmark.sqlite
    python evaluate.py --sqlite cypher_benchmark.sqlite --rows-csv per_question.csv
    python evaluate.py --sqlite cypher_benchmark.sqlite --by-type   # + per query_type table

Metrics (one row per run / mode / model / dataset):
    runnable          LLM query executed without error
    non_null          executed AND returned at least one non-null value
    result_correct    LLM result == gold result (column names ignored;
                      row order ignored unless the gold query has ORDER BY)
    exact_match       query text identical after normalising whitespace/case
    avg_llm_exec_ms   mean Neo4j time for LLM queries that ran
    avg_gold_exec_ms  mean Neo4j time for gold queries
    avg_gen_s         mean LLM generation time
"""

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict


def normalise_value(v):
    """Make values comparable: round floats, sort lists inside cells."""
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, list):
        return [normalise_value(x) for x in v]
    if isinstance(v, dict):
        return {k: normalise_value(x) for k, x in sorted(v.items())}
    return v


def row_key(row):
    """
    A row is a dict like {"name": "Tom", "count": 3}. The LLM may use
    different column aliases than the gold query, so compare the
    *values* only, as a sorted multiset.
    """
    vals = [json.dumps(normalise_value(v), sort_keys=True, default=str)
            for v in row.values()]
    return tuple(sorted(vals))


def results_match(llm_rows, gold_rows, ordered):
    if llm_rows is None or gold_rows is None:
        return False
    if len(llm_rows) != len(gold_rows):
        return False
    llm_keys = [row_key(r) for r in llm_rows]
    gold_keys = [row_key(r) for r in gold_rows]
    if ordered:
        return llm_keys == gold_keys
    return Counter(llm_keys) == Counter(gold_keys)


def has_non_null(rows):
    if not rows:
        return False
    return any(v is not None for r in rows for v in r.values())


def normalise_query(q):
    if not q:
        return ""
    q = q.strip().rstrip(";")
    return re.sub(r"\s+", " ", q).lower()


def load(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cols = {r[1] for r in conn.execute("PRAGMA table_info(benchmark_results)")}
    rows = [dict(r) for r in conn.execute("SELECT * FROM benchmark_results")]
    for r in rows:  # older files have no mode/run_id columns
        for c in ("mode", "prompt_version", "run_id"):
            if c not in cols or r.get(c) is None:
                r[c] = "legacy"
    return rows


def evaluate_row(r):
    llm = json.loads(r["LLM_cypher_result"] or "null") or {}
    gold = json.loads(r["benchmark_cypher_result"] or "null") or {}
    llm_ok = bool(llm.get("success"))
    gold_ok = bool(gold.get("success"))
    ordered = "order by" in normalise_query(r["benchmark_cypher"])
    return {
        "id": r["id"],
        "run_id": r["run_id"],
        "mode": r["mode"],
        "model": r["LLM_model"],
        "dataset": r["dataset"],
        "query_type": r.get("query_type") or "unknown",
        "question": r["question"],
        "gold_ok": gold_ok,
        "runnable": llm_ok,
        "non_null": llm_ok and has_non_null(llm.get("result")),
        "result_correct": llm_ok and gold_ok and results_match(
            llm.get("result"), gold.get("result"), ordered),
        "exact_match": normalise_query(r["LLM_cypher"])
                       == normalise_query(r["benchmark_cypher"]),
        "llm_exec_s": r["LLM_cypher_latency"] if llm_ok else None,
        "gold_exec_s": r["benchmark_cypher_latency"] if gold_ok else None,
        "gen_s": r["time_to_generate"],
        "llm_error": llm.get("error"),
        "llm_cypher": r["LLM_cypher"],
        "gold_cypher": r["benchmark_cypher"],
    }


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, scale=1, digits=2, pct=False):
    if x is None:
        return "-"
    return f"{x * 100:.0f}%" if pct else f"{x * scale:.{digits}f}"


def print_table(rows, keys):
    """One markdown row per distinct combination of `keys`."""
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(str(r[k]) for k in keys)].append(r)

    header = keys + ["n", "gold_ok", "runnable", "non_null",
                     "result_correct", "exact_match", "avg_llm_exec_ms",
                     "avg_gold_exec_ms", "avg_gen_s"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for key, g in sorted(groups.items()):
        n = len(g)
        rate = lambda k: sum(r[k] for r in g) / n
        print("| " + " | ".join(list(key) + [
            str(n),
            fmt(rate("gold_ok"), pct=True),
            fmt(rate("runnable"), pct=True),
            fmt(rate("non_null"), pct=True),
            fmt(rate("result_correct"), pct=True),
            fmt(rate("exact_match"), pct=True),
            fmt(mean(r["llm_exec_s"] for r in g), 1000, 3),
            fmt(mean(r["gold_exec_s"] for r in g), 1000, 3),
            fmt(mean(r["gen_s"] for r in g)),
        ]) + " |")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sqlite", default="cypher_benchmark.sqlite")
    p.add_argument("--rows-csv", help="Also write per-question results here")
    p.add_argument("--by-type", action="store_true",
                   help="Also break results down by MTQ query_type")
    args = p.parse_args()

    rows = [evaluate_row(r) for r in load(args.sqlite)]
    if not rows:
        print("No rows found.")
        return

    print_table(rows, ["run_id", "mode", "model", "dataset"])
    if args.by_type:
        print()
        print_table(rows, ["mode", "model", "dataset", "query_type"])

    bad_gold = [r for r in rows if not r["gold_ok"]]
    if bad_gold:
        print(f"\nWARNING: {len(bad_gold)} gold queries failed. Is the right "
              "graph loaded in Neo4j? Those questions can never be 'correct'.")

    if args.rows_csv:
        with open(args.rows_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nPer-question results written to {args.rows_csv}")


if __name__ == "__main__":
    main()
