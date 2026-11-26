#!/usr/bin/env python3
"""
Compute benchmark metrics from the SQLite file written by main.py.

Usage:
    python evaluate.py --sqlite cypher_benchmark.sqlite
    python evaluate.py --sqlite cypher_benchmark.sqlite --rows-csv per_question.csv
    python evaluate.py --sqlite cypher_benchmark.sqlite --by-type   # + per query_type table
    python evaluate.py --sqlite noise.sqlite --compare RUN_A RUN_B  # paired comparison

Metrics (one row per run / mode / model / dataset):
    runnable          LLM query executed without error
    empty             executed but returned no rows (e.g. wrong value)
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
import math
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


def query_features(cypher):
    """Roadmap Part 6 breakdown of the GOLD query: hops and clause types."""
    u = (cypher or "").upper()
    hops = len(re.findall(r"-\[[^\]]*\]-|-->|<--", cypher or ""))
    return {
        "hops": "0" if hops == 0 else "1" if hops == 1 else "2" if hops == 2 else "3+",
        "filter": bool(re.search(r"\bWHERE\b|\{\s*\w+\s*:", cypher or "")),
        "aggregation": bool(re.search(r"\b(COUNT|SUM|AVG|MIN|MAX|COLLECT)\s*\(", u)),
        "sorting": "ORDER BY" in u or bool(re.search(r"\bLIMIT\b", u)),
        "multi_stage": bool(re.search(r"\bWITH\b", u)),
    }


def evaluate_row(r):
    llm = json.loads(r["LLM_cypher_result"] or "null") or {}
    gold = json.loads(r["benchmark_cypher_result"] or "null") or {}
    llm_ok = bool(llm.get("success"))
    gold_ok = bool(gold.get("success"))
    ordered = "order by" in normalise_query(r["benchmark_cypher"])
    ctx = {}
    try:
        ctx = json.loads(r.get("retrieved_context") or "null") or {}
    except (TypeError, ValueError):
        pass
    ir = ctx.get("ir") or {}
    try:
        attempts = json.loads(r.get("attempts") or "null") or []
    except (TypeError, ValueError):
        attempts = []
    att = {}
    if attempts:
        gold_rows = gold.get("result")
        ok_k = [bool(x.get("success") and gold_ok and results_match(
            x.get("result"), gold_rows, ordered)) for x in attempts]
        for k in range(3):
            att[f"att_correct_{k}"] = ok_k[min(k, len(ok_k) - 1)]
        att["n_attempts"] = len(attempts)
        att["first_failure"] = " ".join(sorted({f["code"] for f in attempts[0].get("failures") or []})) \
            or ("EMPTY" if attempts[0].get("empty") else "-")
        att["blocked"] = attempts[-1].get("stage") in ("static", "explain")
    return {**att,
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
        # ran fine but returned no rows: typical of a wrong value ('female'
        # instead of 'Female') or a non-existent label
        "empty": llm_ok and not llm.get("result"),
        "result_correct": llm_ok and gold_ok and results_match(
            llm.get("result"), gold.get("result"), ordered),
        "exact_match": normalise_query(r["LLM_cypher"])
                       == normalise_query(r["benchmark_cypher"]),
        "llm_exec_s": r["LLM_cypher_latency"] if llm_ok else None,
        "gold_exec_s": r["benchmark_cypher_latency"] if gold_ok else None,
        "gen_s": r["time_to_generate"],
        "llm_error": llm.get("error"),
        "ir_status": ir.get("status") or "-",
        **{f"f_{k}": v for k, v in query_features(r["benchmark_cypher"]).items()},
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

    header = keys + ["n", "gold_ok", "runnable", "empty", "non_null",
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
            fmt(rate("empty"), pct=True),
            fmt(rate("non_null"), pct=True),
            fmt(rate("result_correct"), pct=True),
            fmt(rate("exact_match"), pct=True),
            fmt(mean(r["llm_exec_s"] for r in g), 1000, 3),
            fmt(mean(r["gold_exec_s"] for r in g), 1000, 3),
            fmt(mean(r["gen_s"] for r in g)),
        ]) + " |")


def mcnemar_p(b, c):
    """Exact two-sided McNemar test on the discordant pairs b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def compare_runs(rows, run_a, run_b):
    """
    Paired comparison on the questions both runs answered: how many each
    got right that the other didn't, and whether the difference is more than
    chance (McNemar). Run ids can be given as unique prefixes.
    """
    ids = sorted({r["run_id"] for r in rows})

    def resolve(x):
        hits = [i for i in ids if i == x] or [i for i in ids if i.startswith(x)]
        if len(hits) != 1:
            raise SystemExit(f"run id {x!r} matches {hits or 'nothing'}; runs: {ids}")
        return hits[0]

    a_id, b_id = resolve(run_a), resolve(run_b)
    key = lambda r: (r["question"], r["gold_cypher"])
    a = {key(r): r["result_correct"] for r in rows if r["run_id"] == a_id}
    b = {key(r): r["result_correct"] for r in rows if r["run_id"] == b_id}
    common = sorted(a.keys() & b.keys())
    both = sum(a[k] and b[k] for k in common)
    only_a = sum(a[k] and not b[k] for k in common)
    only_b = sum(b[k] and not a[k] for k in common)
    neither = len(common) - both - only_a - only_b
    n = len(common) or 1
    print(f"\nPaired comparison on {len(common)} shared questions")
    print(f"  A = {a_id}: {(both + only_a) / n:.1%} correct")
    print(f"  B = {b_id}: {(both + only_b) / n:.1%} correct")
    print(f"  both correct {both} | only A {only_a} | only B {only_b} | neither {neither}")
    p = mcnemar_p(only_a, only_b)
    verdict = "significant at 5%" if p < 0.05 else "NOT significant (could be chance)"
    print(f"  McNemar exact p = {p:.3f}  -> {verdict}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sqlite", default="cypher_benchmark.sqlite")
    p.add_argument("--rows-csv", help="Also write per-question results here")
    p.add_argument("--by-type", action="store_true",
                   help="Also break results down by MTQ query_type")
    p.add_argument("--by-feature", action="store_true",
                   help="break results down by hops / filter / aggregation / "
                        "sorting / multi-stage of the gold query (Part 6)")
    p.add_argument("--by-attempt", action="store_true",
                   help="Part 7: accuracy at the first attempt and after 1 and 2 repairs")
    p.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"),
                   help="paired comparison of two runs (run ids or unique prefixes)")
    args = p.parse_args()

    rows = [evaluate_row(r) for r in load(args.sqlite)]
    if not rows:
        print("No rows found.")
        return

    print_table(rows, ["run_id", "mode", "model", "dataset"])
    if args.by_type:
        print()
        print_table(rows, ["run_id", "query_type"])
    if args.by_feature:
        for f in ("hops", "filter", "aggregation", "sorting", "multi_stage"):
            print()
            print_table(rows, ["run_id", f"f_{f}"])
    att_rows = [r for r in rows if "n_attempts" in r]
    if args.by_attempt and att_rows:
        print("\nRepair (Part 7): correct answers by attempt")
        print("| run_id | n | initial | +1 repair | +2 repairs | repaired | broken | "
              "avg attempts | blocked at the end |")
        print("|---|---|---|---|---|---|---|---|---|")
        by = defaultdict(list)
        for r in att_rows:
            by[r["run_id"]].append(r)
        for run, g in sorted(by.items()):
            n = len(g)
            c = [sum(r[f"att_correct_{k}"] for r in g) / n for k in range(3)]
            fixed = sum(r["att_correct_2"] and not r["att_correct_0"] for r in g)
            broke = sum(r["att_correct_0"] and not r["att_correct_2"] for r in g)
            print(f"| {run} | {n} | {c[0]:.1%} | {c[1]:.1%} | {c[2]:.1%} | {fixed} | {broke} "
                  f"| {sum(r['n_attempts'] for r in g) / n:.2f} "
                  f"| {sum(r['blocked'] for r in g) / n:.1%} |")
        print("\nFirst-attempt problem found by the guard (EMPTY = ran, no rows):")
        print_table(att_rows, ["run_id", "first_failure"])
    ir_rows = [r for r in rows if r["ir_status"] != "-"]
    if ir_rows:
        print("\nPlan route (--mode ir): what happened to each question")
        print_table(ir_rows, ["run_id", "ir_status"])
    if args.compare:
        compare_runs(rows, *args.compare)

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
