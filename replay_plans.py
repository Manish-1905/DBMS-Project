import argparse
import json
import os
import sqlite3
from collections import Counter

import pandas as pd

from evaluate import evaluate_row, load, normalise_query, results_match
from main import Neo4jExecutor, extract_json
from query_ir import SchemaInfo, compile_ir, parse_plan, validate
from value_index import load_catalog


def first_plan(raw_response):
    try:
        raws = json.loads(raw_response or "[]")
    except ValueError:
        return None
    if not isinstance(raws, list) or not raws:
        return None
    try:
        return parse_plan(extract_json(raws[0]))
    except ValueError:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sqlite", default="plan.sqlite")
    p.add_argument("--run", default="ir", help="run_id prefix of the ir run")
    p.add_argument("--parquet", default="Benchmark/MTQ-Auto/test-split.parquet")
    p.add_argument("--values", default="values/er.json")
    p.add_argument("--direct", default="grounded",
                   help="run_id prefix of a direct-generation run on the same questions")
    p.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687"))
    p.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    p.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", ""))
    p.add_argument("--neo4j-database", default=os.environ.get("DB", "neo4j"))
    a = p.parse_args()

    schemas = pd.read_parquet(a.parquet).groupby("data_source")["schema"].first().to_dict()
    catalog = load_catalog(a.values) if os.path.exists(a.values) else None
    raw_rows = {r["id"]: r for r in load(a.sqlite)}
    conn = sqlite3.connect(a.sqlite)
    conn.row_factory = sqlite3.Row
    runs = sorted({r["run_id"] for r in raw_rows.values() if r["run_id"].startswith(a.run)})
    if len(runs) != 1:
        raise SystemExit(f"--run {a.run!r} matches {runs or 'nothing'}")
    run = runs[0]
    rows = [r for r in raw_rows.values() if r["run_id"] == run]
    ex = Neo4jExecutor(a.neo4j_uri, a.neo4j_user, a.neo4j_password, a.neo4j_database)

    direct_runs = sorted({x["run_id"] for x in raw_rows.values()
                          if x["run_id"].startswith(a.direct)})
    direct = {}
    if len(direct_runs) == 1:
        for x in raw_rows.values():
            if x["run_id"] == direct_runs[0]:
                e = evaluate_row(x)
                direct[(e["question"], e["gold_cypher"])] = e["result_correct"]
    per_q = []

    trans = Counter()
    old_ok = new_ok = 0
    for r in rows:
        before = evaluate_row(r)
        old_status = before["ir_status"]
        old_ok += before["result_correct"]
        plan = first_plan(r["raw_response"])
        new_status = "no_plan"
        correct = before["result_correct"]
        plan_info = None
        if plan is not None:
            fixed, errs, _ = validate(plan, SchemaInfo(schemas[r["dataset"]]), catalog)
            if not errs:
                new_status = "ok"
                plan_info = (len(fixed.get("paths") or []),
                             bool(fixed.get("with")) or any(x.get("agg") for x in fixed["return"]))
                if old_status != "ok":
                    cypher, params = compile_ir(fixed)
                    res = ex.execute_query(cypher, params)
                    gold = json.loads(r["benchmark_cypher_result"] or "null") or {}
                    ordered = "order by" in normalise_query(r["benchmark_cypher"])
                    correct = bool(res["success"] and gold.get("success") and results_match(
                        res["result"], gold.get("result"), ordered))
            else:
                new_status = "still_invalid"
        trans[(old_status, new_status, correct)] += 1
        new_ok += correct
        key = (before["question"], before["gold_cypher"])
        if key in direct:
            per_q.append((plan_info, correct if plan_info else None, direct[key]))
    ex.close()

    n = len(rows)
    print(f"run {run}: {n} questions")
    print(f"first plan valid: before {sum(v for (o, _, _), v in trans.items() if o == 'ok')}"
          f" -> now {sum(v for (_, s, _), v in trans.items() if s == 'ok')}")
    print("\n| original outcome | first plan now | n | correct |")
    print("|---|---|---|---|")
    groups = Counter()
    good = Counter()
    for (o, s, c), v in trans.items():
        groups[(o, s)] += v
        good[(o, s)] += v if c else 0
    for (o, s), v in sorted(groups.items()):
        print(f"| {o} | {s} | {v} | {good[(o, s)] / v:.0%} |")
    print(f"\ncorrect overall: original {old_ok / n:.1%} -> with the current checker "
          f"(estimate) {new_ok / n:.1%}")

    if per_q:
        rules = {
            "always direct generation": lambda pi: False,
            "plan whenever it is valid": lambda pi: True,
            "plan if >= 2 paths (multi-hop)": lambda pi: pi[0] >= 2,
            "plan if >= 2 paths and no aggregation": lambda pi: pi[0] >= 2 and not pi[1],
            "plan if no aggregation": lambda pi: not pi[1],
        }
        print(f"\nRouting rules on {len(per_q)} questions (vs direct run "
              f"{direct_runs[0]}); the rule only sees the plan, never the gold query:")
        print("| rule | uses plan | correct |")
        print("|---|---|---|")
        for name, use in rules.items():
            used = ok = 0
            for pi, plan_ok, direct_ok in per_q:
                if pi is not None and use(pi):
                    used += 1
                    ok += plan_ok
                else:
                    ok += direct_ok
            print(f"| {name} | {used} | {ok / len(per_q):.1%} |")


if __name__ == "__main__":
    main()
