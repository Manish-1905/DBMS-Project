import argparse
import json
import os
import re
import sqlite3

import pandas as pd

from cypher_guard import classify_db_error, policy_check, schema_check
from evaluate import evaluate_row, load, mcnemar_p
from gold_schema import gold_literals
from query_ir import SchemaInfo

_PARAM = re.compile(r"\$(\w+)")


def inline_params(cypher, params):
    if not params:
        return cypher

    def lit(v):
        if isinstance(v, str):
            return "'" + v.replace("\\", "\\\\").replace("'", "\\'") + "'"
        if isinstance(v, list):
            return "[" + ", ".join(lit(x) for x in v) + "]"
        return json.dumps(v)
    return _PARAM.sub(lambda m: lit(params[m.group(1)]) if m.group(1) in params else m.group(0),
                      cypher)


def load_run(spec):
    label, _, rest = spec.partition("=")
    path, _, prefix = rest.rpartition(":")
    if not label or not path or not prefix:
        raise SystemExit(f"--run must look like NAME=file.sqlite:run_id_prefix, got {spec!r}")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found")
    raw = load(path)
    ids = sorted({r["run_id"] for r in raw if r["run_id"] == prefix}) or \
        sorted({r["run_id"] for r in raw if r["run_id"].startswith(prefix)})
    if len(ids) != 1:
        raise SystemExit(f"{spec}: run id prefix {prefix!r} matches {ids or 'nothing'}")
    cols = {r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(benchmark_results)")}
    rows = []
    for r in raw:
        if r["run_id"] != ids[0]:
            continue
        e = evaluate_row(r)
        try:
            ctx = json.loads(r.get("retrieved_context") or "null") or {}
        except ValueError:
            ctx = {}
        params = (ctx.get("ir") or {}).get("params") or {}
        e["cypher_inlined"] = inline_params(e["llm_cypher"] or "", params)
        e["tokens"] = ((r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0)) \
            if "prompt_tokens" in cols and r.get("prompt_tokens") is not None else None
        rows.append(e)
    return label, ids[0], rows


def metrics(rows, schemas):
    n = len(rows)
    syn = sch = ent_ok = ent_n = 0
    for r in rows:
        cy = r["cypher_inlined"]
        if not cy.strip():
            continue
        err_f1 = bool(r["llm_error"]) and classify_db_error(r["llm_error"]) == "F1"
        if not err_f1 and not any(f["code"] == "F1" for f in policy_check(cy)):
            syn += 1
        if not any(f["code"] in ("F2", "F3", "F4", "F5", "F6")
                   for f in schema_check(cy, schemas[r["dataset"]])):
            sch += 1
    for r in rows:
        gold = gold_literals(r["gold_cypher"])
        if not gold:
            continue
        ent_n += 1
        got = gold_literals(r["cypher_inlined"])
        props = {p for p, _ in gold}
        if {x for x in got if x[0] in props} == gold:
            ent_ok += 1
    m = {
        "n": n,
        "syntax": syn / n, "schema": sch / n,
        "entity": ent_ok / ent_n if ent_n else None, "entity_n": ent_n,
        "execution": sum(r["runnable"] for r in rows) / n,
        "result": sum(r["result_correct"] for r in rows) / n,
        "empty": sum(r["empty"] for r in rows) / n,
    }
    gen = [r["gen_s"] for r in rows if r["gen_s"] is not None]
    ex = [r["llm_exec_s"] for r in rows if r["llm_exec_s"] is not None]
    m["latency"] = (sum(gen) / len(gen) if gen else 0) + (sum(ex) / len(ex) if ex else 0)
    tok = [r["tokens"] for r in rows if r["tokens"]]
    m["tokens"] = sum(tok) / len(tok) if tok else None
    att = [r for r in rows if "n_attempts" in r]
    if att:
        flagged = [r for r in att if r["first_failure"] != "-"]
        fixed = sum(r["att_correct_2"] and not r["att_correct_0"] for r in flagged)
        m["repair"] = (fixed, len(flagged))
        clean = [r for r in att if r["first_failure"] == "-"]
        m["selective"] = (len(clean) / n, sum(r["result_correct"] for r in clean) / len(clean)
                          if clean else 0)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True,
                   help="NAME=file.sqlite:run_id_prefix (repeat, in system order)")
    p.add_argument("--parquet", default="Benchmark/MTQ-Auto/test-split.parquet")
    p.add_argument("--all-questions", action="store_true",
                   help="don't restrict to the questions shared by all runs")
    p.add_argument("--out", help="also write the markdown table here")
    a = p.parse_args()

    schemas = {k: SchemaInfo(v) for k, v in
               pd.read_parquet(a.parquet).groupby("data_source")["schema"].first().items()}
    runs = [load_run(s) for s in a.run]
    key = lambda r: (r["question"], r["gold_cypher"])
    if not a.all_questions:
        common = set.intersection(*[{key(r) for r in rows} for _, _, rows in runs])
        runs = [(lab, rid, [r for r in rows if key(r) in common]) for lab, rid, rows in runs]
        runs = [(lab, rid, list({key(r): r for r in rows}.values())) for lab, rid, rows in runs]
    ms = [(lab, rid, metrics(rows, schemas), rows) for lab, rid, rows in runs]

    pct = lambda x: "–" if x is None else f"{x:.1%}"
    lines = []
    lines.append("| Metric | " + " | ".join(lab for lab, _, _, _ in ms) + " |")
    lines.append("|---|" + "---|" * len(ms))
    row = lambda name, f: lines.append(f"| {name} | " + " | ".join(f(m) for _, _, m, _ in ms) + " |")
    row("questions", lambda m: str(m["n"]))
    row("syntax accuracy", lambda m: pct(m["syntax"]))
    row("schema accuracy", lambda m: pct(m["schema"]))
    row("entity accuracy", lambda m: pct(m["entity"]) + (f" (n={m['entity_n']})" if m["entity_n"] else ""))
    row("execution accuracy", lambda m: pct(m["execution"]))
    row("**result accuracy**", lambda m: f"**{pct(m['result'])}**")
    row("empty-result rate", lambda m: pct(m["empty"]))
    row("repair success", lambda m: f"{m['repair'][0]}/{m['repair'][1]}" if "repair" in m else "–")
    row("selective: answered", lambda m: pct(m["selective"][0]) if "selective" in m else "–")
    row("selective: accuracy", lambda m: pct(m["selective"][1]) if "selective" in m else "–")
    row("avg latency (s)", lambda m: f"{m['latency']:.2f}")
    row("avg tokens", lambda m: f"{m['tokens']:.0f}" if m["tokens"] else "–")
    row("avg cost", lambda m: "local ($0)")

    lines.append("")
    lines.append("Paired comparison with the previous system (exact McNemar, result accuracy):")
    lines.append("")
    lines.append("| Comparison | only previous correct | only this correct | p |")
    lines.append("|---|---|---|---|")
    for (la, _, _, ra), (lb, _, _, rb) in zip(ms, ms[1:]):
        A = {key(r): r["result_correct"] for r in ra}
        B = {key(r): r["result_correct"] for r in rb}
        common = A.keys() & B.keys()
        oa = sum(A[k] and not B[k] for k in common)
        ob = sum(B[k] and not A[k] for k in common)
        pv = mcnemar_p(oa, ob)
        lines.append(f"| {la} → {lb} | {oa} | {ob} | {pv:.3f}{' *' if pv < 0.05 else ''} |")
    lines.append("")
    lines.append("Runs: " + "; ".join(f"{lab} = {rid}" for lab, rid, _, _ in ms))
    if not a.all_questions:
        lines.append(f"Restricted to the {ms[0][2]['n']} questions answered by every run.")
    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
        print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
