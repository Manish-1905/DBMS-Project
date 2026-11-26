#!/usr/bin/env python3
"""
Validation, safe execution and failure classification for generated Cypher.
(Roadmap Part 7.1-7.6)

    generated Cypher
      -> policy check   read-only, one statement, no admin/APOC calls,
                        bounded variable-length paths              (F13, F12)
      -> schema check   labels, relationship types and DIRECTION,
                        properties, value types                    (F2-F6)
      -> value check    compared values exist in the graph         (F7, F8)
      -> EXPLAIN        Neo4j parses and plans it without running:
                        syntax errors, unknown-name warnings, cost (F1, F12)
      -> execute        read-only session, time budget, row cap    (F1, F6, F12, F13)
      -> result         "empty" (no rows) is reported separately

Failure taxonomy (roadmap 7.6). The guard can detect F1-F8, F12, F13 by
itself; F9-F11 and F14 need the gold query (error_taxonomy.py):

    F1  syntax error               F8  value mismatch
    F2  nonexistent label          F9  incorrect traversal
    F3  nonexistent relationship   F10 aggregation error
    F4  wrong relationship direction  F11 temporal error
    F5  nonexistent property       F12 performance failure
    F6  wrong property type        F13 authorization (write / admin)
    F7  entity resolution error    F14 semantic failure
"""

import re
import time

from gold_schema import _IDENT, _clean, _STRINGS, _var_map, gold_schema_set, literal_spans
from query_ir import SchemaInfo, _close

TAXONOMY = {
    "F1": "syntax error", "F2": "nonexistent label", "F3": "nonexistent relationship",
    "F4": "wrong relationship direction", "F5": "nonexistent property",
    "F6": "wrong property type", "F7": "entity resolution error",
    "F8": "value mismatch", "F9": "incorrect traversal", "F10": "aggregation error",
    "F11": "temporal error", "F12": "performance failure", "F13": "authorization failure",
    "F14": "semantic failure",
}

# Roadmap 7.2 budgets
DEFAULT_BUDGET = {"timeout_s": 5.0, "max_rows": 10000, "max_path_depth": 5,
                  "max_estimated_rows": 10_000_000}

_WRITE = re.compile(r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|"
                    r"GRANT|REVOKE|DENY|ALTER|RENAME|START|STOP|TERMINATE)\b", re.I)
_ADMIN = re.compile(r"\bSHOW\b", re.I)
_LOAD = re.compile(r"\bLOAD\s+CSV\b", re.I)
_USE = re.compile(r"^\s*USE\b|\bUSE\s+[`\w]", re.I)
_CALL_PROC = re.compile(r"\bCALL\s+([A-Za-z_][\w.]*)", re.I)
_READ_PROCS = {"db.labels", "db.relationshiptypes", "db.propertykeys",
               "db.schema.visualization", "db.schema.nodetypeproperties",
               "db.schema.reltypeproperties"}
_VARLEN = re.compile(r"\[[^\]]*?\*\s*(\d*)\s*(?:\.\.\s*(\d*))?[^\]]*\]")
_LOW_CARD = 200


def failure(code, message):
    return {"code": code, "name": TAXONOMY[code], "message": message}


# ------------------------------------------------------------
# 7.1 static checks (no database needed)
# ------------------------------------------------------------

def policy_check(cypher, budget=None):
    """Read-only, single statement, no admin calls, bounded paths."""
    b = {**DEFAULT_BUDGET, **(budget or {})}
    q = _STRINGS.sub("''", cypher or "")
    out = []
    if not q.strip():
        return [failure("F1", "empty query")]
    if ";" in q.strip().rstrip(";"):
        out.append(failure("F13", "only one statement is allowed (found ';')"))
    w = _WRITE.search(q)
    if w:
        out.append(failure("F13", f"'{w.group(1).upper()}' would modify the database; "
                                  "only read queries (MATCH ... RETURN) are allowed"))
    if _ADMIN.search(q):
        out.append(failure("F13", "administration commands (SHOW ...) are not allowed"))
    if _LOAD.search(q):
        out.append(failure("F13", "reading external files (LOAD CSV) is not allowed"))
    if _USE.search(q):
        out.append(failure("F13", "switching databases with USE is not allowed"))
    for m in _CALL_PROC.finditer(q):
        if m.group(1).lower() not in _READ_PROCS:
            out.append(failure("F13", f"procedure '{m.group(1)}' is not allowed"))
    for m in _VARLEN.finditer(q):
        lo, hi = m.group(1), m.group(2)
        upper = int(hi) if hi else (int(lo) if lo and ".." not in m.group(0) else None)
        if upper is None:
            out.append(failure("F12", f"unbounded variable-length path {m.group(0)}; "
                                      f"give an upper bound (at most {b['max_path_depth']})"))
        elif upper > b["max_path_depth"]:
            out.append(failure("F12", f"path length up to {upper} exceeds the limit of "
                                      f"{b['max_path_depth']}"))
    return out


_TRIPLE = re.compile(
    rf"(?=\(\s*({_IDENT})?\s*((?::\s*(?:{_IDENT})\s*)*)[^)]*\)\s*(<?)-\s*\[\s*({_IDENT})?\s*"
    rf":\s*({_IDENT}(?:\s*\|\s*:?\s*{_IDENT})*)[^\]]*\]\s*-(>?)\s*\(\s*({_IDENT})?\s*"
    rf"((?::\s*(?:{_IDENT})\s*)*))")
_CMP = re.compile(rf"({_IDENT})\s*\.\s*({_IDENT})\s*(=|<>|<=|>=|<|>)\s*"
                  r"('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|-?\d+(?:\.\d+)?)")


def schema_check(cypher, schema):
    """Labels, relationship types, directions, properties and value types."""
    sch = schema if isinstance(schema, SchemaInfo) else SchemaInfo(schema)
    q = _STRINGS.sub("''", cypher or "")
    used = gold_schema_set(cypher)
    out = []
    for L in sorted(used["labels"] - set(sch.node_props)):
        sug = _close(L, sch.node_props)
        out.append(failure("F2", f"label '{L}' does not exist"
                                 + (f" (did you mean {', '.join(sug)}?)" if sug else "")))
    for T in sorted(used["rels"] - sch.rel_types):
        sug = _close(T, sch.rel_types)
        out.append(failure("F3", f"relationship type '{T}' does not exist"
                                 + (f" (did you mean {', '.join(sug)}?)" if sug else "")))
    for p in sorted(used["props"]):
        owner, _, prop = p.partition(".")
        props = sch.node_props.get(owner, sch.rel_props.get(owner))
        if props is not None and prop not in props:
            sug = _close(prop, props)
            out.append(failure("F5", f"{owner} has no property '{prop}'"
                                     + (f" (did you mean {', '.join(sug)}?)" if sug else
                                        f"; its properties: {', '.join(sorted(props))}")))
    # direction
    var_labels = _var_map(q)
    for m in _TRIPLE.finditer(q):
        la = {_clean(x) for x in re.findall(_IDENT, m.group(2) or "")} or \
            var_labels.get(_clean(m.group(1) or ""), set())
        lb = {_clean(x) for x in re.findall(_IDENT, m.group(8) or "")} or \
            var_labels.get(_clean(m.group(7) or ""), set())
        types = {_clean(x) for x in re.findall(_IDENT, m.group(5))}
        left, right = m.group(3) == "<", m.group(6) == ">"
        if left == right:          # undirected, or both arrows: nothing to check
            continue
        src, dst = (lb, la) if left else (la, lb)
        for t in types & sch.rel_types:
            for a in src & set(sch.node_props):
                for b in dst & set(sch.node_props):
                    if (a, t, b) not in sch.triples and (b, t, a) in sch.triples:
                        out.append(failure("F4", f"({a})-[:{t}]->({b}) has the wrong "
                                                 f"direction; the graph has ({b})-[:{t}]->({a})"))
    # value types
    for m in _CMP.finditer(cypher or ""):
        v, prop, lit = _clean(m.group(1)), _clean(m.group(2)), m.group(4)
        for owner in var_labels.get(v, ()):
            dtype = sch.node_props.get(owner, sch.rel_props.get(owner, {})).get(prop)
            if not dtype:
                continue
            is_str = lit[0] in "'\""
            if dtype in ("INTEGER", "FLOAT") and is_str:
                out.append(failure("F6", f"{owner}.{prop} is a number ({dtype}) but is "
                                         f"compared with the text {lit}"))
            elif dtype == "STRING" and not is_str:
                out.append(failure("F6", f"{owner}.{prop} is text but is compared with "
                                         f"the number {lit}"))
    return _dedupe(out)


def value_check(cypher, catalog):
    """Compared values must exist in the graph (catalog from value_index.py)."""
    props = (catalog or {}).get("properties", catalog or {})
    out = []
    for _, _, key, value in literal_spans(cypher or ""):
        known = (props.get(key) or {}).get("values")
        if not known or value in known:
            continue
        code = "F8" if len(known) <= _LOW_CARD else "F7"
        same_case = [k for k in known if k.lower() == value.lower()]
        if same_case:
            hint = f"the graph stores {same_case[0]!r}"
        elif len(known) <= 30:
            hint = "existing values: " + ", ".join(repr(k) for k in known)
        else:
            sug = _close(value, list(known)[:20000], n=5)
            hint = ("closest existing values: " + ", ".join(repr(s) for s in sug)) if sug \
                else "no similar value exists in the graph"
        out.append(failure(code, f"{key} has no value {value!r}; {hint}"))
    return _dedupe(out)


def _dedupe(fails):
    seen, out = set(), []
    for f in fails:
        if (f["code"], f["message"]) not in seen:
            seen.add((f["code"], f["message"]))
            out.append(f)
    return out


def static_check(cypher, schema, catalog=None, budget=None):
    return policy_check(cypher, budget) + schema_check(cypher, schema) + \
        value_check(cypher, catalog)


# ------------------------------------------------------------
# Database errors -> taxonomy
# ------------------------------------------------------------

def classify_db_error(message, code=""):
    t = f"{code} {message}"
    if re.search(r"SyntaxError|Invalid input|syntax error|Variable `\w+` not defined", t, re.I):
        return "F1"
    if re.search(r"TimedOut|timeout|timed out|MemoryLimit|OutOfMemory|memory", t, re.I):
        return "F12"
    if re.search(r"AccessMode|Forbidden|not allowed|permission|read-only|write", t, re.I):
        return "F13"
    if re.search(r"TypeError|ArgumentError|Type mismatch|expected .*but was|Cannot ", t, re.I):
        return "F6"
    return "F1"


# ------------------------------------------------------------
# 7.3 EXPLAIN and 7.4 guarded execution
# ------------------------------------------------------------

_NOTE_CODES = {"UnknownLabelWarning": "F2", "UnknownRelationshipTypeWarning": "F3",
               "UnknownPropertyKeyWarning": "F5"}


class Guard:
    """Checks and runs generated Cypher against one database."""

    def __init__(self, driver, database, schema, catalog=None, budget=None):
        from neo4j import READ_ACCESS
        self.driver, self.database = driver, database
        self.schema = SchemaInfo(schema)
        self.catalog = catalog
        self.budget = {**DEFAULT_BUDGET, **(budget or {})}
        self.READ_ACCESS = READ_ACCESS

    def explain(self, cypher, params=None):
        """(failures, warnings, estimated_rows) without executing the query."""
        from neo4j import Query
        try:
            with self.driver.session(database=self.database,
                                     default_access_mode=self.READ_ACCESS) as s:
                summary = s.run(Query("EXPLAIN " + cypher, timeout=self.budget["timeout_s"]),
                                params or {}).consume()
        except Exception as e:
            msg = str(getattr(e, "message", "") or e).splitlines()[0][:300]
            return [failure(classify_db_error(msg, getattr(e, "code", "") or ""), msg)], [], None
        fails, warns = [], []
        for n in (getattr(summary, "notifications", None) or []):
            code = n.get("code", "") if isinstance(n, dict) else getattr(n, "code", "")
            desc = n.get("description", "") if isinstance(n, dict) else \
                getattr(n, "description", "")
            short = code.split(".")[-1]
            if short in _NOTE_CODES:
                fails.append(failure(_NOTE_CODES[short], desc[:200] or short))
            elif short:
                warns.append(f"{short}: {desc[:150]}")
        est = None
        plan = getattr(summary, "plan", None)
        if isinstance(plan, dict):
            est = (plan.get("args") or {}).get("EstimatedRows")
        if est and est > self.budget["max_estimated_rows"]:
            fails.append(failure("F12", f"Neo4j estimates {int(est):,} rows; the query is "
                                        "too expensive (add filters or a LIMIT)"))
        return _dedupe(fails), warns, est

    def execute(self, cypher, params=None):
        """Read-only run with time budget and row cap."""
        from neo4j import Query
        t0 = time.perf_counter()
        rows, truncated = [], False
        try:
            with self.driver.session(database=self.database,
                                     default_access_mode=self.READ_ACCESS) as s:
                res = s.run(Query(cypher, timeout=self.budget["timeout_s"]), params or {})
                for rec in res:
                    if len(rows) >= self.budget["max_rows"]:
                        truncated = True
                        break
                    rows.append(rec.data())
            return {"success": True, "error": None, "result": rows, "truncated": truncated,
                    "latency": time.perf_counter() - t0}
        except Exception as e:
            msg = str(getattr(e, "message", "") or e).splitlines()[0][:300]
            return {"success": False, "error": msg, "result": None, "truncated": False,
                    "latency": time.perf_counter() - t0,
                    "code": classify_db_error(msg, getattr(e, "code", "") or "")}

    def run(self, cypher, params=None):
        """
        Full pipeline. Returns
          {"stage": "static"|"explain"|"execute"|"ok", "ok": bool,
           "failures": [...], "warnings": [...], "empty": bool,
           "success", "error", "result", "latency", "truncated", "estimated_rows"}
        "ok" means it ran and returned rows. A query that fails a check is
        NOT executed.
        """
        out = {"stage": "static", "ok": False, "failures": [], "warnings": [],
               "empty": False, "success": False, "error": None, "result": None,
               "latency": None, "truncated": False, "estimated_rows": None}
        fails = static_check(cypher, self.schema, self.catalog, self.budget)
        blocking = [f for f in fails if f["code"] in ("F1", "F12", "F13")]
        if blocking:
            out.update(failures=fails, error="; ".join(f["message"] for f in blocking))
            return out
        out["stage"] = "explain"
        efails, warns, est = self.explain(cypher, params)
        out["warnings"], out["estimated_rows"] = warns, est
        fails = _dedupe(fails + efails)
        if any(f["code"] in ("F1", "F12", "F13") for f in efails):
            out.update(failures=fails, error="; ".join(f["message"] for f in efails))
            return out
        out["stage"] = "execute"
        ex = self.execute(cypher, params)
        out.update({k: ex[k] for k in ("success", "error", "result", "latency", "truncated")})
        if not ex["success"]:
            fails.append(failure(ex["code"], ex["error"]))
            out["failures"] = _dedupe(fails)
            return out
        out["failures"] = fails            # schema/value problems that still ran
        out["empty"] = not ex["result"]
        out["ok"] = not out["empty"] and not fails
        out["stage"] = "ok" if out["ok"] else "execute"
        return out


# ------------------------------------------------------------
# 7.5 repair feedback ("tell it WHY the query failed")
# ------------------------------------------------------------

def repair_feedback(cypher, outcome, schema, catalog=None, grounded=None):
    sch = schema if isinstance(schema, SchemaInfo) else SchemaInfo(schema)
    lines = ["Your previous Cypher query:", cypher or "(none)", ""]
    fails = outcome.get("failures") or []
    if fails:
        lines.append("It has these problems:")
        lines += [f"- [{f['code']} {f['name']}] {f['message']}" for f in fails]
    elif outcome.get("error"):
        lines.append(f"Neo4j rejected it: {outcome['error']}")
    codes = {f["code"] for f in fails}
    if codes & {"F2", "F3", "F4", "F5"}:
        lines.append("")
        lines.append("Valid relationships (with direction):")
        lines += [f"  ({a})-[:{t}]->({b})" for a, t, b in sorted(sch.triples)]
        lines.append("Properties:")
        lines += [f"  {L}: {', '.join(sorted(ps))}" for L, ps in sorted(sch.node_props.items())]
        lines += [f"  [:{T}]: {', '.join(sorted(ps))}" for T, ps in sorted(sch.rel_props.items())]
    if "F13" in codes:
        lines.append("Only read queries are allowed: MATCH ... (WHERE ...) RETURN ...")
    if "F12" in codes:
        lines.append("Make it cheaper: bound path lengths (e.g. *1..3), add filters or a LIMIT.")
    if outcome.get("empty") and not fails:
        lines.append("It ran but returned NO rows.")
        lines.append("Check that compared values are spelled exactly as stored, that "
                     "relationship directions are right, and that the filters are not "
                     "stricter than the question.")
        if grounded:
            lines.append("Values that exist in the graph for this question:")
            lines += [f"  {g['property']} = {g['value']!r}" for g in grounded]
        lines.append("If the correct answer really is empty, return the same query.")
    lines.append("")
    lines.append("Write a corrected Cypher query for the question.")
    return "\n".join(lines)
