#!/usr/bin/env python3
"""
Graph Query IR: plan first, then Cypher. (Roadmap Part 6.1-6.3)

The LLM fills in a structured plan instead of writing Cypher. The plan is
checked against the schema (and the value catalog), auto-fixed where the fix
is certain, and compiled to Cypher by plain code, so the syntax is always
valid and values are passed as parameters.

IR format (all keys except "nodes" and "return" are optional):

{
  "nodes":    [{"var": "u", "label": "User"}, {"var": "m", "label": "Movie"}],
  "paths":    [{"from": "u", "rel": "WATCHED", "to": "m", "var": "w"}],
  "filters":  [{"expr": "u.gender", "op": "=", "value": "Female"},
               {"or": [{"expr": "m.year", "op": ">", "value": 2000},
                       {"expr": "m.title", "op": "STARTS WITH", "value": "The"}]}],
  "with": {                                   # one intermediate aggregation
     "group_by":   [{"expr": "u"}, {"expr": "u.state", "as": "state"}],
     "aggregates": [{"agg": "count", "expr": "m", "distinct": true, "as": "nMovies"}],
     "having":     [{"expr": "nMovies", "op": ">", "value": 5}]
  },
  "return":   [{"expr": "u.state", "as": "state"},
               {"agg": "avg", "expr": "nMovies", "as": "avgMovies"}],
  "distinct": false,
  "order_by": [{"expr": "avgMovies", "desc": true}],
  "limit":    10
}

Grouping follows Cypher: non-aggregated return items are the group keys.
Questions the IR can't express (OPTIONAL MATCH, several MATCH stages, CASE,
sub-queries...) make the LLM answer {"unsupported": true}; main.py then falls
back to direct Cypher generation.
"""

import difflib
import json
import re

AGGS = {"count", "sum", "avg", "min", "max", "collect"}
OPS = {"=", "<>", "<", "<=", ">", ">=", "IN", "CONTAINS", "STARTS WITH", "ENDS WITH",
       "IS NULL", "IS NOT NULL"}
_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_EXPR = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\.([A-Za-z_][A-Za-z0-9_]*))?$")


class IRError(Exception):
    pass


# Words small models use for aggregates -> (aggregate, distinct)
AGG_SYNONYMS = {
    "average": ("avg", False), "mean": ("avg", False), "total": ("sum", False),
    "minimum": ("min", False), "maximum": ("max", False), "number": ("count", False),
    "count_distinct": ("count", True), "countdistinct": ("count", True),
    "distinct_count": ("count", True), "count distinct": ("count", True),
    "list": ("collect", False),
}
_AGG_IN_EXPR = re.compile(r"^\s*([A-Za-z_]+)\s*\(\s*(DISTINCT\s+)?([^()]*?)\s*\)\s*$", re.I)


def _normalise_agg(item, warnings, where):
    """
    Certain fixes for aggregate items: synonyms ("average" -> avg,
    "count_distinct" -> count + distinct) and Cypher written into "expr"
    ("count(DISTINCT m)" -> agg count, expr m, distinct).
    """
    m = _AGG_IN_EXPR.match(str(item.get("expr") or ""))
    if m and not item.get("agg"):
        name = m.group(1).lower()
        name, dist = AGG_SYNONYMS.get(name, (name, False))
        if name in AGGS:
            warnings.append(f"{where}: '{item['expr']}' -> agg {name} of '{m.group(3)}'")
            item["agg"], item["expr"] = name, m.group(3) or "*"
            item["distinct"] = bool(m.group(2)) or dist or item.get("distinct", False)
    agg = item.get("agg")
    if agg:
        key = str(agg).strip().lower().replace("-", "_")
        if key not in AGGS and key in AGG_SYNONYMS:
            name, dist = AGG_SYNONYMS[key]
            warnings.append(f"{where}: aggregate '{agg}' -> '{name}'")
            item["agg"] = name
            item["distinct"] = item.get("distinct", False) or dist


def _q(name):
    return name if _ID.match(name) else "`" + name.replace("`", "``") + "`"


def _close(word, options, n=3):
    lower = {o.lower(): o for o in options}
    hits = difflib.get_close_matches(word.lower(), list(lower), n=n, cutoff=0.5)
    return [lower[h] for h in hits]


def _fix_case(word, options):
    """Exact match, or a unique case-insensitive match."""
    if word in options:
        return word
    m = [o for o in options if o.lower() == str(word).lower()]
    return m[0] if len(m) == 1 else None


class SchemaInfo:
    def __init__(self, schema):
        s = json.loads(schema) if isinstance(schema, str) else schema
        self.node_props = {L: {p["property"]: p["datatype"] for p in ps}
                           for L, ps in s.get("node_props", {}).items()}
        self.rel_props = {T: {p["property"]: p["datatype"] for p in ps}
                          for T, ps in s.get("rel_props", {}).items()}
        self.triples = {(r["start"], r["type"], r["end"]) for r in s.get("relationships", [])}
        for a, _, b in self.triples:
            self.node_props.setdefault(a, {})
            self.node_props.setdefault(b, {})
        self.rel_types = {t for _, t, _ in self.triples} | set(self.rel_props)


# ------------------------------------------------------------
# Validation (6.2)
# ------------------------------------------------------------

def validate(ir, schema, catalog=None):
    """
    Check an IR against the schema (and the value catalog, if given).
    Returns (fixed_ir, errors, warnings). Errors are written to be shown to
    the LLM ("why it failed"); warnings are certain automatic fixes.
    """
    sch = schema if isinstance(schema, SchemaInfo) else SchemaInfo(schema)
    ir = json.loads(json.dumps(ir))          # deep copy
    errors, warnings = [], []
    cat = (catalog or {}).get("properties", catalog or {})

    if not isinstance(ir, dict):
        return ir, ["the plan must be a JSON object"], warnings

    # ---- nodes ----
    var_label = {}
    nodes = ir.get("nodes") or []
    if not nodes:
        errors.append('"nodes" is empty: list the node variables and their labels')
    for n in nodes:
        v, L = n.get("var"), n.get("label")
        if not v or not _ID.match(str(v)):
            errors.append(f"node variable {v!r} is not a valid name")
            continue
        if v in var_label:
            errors.append(f"node variable '{v}' is declared twice")
            continue
        fixed = _fix_case(L, sch.node_props)
        if fixed is None:
            sug = _close(str(L), sch.node_props)
            errors.append(f"label '{L}' does not exist"
                          + (f"; did you mean {', '.join(sug)}?" if sug else
                             f"; labels are: {', '.join(sorted(sch.node_props))}"))
            continue
        if fixed != L:
            warnings.append(f"label '{L}' -> '{fixed}'")
            n["label"] = fixed
        var_label[v] = fixed

    # ---- label names used instead of variables (a common slip) ----
    def resolve_var(name, where, declare=True):
        """
        "Genre" used where a variable is expected: use the variable declared
        with that label if there is exactly one; otherwise declare one.
        """
        if name in var_label or not isinstance(name, str):
            return name
        label = _fix_case(name, sch.node_props)
        if label is None:
            return name
        same = [v for v, L in var_label.items() if L == label]
        if len(same) == 1:
            warnings.append(f"{where}: label '{name}' used as a variable -> '{same[0]}'")
            return same[0]
        if not same and declare:
            new = label[0].lower() + "".join(c for c in label[1:] if c.isalnum())
            while new in var_label:
                new += "_"
            var_label[new] = label
            ir.setdefault("nodes", []).append({"var": new, "label": label})
            warnings.append(f"{where}: undeclared '{name}' -> new node ({new}:{label})")
            return new
        return name

    for i, p in enumerate(ir.get("paths") or []):
        for end in ("from", "to"):
            p[end] = resolve_var(p.get(end), f"paths[{i}].{end}")

    def resolve_expr_label(e, where):
        """'Movie.title' where Movie is a label, not a variable -> 'm.title'."""
        m = _EXPR.match(str(e or ""))
        if m and m.group(1) not in var_label:
            v = resolve_var(m.group(1), where, declare=False)
            if v != m.group(1):
                return v + (f".{m.group(2)}" if m.group(2) else "")
        return e

    for i, f in enumerate(ir.get("filters") or []):
        if "expr" in f:
            f["expr"] = resolve_expr_label(f["expr"], f"filters[{i}]")
    for i, r in enumerate(ir.get("return") or []):
        _normalise_agg(r, warnings, f"return[{i}]")
        r["expr"] = resolve_expr_label(r.get("expr"), f"return[{i}]")
    for i, o in enumerate(ir.get("order_by") or []):
        o["expr"] = resolve_expr_label(o.get("expr"), f"order_by[{i}]")
    if isinstance(ir.get("with"), dict):
        for i, g in enumerate(ir["with"].get("group_by") or []):
            g["expr"] = resolve_expr_label(g.get("expr"), f"with.group_by[{i}]")
        for i, a in enumerate(ir["with"].get("aggregates") or []):
            _normalise_agg(a, warnings, f"with.aggregates[{i}]")
            a["expr"] = resolve_expr_label(a.get("expr"), f"with.aggregates[{i}]")

    # ---- paths ----
    rel_var_type = {}
    parent = {v: v for v in var_label}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in ir.get("paths") or []:
        a, b, t = p.get("from"), p.get("to"), p.get("rel")
        if a not in var_label or b not in var_label:
            missing = [x for x in (a, b) if x not in var_label]
            errors.append(f"path {a}-[:{t}]->{b} uses undeclared node variable(s) "
                          f"{', '.join(map(str, missing))}")
            continue
        fixed = _fix_case(t, sch.rel_types)
        if fixed is None:
            sug = _close(str(t), sch.rel_types)
            errors.append(f"relationship type '{t}' does not exist"
                          + (f"; did you mean {', '.join(sug)}?" if sug else
                             f"; types are: {', '.join(sorted(sch.rel_types))}"))
            continue
        if fixed != t:
            warnings.append(f"relationship '{t}' -> '{fixed}'")
            p["rel"] = t = fixed
        La, Lb = var_label[a], var_label[b]
        if (La, t, Lb) not in sch.triples:
            if (Lb, t, La) in sch.triples:
                warnings.append(f"direction fixed: ({Lb})-[:{t}]->({La})")
                p["from"], p["to"] = b, a
            else:
                ok = [f"({x})-[:{t}]->({y})" for x, tt, y in sorted(sch.triples) if tt == t]
                errors.append(f"({La})-[:{t}]->({Lb}) does not exist; {t} connects "
                              + ", ".join(ok))
                continue
        if p.get("var"):
            if not _ID.match(str(p["var"])) or p["var"] in var_label:
                errors.append(f"relationship variable {p['var']!r} is invalid or reused")
            else:
                rel_var_type[p["var"]] = t
        parent[find(a)] = find(b)

    if len(var_label) > 1 and len({find(v) for v in var_label}) > 1 and not errors:
        groups = {}
        for v in var_label:
            groups.setdefault(find(v), []).append(v)
        errors.append("the pattern is not connected (would multiply unrelated rows): "
                      + " | ".join(", ".join(g) for g in groups.values())
                      + "; add a path between them")

    # ---- expressions ----
    def check_expr(e, where, aliases=(), allow_star=False):
        """Returns the (possibly fixed) expression string or None."""
        if e == "*" and allow_star:
            return e
        m = _EXPR.match(str(e or ""))
        if not m:
            errors.append(f"{where}: '{e}' is not 'var' or 'var.property'")
            return None
        v, prop = m.group(1), m.group(2)
        if prop is None:
            if v in var_label or v in rel_var_type or v in aliases:
                return e
            errors.append(f"{where}: unknown variable '{v}'")
            return None
        if v in var_label:
            owner, props = var_label[v], sch.node_props.get(var_label[v], {})
        elif v in rel_var_type:
            owner, props = rel_var_type[v], sch.rel_props.get(rel_var_type[v], {})
        else:
            errors.append(f"{where}: unknown variable '{v}'")
            return None
        fixed = _fix_case(prop, props)
        if fixed is None:
            has = sorted(L for L, ps in sch.node_props.items() if prop in ps)
            sug = _close(prop, props)
            msg = f"{where}: {owner} has no property '{prop}'"
            msg += f"; did you mean {', '.join(sug)}?" if sug else \
                f"; its properties are: {', '.join(sorted(props)) or 'none'}"
            if has:
                msg += f" ('{prop}' exists on {', '.join(has)})"
            errors.append(msg)
            return None
        if fixed != prop:
            warnings.append(f"property '{v}.{prop}' -> '{v}.{fixed}'")
        return f"{v}.{fixed}"

    def check_value(expr, op, value, where):
        """Value-catalog check for exact comparisons on text properties."""
        if not cat or op not in ("=", "<>", "IN") or "." not in str(expr):
            return value
        v, prop = expr.split(".")
        key = f"{var_label.get(v, rel_var_type.get(v))}.{prop}"
        known = (cat.get(key) or {}).get("values")
        if not known:
            return value
        vals = value if isinstance(value, list) else [value]
        out = []
        for x in vals:
            if not isinstance(x, str) or x in known:
                out.append(x)
                continue
            fixed = _fix_case(x, known)
            if fixed is not None:
                warnings.append(f"value {x!r} -> {fixed!r} ({key})")
                out.append(fixed)
                continue
            if len(known) <= 50:
                hint = "existing values: " + ", ".join(repr(k) for k in list(known)[:50])
            else:
                sug = _close(x, list(known)[:20000])
                hint = ("closest existing values: " + ", ".join(repr(s) for s in sug)) \
                    if sug else "no similar value exists"
            errors.append(f"{where}: {key} has no value {x!r}; {hint}")
            out.append(x)
        return out if isinstance(value, list) else out[0]

    def check_filter(f, where, aliases=()):
        if "or" in f:
            if not isinstance(f["or"], list) or len(f["or"]) < 2:
                errors.append(f"{where}: 'or' needs a list of at least two conditions")
                return
            for i, g in enumerate(f["or"]):
                check_filter(g, f"{where}.or[{i}]", aliases)
            return
        op = str(f.get("op", "")).upper().strip()
        if op not in OPS:
            errors.append(f"{where}: operator {f.get('op')!r} not allowed; use one of "
                          + ", ".join(sorted(OPS)))
            return
        f["op"] = op
        e = check_expr(f.get("expr"), where, aliases)
        if e is None:
            return
        f["expr"] = e
        if op in ("IS NULL", "IS NOT NULL"):
            f.pop("value", None)
            return
        if "value" not in f:
            errors.append(f"{where}: missing 'value'")
            return
        if op == "IN" and not isinstance(f["value"], list):
            f["value"] = [f["value"]]
        f["value"] = check_value(e, op, f["value"], where)

    for i, f in enumerate(ir.get("filters") or []):
        check_filter(f, f"filters[{i}]")

    # ---- with (one intermediate aggregation) ----
    stage_aliases = set()
    kept_vars = set()
    group_map = {}          # "u.state" -> "state"
    w = ir.get("with")
    if w:
        for i, g in enumerate(w.get("group_by") or []):
            e = check_expr(g.get("expr"), f"with.group_by[{i}]")
            if e is None:
                continue
            g["expr"] = e
            if "." in e:
                alias = g.get("as") or e.replace(".", "_")
                if not _ID.match(alias):
                    errors.append(f"with.group_by[{i}]: alias {alias!r} is not a valid name")
                    continue
                g["as"] = alias
                group_map[e] = alias
                stage_aliases.add(alias)
            else:
                kept_vars.add(e)
        for i, a in enumerate(w.get("aggregates") or []):
            _check_agg(a, f"with.aggregates[{i}]", check_expr, errors)
            if a.get("as"):
                stage_aliases.add(a["as"])
        if not w.get("aggregates"):
            errors.append("with: needs at least one aggregate (otherwise leave 'with' out)")
        for i, h in enumerate(w.get("having") or []):
            if h.get("expr") not in stage_aliases:
                errors.append(f"with.having[{i}]: must use an alias defined in 'with' "
                              f"({', '.join(sorted(stage_aliases)) or 'none'})")
            elif str(h.get("op", "")).upper() not in OPS - {"IN"}:
                errors.append(f"with.having[{i}]: operator {h.get('op')!r} not allowed")

    # ---- return ----
    ret = ir.get("return") or []
    if not ret:
        errors.append('"return" is empty')
    aliases_out = set()

    def check_after_with(e, where):
        """After 'with' only the kept variables and the with-aliases exist."""
        if not w:
            return check_expr(e, where, allow_star=True)
        if e in stage_aliases or e == "*":
            return e
        if e in group_map:
            return group_map[e]
        v = str(e).split(".")[0]
        if v in kept_vars:
            return check_expr(e, where)
        errors.append(f"{where}: '{e}' is not available after 'with'; available: "
                      + ", ".join(sorted(stage_aliases | kept_vars)))
        return None

    for i, r in enumerate(ret):
        where = f"return[{i}]"
        if r.get("agg"):
            agg = str(r["agg"]).lower()
            if agg not in AGGS:
                errors.append(f"{where}: aggregate {r['agg']!r} not allowed; use "
                              + ", ".join(sorted(AGGS)))
                continue
            r["agg"] = agg
            if r.get("expr") == "*" and agg != "count":
                errors.append(f"{where}: only count can use '*'")
                continue
        e = check_after_with(r.get("expr"), where)
        if e is None:
            continue
        r["expr"] = e
        alias = r.get("as")
        if alias is not None:
            if not _ID.match(str(alias)):
                errors.append(f"{where}: alias {alias!r} is not a valid name")
            elif alias in aliases_out:
                errors.append(f"{where}: alias '{alias}' used twice")
            aliases_out.add(alias)

    for i, o in enumerate(ir.get("order_by") or []):
        e = o.get("expr")
        if e in aliases_out:
            continue
        fixed = check_after_with(e, f"order_by[{i}]")
        if fixed is not None:
            o["expr"] = fixed

    lim = ir.get("limit")
    if lim is not None and (not isinstance(lim, int) or not 0 < lim <= 100000):
        errors.append(f"limit {lim!r} must be a positive integer")
    return ir, errors, warnings


def _check_agg(a, where, check_expr, errors):
    agg = str(a.get("agg", "")).lower()
    if agg not in AGGS:
        errors.append(f"{where}: aggregate {a.get('agg')!r} not allowed; use "
                      + ", ".join(sorted(AGGS)))
        return
    a["agg"] = agg
    if not a.get("as") or not _ID.match(str(a["as"])):
        errors.append(f"{where}: needs a valid alias in 'as'")
    if a.get("expr") == "*":
        if agg != "count":
            errors.append(f"{where}: only count can use '*'")
        return
    e = check_expr(a.get("expr"), where)
    if e is not None:
        a["expr"] = e


# ------------------------------------------------------------
# Compilation (6.3)
# ------------------------------------------------------------

def compile_ir(ir, use_params=True):
    """IR (already validated) -> (cypher, params)."""
    params = {}

    def val(v):
        if use_params:
            name = f"p{len(params)}"
            params[name] = v
            return f"${name}"
        return json.dumps(v, ensure_ascii=False).replace('"', "'") \
            if isinstance(v, (str, list)) else json.dumps(v)

    labels = {n["var"]: n["label"] for n in ir["nodes"]}
    seen = set()

    def node(v):
        if v in seen:
            return f"({v})"
        seen.add(v)
        return f"({v}:{_q(labels[v])})"

    patterns = []
    for p in ir.get("paths") or []:
        rv = p.get("var") or ""
        patterns.append(f"{node(p['from'])}-[{rv}:{_q(p['rel'])}]->{node(p['to'])}")
    for v in labels:
        if v not in seen:
            patterns.append(node(v))
    out = ["MATCH " + ", ".join(patterns)]

    def cond(f):
        if "or" in f:
            return "(" + " OR ".join(cond(g) for g in f["or"]) + ")"
        if f["op"] in ("IS NULL", "IS NOT NULL"):
            return f"{f['expr']} {f['op']}"
        return f"{f['expr']} {f['op']} {val(f['value'])}"

    if ir.get("filters"):
        out.append("WHERE " + " AND ".join(cond(f) for f in ir["filters"]))

    def agg_expr(item):
        inner = item["expr"]
        if item.get("distinct") and inner != "*":
            inner = "DISTINCT " + inner
        return f"{item['agg']}({inner})" if item.get("agg") else inner

    w = ir.get("with")
    if w:
        items = []
        for g in w.get("group_by") or []:
            items.append(f"{g['expr']} AS {g['as']}" if "." in g["expr"] else g["expr"])
        items += [f"{agg_expr(a)} AS {a['as']}" for a in w["aggregates"]]
        out.append("WITH " + ", ".join(items))
        if w.get("having"):
            out.append("WHERE " + " AND ".join(
                f"{h['expr']} {h['op'].upper()} {val(h['value'])}" for h in w["having"]))

    ret = []
    for r in ir["return"]:
        e = agg_expr(r)
        ret.append(f"{e} AS {r['as']}" if r.get("as") else e)
    out.append("RETURN " + ("DISTINCT " if ir.get("distinct") else "") + ", ".join(ret))
    if ir.get("order_by"):
        out.append("ORDER BY " + ", ".join(
            o["expr"] + (" DESC" if o.get("desc") else "") for o in ir["order_by"]))
    if ir.get("limit"):
        out.append(f"LIMIT {int(ir['limit'])}")
    return " ".join(out), params


# ------------------------------------------------------------
# Prompting (6.1)
# ------------------------------------------------------------

IR_SYSTEM_PROMPT = r"""
You plan graph database queries. Do NOT write Cypher. Fill in a JSON query
plan that uses ONLY the labels, relationship types (with their direction)
and properties in the given graph schema.

Plan format:
{
  "nodes":    [{"var": "<short name>", "label": "<node label>"}],
  "paths":    [{"from": "<var>", "rel": "<RELATIONSHIP_TYPE>", "to": "<var>", "var": "<optional rel var>"}],
  "filters":  [{"expr": "<var>.<property>", "op": "<operator>", "value": <value>}],
  "with":     {"group_by": [...], "aggregates": [...], "having": [...]},
  "return":   [{"expr": "<var> or <var>.<property> or alias", "agg": "<optional>", "distinct": false, "as": "<alias>"}],
  "distinct": false,
  "order_by": [{"expr": "<alias or var.property>", "desc": true}],
  "limit":    <integer or null>
}

Rules:
- operators: =, <>, <, <=, >, >=, IN, CONTAINS, STARTS WITH, ENDS WITH, IS NULL, IS NOT NULL
- several conditions in "filters" are combined with AND; for OR write
  {"or": [<condition>, <condition>]}
- aggregates: count, sum, avg, min, max, collect; "expr": "*" only with count
- non-aggregated return items are the grouping keys (like Cypher)
- "with" is ONLY for a two-step aggregation, e.g. "average number of movies per
  user": first count per user in "with", then average in "return"
- every node must be connected through "paths"
- use the values listed under "Values that exist in the graph" exactly
- if the question needs something this format cannot express (OPTIONAL
  MATCH, several separate MATCH steps, CASE, sub-queries), answer
  {"unsupported": true}

Example 1 (schema: (Person)-[:WORKS_AT]->(Company), Person{name, age}, Company{name, city})
Question: How many people over 30 work at each company in Berlin? Show the top 5.
{"nodes": [{"var": "p", "label": "Person"}, {"var": "c", "label": "Company"}],
 "paths": [{"from": "p", "rel": "WORKS_AT", "to": "c"}],
 "filters": [{"expr": "p.age", "op": ">", "value": 30}, {"expr": "c.city", "op": "=", "value": "Berlin"}],
 "return": [{"expr": "c.name", "as": "company"}, {"agg": "count", "expr": "p", "as": "people"}],
 "order_by": [{"expr": "people", "desc": true}], "limit": 5}

Example 2 (same schema)
Question: What is the average number of employees per company, for each city?
{"nodes": [{"var": "p", "label": "Person"}, {"var": "c", "label": "Company"}],
 "paths": [{"from": "p", "rel": "WORKS_AT", "to": "c"}],
 "with": {"group_by": [{"expr": "c"}],
          "aggregates": [{"agg": "count", "expr": "p", "as": "employees"}]},
 "return": [{"expr": "c.city", "as": "city"}, {"agg": "avg", "expr": "employees", "as": "avgEmployees"}]}

Return the JSON plan only.
"""


def ir_user_prompt(question, schema, grounding="", feedback=""):
    parts = [f"Graph Schema:\n{schema}"]
    if grounding:
        parts.append(grounding)
    parts.append(f"Question:\n{question}")
    if feedback:
        parts.append(feedback)
    parts.append("Return the JSON plan only.")
    return "\n\n".join(parts)


def feedback_text(errors, previous):
    return ("Your previous plan was rejected:\n"
            + json.dumps(previous, ensure_ascii=False)
            + "\nProblems:\n" + "\n".join(f"- {e}" for e in errors)
            + "\nFix these problems and return the corrected plan.")


def parse_plan(obj):
    """LLM output (dict) -> plan dict, or None if unsupported/empty."""
    if not isinstance(obj, dict):
        return None
    if obj.get("unsupported"):
        return None
    if "plan" in obj and isinstance(obj["plan"], dict):
        obj = obj["plan"]
    return obj if obj.get("nodes") or obj.get("return") else None
