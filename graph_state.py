#!/usr/bin/env python3
"""
Graph-state experiment: same schema, same questions, different stored values.
(Roadmap Part 5.5)

A Text2Cypher system that relies on what it "knows" keeps writing the values
it expects ('Female', 'US'); a graph-aware system reads the current graph and
adapts. To test this, make a COPY of the ER graph whose values are recoded:

  state B ("recoded")    User.gender  Female/Male -> F/M
                         User.country US          -> United States
                         Genre.name   Drama       -> DRAMA (upper case)
  state C ("relabelled") User.state   Ohio        -> OH (postal codes)
                         User.gender  Female/Male -> female/male
                         Movie.title  The Matrix  -> Matrix, The

Steps (the original database is never written to):

    # 1. copy the graph into a new database and recode it
    uv run graph_state.py create --source $DB --target erstateb --state B

    # 2. value catalog of the recoded graph
    uv run value_index.py --neo4j-database erstateb --out values/er_stateB.json

    # 3. gold queries rewritten to the new values (questions unchanged)
    uv run graph_state.py rewrite-gold --state B \\
        --parquet Benchmark/MTQ-Auto/test-split.parquet \\
        --out Benchmark/MTQ-Auto/test-split-stateB.parquet

    # 4. static vs graph-aware, on the questions whose values changed
    uv run main.py --parquet Benchmark/MTQ-Auto/test-split-stateB.parquet \\
        --neo4j-database erstateb --mode retrieved_schema ...
    uv run main.py --parquet Benchmark/MTQ-Auto/test-split-stateB.parquet \\
        --neo4j-database erstateb --mode grounded --values-file values/er_stateB.json ...

`create` needs a Neo4j edition that can create databases (Neo4j Desktop's
developer licence can). Otherwise create the database yourself and add
--no-create.
"""

import argparse
import json
import os
import re

from neo4j import GraphDatabase, READ_ACCESS

from gold_schema import literal_spans
from grounding import _US_STATES

# ------------------------------------------------------------
# Graph states: (Label, property) -> transform
# transform is a dict {old: new} or one of "upper", "lower", "postal", "article_last"
# ------------------------------------------------------------

STATES = {
    "B": {
        ("User", "gender"): {"Female": "F", "Male": "M"},
        ("User", "country"): {"US": "United States"},
        ("Genre", "name"): "upper",
    },
    "C": {
        ("User", "state"): "postal",
        ("User", "gender"): {"Female": "female", "Male": "male"},
        ("Movie", "title"): "article_last",
    },
}

_ARTICLES = ("The ", "A ", "An ")


def transform_value(value, how):
    if not isinstance(value, str):
        return value
    if isinstance(how, dict):
        return how.get(value, value)
    if how == "upper":
        return value.upper()
    if how == "lower":
        return value.lower()
    if how == "postal":
        code = _US_STATES.get(value.lower())
        return code.upper() if code else value
    if how == "article_last":
        for a in _ARTICLES:
            if value.startswith(a) and len(value) > len(a):
                return f"{value[len(a):]}, {a.strip()}"
        return value
    raise ValueError(f"unknown transform {how!r}")


def state_mapping(state):
    if state not in STATES:
        raise SystemExit(f"unknown state {state!r}; choose from {sorted(STATES)}")
    return {f"{label}.{prop}": how for (label, prop), how in STATES[state].items()}


# ------------------------------------------------------------
# Gold query rewriting (no Neo4j needed)
# ------------------------------------------------------------

def rewrite_cypher(cypher, mapping):
    """Replace compared literals of mapped properties; returns (cypher, changed)."""
    spans = [s for s in literal_spans(cypher) if s[2] in mapping]
    changed = False
    out, last = [], 0
    for start, end, prop, value in sorted(set((s, e, p, v) for s, e, p, v in spans)):
        if start < last:
            continue
        new = transform_value(value, mapping[prop])
        quote = cypher[start]
        esc = new.replace("\\", "\\\\").replace(quote, "\\" + quote)
        out.append(cypher[last:start] + quote + esc + quote)
        last = end
        changed |= new != value
    out.append(cypher[last:])
    return "".join(out), changed


def rewrite_gold(parquet, out, state, affected_only=False):
    import pandas as pd
    df = pd.read_parquet(parquet)
    mapping = state_mapping(state)
    res = [rewrite_cypher(c, mapping) for c in df["cypher"]]
    df["original_cypher"] = df["cypher"]
    df["cypher"] = [c for c, _ in res]
    df["affected"] = [ch for _, ch in res]
    df["graph_state"] = state
    if affected_only:
        df = df[df["affected"]].reset_index(drop=True)
    df.to_parquet(out, index=False)
    print(f"state {state}: {int(df['affected'].sum())} of {len(df)} gold queries changed")
    ex = df[df["affected"]].head(3)
    for _, r in ex.iterrows():
        print(f"  {r['original_cypher'][:110]}\n  -> {r['cypher'][:110]}\n")
    print(f"written to {out}")


# ------------------------------------------------------------
# Copy + recode (writes ONLY to the target database)
# ------------------------------------------------------------

def _q(name):
    return "`" + name.replace("`", "``") + "`"


def clone(driver, source, target, batch=5000):
    """Copy all nodes and relationships from `source` to (empty) `target`."""
    def wr(cypher, **kw):
        return driver.execute_query(cypher, database_=target, **kw)

    wr("CREATE INDEX clone_cid IF NOT EXISTS FOR (n:`__Clone`) ON (n.`__cid`)")
    wr("CALL db.awaitIndexes(300)")
    n_nodes = 0
    with driver.session(database=source, default_access_mode=READ_ACCESS) as s:
        result = s.run("MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, "
                       "properties(n) AS props")
        groups = {}
        for rec in result:
            key = tuple(sorted(rec["labels"]))
            groups.setdefault(key, []).append({"id": rec["id"], "props": rec["props"]})
            if len(groups[key]) >= batch:
                _write_nodes(wr, key, groups.pop(key))
            n_nodes += 1
        for key, rows in groups.items():
            _write_nodes(wr, key, rows)
    n_rels = 0
    with driver.session(database=source, default_access_mode=READ_ACCESS) as s:
        result = s.run("MATCH (a)-[r]->(b) RETURN elementId(a) AS a, elementId(b) AS b, "
                       "type(r) AS type, properties(r) AS props")
        groups = {}
        for rec in result:
            groups.setdefault(rec["type"], []).append(
                {"a": rec["a"], "b": rec["b"], "props": rec["props"]})
            if len(groups[rec["type"]]) >= batch:
                _write_rels(wr, rec["type"], groups.pop(rec["type"]))
            n_rels += 1
        for t, rows in groups.items():
            _write_rels(wr, t, rows)
    # remove the helper label/property/index
    while True:
        left = wr("MATCH (n:`__Clone`) WITH n LIMIT $b REMOVE n:`__Clone`, n.`__cid` "
                  "RETURN count(*) AS c", b=batch).records[0]["c"]
        if left == 0:
            break
    wr("DROP INDEX clone_cid IF EXISTS")
    return n_nodes, n_rels


def _write_nodes(wr, labels, rows):
    lab = "".join(f":{_q(L)}" for L in labels)
    wr(f"UNWIND $rows AS row CREATE (n{lab}:`__Clone`) SET n = row.props, n.`__cid` = row.id",
       rows=rows)


def _write_rels(wr, rtype, rows):
    wr(f"UNWIND $rows AS row "
       f"MATCH (a:`__Clone` {{`__cid`: row.a}}), (b:`__Clone` {{`__cid`: row.b}}) "
       f"CREATE (a)-[r:{_q(rtype)}]->(b) SET r = row.props", rows=rows)


def recode(driver, target, state):
    """Apply a state's value changes in `target`; returns {prop: changed count}."""
    counts = {}
    for (label, prop), how in STATES[state].items():
        L, P = _q(label), _q(prop)
        values = [r["v"] for r in driver.execute_query(
            f"MATCH (n:{L}) WHERE n.{P} IS NOT NULL RETURN DISTINCT n.{P} AS v",
            database_=target).records]
        m = {v: transform_value(v, how) for v in values if isinstance(v, str)}
        m = {k: v for k, v in m.items() if k != v}
        if not m:
            counts[f"{label}.{prop}"] = 0
            continue
        res = driver.execute_query(
            f"MATCH (n:{L}) WHERE n.{P} IN $old SET n.{P} = $m[n.{P}] "
            f"RETURN count(n) AS c", old=list(m), m=m, database_=target).records
        counts[f"{label}.{prop}"] = res[0]["c"]
    return counts


def create(a):
    if not a.neo4j_password:
        raise SystemExit("No Neo4j password. Set NEO4J_PASSWORD or pass --neo4j-password.")
    if a.target == a.source or not re.fullmatch(r"[a-z][a-z0-9.-]{2,62}", a.target):
        raise SystemExit("--target must be a NEW database name (lower case, e.g. erstateb), "
                         "different from --source")
    if a.target in ("neo4j", "system"):
        raise SystemExit("refusing to write into 'neo4j' or 'system'")
    driver = GraphDatabase.driver(a.neo4j_uri, auth=(a.neo4j_user, a.neo4j_password))
    try:
        if not a.no_create:
            try:
                driver.execute_query(f"CREATE DATABASE `{a.target}` IF NOT EXISTS WAIT",
                                     database_="system")
            except Exception as e:
                raise SystemExit(f"Could not create database '{a.target}' ({e}).\n"
                                 f"Create it in Neo4j Desktop and re-run with --no-create.")
        existing = driver.execute_query("MATCH (n) RETURN count(n) AS c",
                                        database_=a.target).records[0]["c"]
        if existing:
            if not a.replace:
                raise SystemExit(f"database '{a.target}' already has {existing} nodes; "
                                 f"add --replace to wipe and rebuild it")
            while driver.execute_query(
                    "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(*) AS c",
                    database_=a.target).records[0]["c"]:
                pass
        print(f"copying '{a.source}' -> '{a.target}' ...")
        n, r = clone(driver, a.source, a.target)
        print(f"  {n} nodes, {r} relationships copied")
        counts = recode(driver, a.target, a.state)
        print(f"recoded for state {a.state}: " + json.dumps(counts))
        print(f"next: uv run value_index.py --neo4j-database {a.target} "
              f"--out values/er_state{a.state}.json")
    finally:
        driver.close()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("create", help="copy the graph to a new database and recode it")
    c.add_argument("--source", default=os.environ.get("DB", "neo4j"))
    c.add_argument("--target", required=True, help="new database, e.g. erstateb")
    c.add_argument("--state", required=True, choices=sorted(STATES))
    c.add_argument("--no-create", action="store_true",
                   help="the target database already exists (created by hand)")
    c.add_argument("--replace", action="store_true", help="wipe a non-empty target first")
    c.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687"))
    c.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    c.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", ""))

    g = sub.add_parser("rewrite-gold", help="rewrite gold queries for a state")
    g.add_argument("--state", required=True, choices=sorted(STATES))
    g.add_argument("--parquet", default="Benchmark/MTQ-Auto/test-split.parquet")
    g.add_argument("--out", required=True)
    g.add_argument("--affected-only", action="store_true",
                   help="keep only questions whose gold values changed")

    a = p.parse_args()
    if a.cmd == "create":
        create(a)
    else:
        rewrite_gold(a.parquet, a.out, a.state, a.affected_only)


if __name__ == "__main__":
    main()
