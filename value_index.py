#!/usr/bin/env python3
"""
Value catalog: every distinct text value stored in the graph. (Roadmap Part 5.2)

The grounding step looks user phrases up in this catalog ("female" ->
User.gender = 'Female'). It reflects the CURRENT graph state, so rebuild it
whenever the data changes (that's the point of Part 5.5).

    uv run value_index.py --neo4j-database $DB --out values/er.json

Output:
    {
      "database": "neo4j",
      "state_version": "<hash of all values>",   # changes when any value changes
      "properties": {
        "User.gender": {"distinct": 2, "values": {"Female": 25, "Male": 25}},
        "Movie.title": {"distinct": 9125, "values": {...}},
        "User.email":  {"distinct": 50000, "skipped": "too many distinct values"}
      }
    }
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


from schema_inspect import GraphInspector, _q

DEFAULT_MAX_VALUES = 50000


def build_catalog(inspector, max_values=DEFAULT_MAX_VALUES):
    props = {}
    for label, plist in sorted(inspector.node_props().items()):
        for p in plist:
            if p["datatype"] != "STRING":
                continue
            key = f"{label}.{p['property']}"
            L, P = _q(label), _q(p["property"])
            distinct = inspector.run(
                f"MATCH (n:{L}) WHERE n.{P} IS NOT NULL "
                f"RETURN count(DISTINCT n.{P}) AS d")[0]["d"]
            if distinct > max_values:
                props[key] = {"distinct": distinct, "skipped": "too many distinct values"}
                continue
            rows = inspector.run(
                f"MATCH (n:{L}) WHERE n.{P} IS NOT NULL "
                f"RETURN n.{P} AS v, count(*) AS f")
            values = {}
            for r in rows:
                vs = r["v"] if isinstance(r["v"], list) else [r["v"]]
                for v in vs:
                    if isinstance(v, str) and v.strip():
                        values[v] = values.get(v, 0) + r["f"]
            props[key] = {"distinct": distinct,
                          "values": dict(sorted(values.items(), key=lambda x: (-x[1], x[0])))}
    return {"properties": props, "state_version": state_version(props)}


def state_version(props):
    h = hashlib.sha256()
    for key in sorted(props):
        for v in sorted(props[key].get("values", {})):
            h.update(f"{key}\t{v}\n".encode())
    return h.hexdigest()[:16]


def load_catalog(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687"))
    p.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    p.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", ""))
    p.add_argument("--neo4j-database", default=os.environ.get("DB", "neo4j"))
    p.add_argument("--max-values", type=int, default=DEFAULT_MAX_VALUES,
                   help="skip properties with more distinct values than this")
    p.add_argument("--out", required=True, help="e.g. values/er.json")
    a = p.parse_args()
    if not a.neo4j_password:
        p.error("No Neo4j password. Set NEO4J_PASSWORD or pass --neo4j-password.")

    gi = GraphInspector(a.neo4j_uri, a.neo4j_user, a.neo4j_password, a.neo4j_database)
    try:
        cat = build_catalog(gi, a.max_values)
    finally:
        gi.close()
    cat = {"database": a.neo4j_database, **cat}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(cat, indent=1, ensure_ascii=False))
    print(f"state_version {cat['state_version']} from database '{a.neo4j_database}'")
    for key, v in cat["properties"].items():
        note = v.get("skipped") or f"{len(v['values'])} values"
        print(f"  {key:<22} distinct={v['distinct']:<7} {note}")
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()
