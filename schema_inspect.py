#!/usr/bin/env python3
"""
Extract the schema of the graph currently loaded in Neo4j.
(Roadmap Part 3: "schema = graph.inspect()")

Output uses the same JSON format as MTQ's `schema` column, plus extras:
    {
      "node_props":    {Label: [{"property", "datatype"}, ...]},
      "rel_props":     {TYPE:  [{"property", "datatype"}, ...]},
      "relationships": [{"start", "type", "end"}, ...],
      "indexes":       [...],
      "constraints":   [...],
      "value_stats":   {"Label.prop": {"distinct", "non_null", "top_values" | "min"/"max"}},
      "schema_version": "<sha256 of the structural part>"
    }

schema_version changes whenever labels / relationship types / properties
change, but NOT when only the data values change. (value_stats are about
the graph *state*, which is what Part 5 uses.)

    uv run schema_inspect.py --out schemas/er.json
    uv run schema_inspect.py --out schemas/er.json --compare-mtq --data-source er
    uv run schema_inspect.py --no-values      # skip value stats (faster)
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from neo4j import GraphDatabase, READ_ACCESS


# Neo4j property type names -> the names used in MTQ / LangChain schemas.
# Older Neo4j 5 versions report Java-style names ("String", "Long",
# "StringArray"); newer ones report Cypher types ("STRING NOT NULL",
# "INTEGER", "LIST<STRING NOT NULL>"). Keys here are upper-case, no spaces.
TYPE_MAP = {
    "STRING": "STRING", "LONG": "INTEGER", "INTEGER": "INTEGER", "INT": "INTEGER",
    "DOUBLE": "FLOAT", "FLOAT": "FLOAT", "BOOLEAN": "BOOLEAN",
    "DATE": "DATE", "DATETIME": "DATE_TIME", "ZONEDDATETIME": "DATE_TIME",
    "ZONED_DATETIME": "DATE_TIME", "LOCALDATETIME": "LOCAL_DATE_TIME",
    "LOCAL_DATETIME": "LOCAL_DATE_TIME", "TIME": "TIME", "ZONEDTIME": "TIME",
    "ZONED_TIME": "TIME", "LOCALTIME": "LOCAL_TIME", "LOCAL_TIME": "LOCAL_TIME",
    "DURATION": "DURATION", "POINT": "POINT",
}

# Only keep the most common values when a property has this few distinct values.
MAX_CATEGORICAL = 50
TOP_K_VALUES = 20


def _type(types):
    if not types:
        return "UNKNOWN"
    t = str(types[0]).strip()
    t = re.sub(r"\s+NOT\s+NULL$", "", t, flags=re.IGNORECASE)
    u = re.sub(r"\s+", "_", t.upper())
    if u.endswith("ARRAY") or u.startswith("LIST"):
        return "LIST"
    return TYPE_MAP.get(u, u)


def _q(name):
    """Backtick-quote an identifier for use in Cypher."""
    return "`" + name.replace("`", "``") + "`"


class GraphInspector:
    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def run(self, cypher, **params):
        with self.driver.session(database=self.database,
                                 default_access_mode=READ_ACCESS) as s:
            return s.run(cypher, **params).data()

    # ---------------- structure ----------------

    def node_props(self):
        out = {}
        for r in self.run("CALL db.schema.nodeTypeProperties()"):
            for label in r["nodeLabels"]:
                props = out.setdefault(label, [])
                if r["propertyName"] and not any(
                        p["property"] == r["propertyName"] for p in props):
                    props.append({"property": r["propertyName"],
                                  "datatype": _type(r["propertyTypes"])})
        return out

    def rel_props(self):
        out = {}
        for r in self.run("CALL db.schema.relTypeProperties()"):
            rtype = r["relType"].lstrip(":").strip("`")
            props = out.setdefault(rtype, [])
            if r["propertyName"] and not any(
                    p["property"] == r["propertyName"] for p in props):
                props.append({"property": r["propertyName"],
                              "datatype": _type(r["propertyTypes"])})
        return {k: v for k, v in out.items() if v}

    def relationships(self):
        """Actual (start label, type, end label) triples, with direction."""
        rows = self.run(
            "MATCH (a)-[r]->(b) "
            "UNWIND labels(a) AS s UNWIND labels(b) AS e "
            "RETURN DISTINCT s AS start, type(r) AS type, e AS end"
        )
        return sorted(rows, key=lambda r: (r["start"], r["type"], r["end"]))

    def indexes(self):
        try:
            rows = self.run("SHOW INDEXES YIELD name, type, entityType, "
                            "labelsOrTypes, properties")
        except Exception:
            return []
        return sorted(rows, key=lambda r: r["name"])

    def constraints(self):
        try:
            rows = self.run("SHOW CONSTRAINTS YIELD name, type, entityType, "
                            "labelsOrTypes, properties")
        except Exception:
            return []
        return sorted(rows, key=lambda r: r["name"])

    # ---------------- values (graph state) ----------------

    def value_stats(self, node_props):
        stats = {}
        for label, props in node_props.items():
            for p in props:
                key = f"{label}.{p['property']}"
                L, P = _q(label), _q(p["property"])
                base = self.run(
                    f"MATCH (n:{L}) WHERE n.{P} IS NOT NULL "
                    f"RETURN count(*) AS non_null, count(DISTINCT n.{P}) AS distinct"
                )[0]
                entry = dict(base)
                if p["datatype"] == "STRING" and 0 < base["distinct"] <= MAX_CATEGORICAL:
                    entry["top_values"] = self.run(
                        f"MATCH (n:{L}) WHERE n.{P} IS NOT NULL "
                        f"RETURN n.{P} AS value, count(*) AS freq "
                        f"ORDER BY freq DESC LIMIT $k", k=TOP_K_VALUES)
                elif p["datatype"] in ("INTEGER", "FLOAT", "DATE", "DATE_TIME"):
                    mm = self.run(
                        f"MATCH (n:{L}) WHERE n.{P} IS NOT NULL "
                        f"RETURN min(n.{P}) AS min, max(n.{P}) AS max")[0]
                    entry.update({k: str(v) if v is not None else None
                                  for k, v in mm.items()})
                stats[key] = entry
        return stats

    # ---------------- all together ----------------

    def inspect(self, with_values=True):
        schema = {
            "node_props": self.node_props(),
            "rel_props": self.rel_props(),
            "relationships": self.relationships(),
            "indexes": self.indexes(),
            "constraints": self.constraints(),
        }
        schema["schema_version"] = schema_version(schema)
        if with_values:
            schema["value_stats"] = self.value_stats(schema["node_props"])
        return schema


def schema_version(schema):
    """Deterministic hash of the structural schema (order independent)."""
    structural = {
        "node_props": {k: sorted((p["property"], p["datatype"]) for p in v)
                       for k, v in schema.get("node_props", {}).items()},
        "rel_props": {k: sorted((p["property"], p["datatype"]) for p in v)
                      for k, v in schema.get("rel_props", {}).items()},
        "relationships": sorted((r["start"], r["type"], r["end"])
                                for r in schema.get("relationships", [])),
    }
    blob = json.dumps(structural, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def compare_with_mtq(extracted, mtq_schema):
    """Part 3.5: how much of MTQ's schema did we recover?"""
    mtq = json.loads(mtq_schema) if isinstance(mtq_schema, str) else mtq_schema

    def sets(s):
        return {
            "labels": set(s.get("node_props", {})),
            "rels": {r["type"] for r in s.get("relationships", [])},
            "directions": {(r["start"], r["type"], r["end"])
                           for r in s.get("relationships", [])},
            "props": {(L, p["property"]) for L, ps in s.get("node_props", {}).items()
                      for p in ps},
            "types": {(L, p["property"], p["datatype"])
                      for L, ps in s.get("node_props", {}).items() for p in ps},
        }

    got, gold = sets(extracted), sets(mtq)
    report = {}
    for k in gold:
        g = gold[k]
        report[f"{k}_recall"] = (len(g & got[k]) / len(g)) if g else 1.0
        missing = g - got[k]
        if missing:
            report[f"{k}_missing"] = sorted(map(str, missing))[:20]
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687"))
    p.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    p.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", ""))
    p.add_argument("--neo4j-database", default="neo4j")
    p.add_argument("--out", help="write schema JSON here (e.g. schemas/er.json)")
    p.add_argument("--no-values", action="store_true", help="skip value statistics")
    p.add_argument("--compare-mtq", action="store_true",
                   help="compare against the MTQ schema for --data-source")
    p.add_argument("--parquet", default="Benchmark/MTQ-Auto/test.parquet")
    p.add_argument("--data-source", help="MTQ data_source of the loaded graph, e.g. er")
    a = p.parse_args()

    if not a.neo4j_password:
        p.error("No Neo4j password. Set NEO4J_PASSWORD or pass --neo4j-password.")

    gi = GraphInspector(a.neo4j_uri, a.neo4j_user, a.neo4j_password, a.neo4j_database)
    try:
        schema = gi.inspect(with_values=not a.no_values)
    finally:
        gi.close()

    print(f"schema_version: {schema['schema_version']}")
    print(f"labels: {len(schema['node_props'])}  "
          f"relationship triples: {len(schema['relationships'])}  "
          f"indexes: {len(schema['indexes'])}  constraints: {len(schema['constraints'])}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(schema, indent=2, default=str))
        print(f"written to {a.out}")

    if a.compare_mtq:
        if not a.data_source:
            p.error("--compare-mtq needs --data-source")
        import pandas as pd
        df = pd.read_parquet(a.parquet)
        mtq_schema = df[df["data_source"] == a.data_source]["schema"].iloc[0]
        print(json.dumps(compare_with_mtq(schema, mtq_schema), indent=2))


if __name__ == "__main__":
    main()
