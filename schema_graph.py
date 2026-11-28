import argparse
import json
import os

from neo4j import GraphDatabase, WRITE_ACCESS

from semantic_layer import load_schema, load_semantics, schema_elements
from schema_inspect import schema_version


def build_rows(schema, semantics, data_source):
    sem = semantics or {}
    labels, rtypes, node_props, rel_props, rels = schema_elements(schema)
    stats = schema.get("value_stats", {})

    def s(section, name):
        e = sem.get(section, {}).get(name) or {}
        return e.get("description", ""), e.get("aliases", [])

    def prop_row(owner, p):
        key = f"{owner}.{p['property']}"
        desc, aliases = s("properties", key)
        vs = stats.get(key, {})
        return {"owner": owner, "name": p["property"], "datatype": p["datatype"],
                "description": desc, "aliases": aliases,
                "distinct": vs.get("distinct"), "non_null": vs.get("non_null"),
                "min": vs.get("min"), "max": vs.get("max"),
                "top_values": json.dumps(vs["top_values"], default=str)
                if vs.get("top_values") else None}

    node_types = [{"name": L, "description": s("labels", L)[0],
                   "aliases": s("labels", L)[1]} for L in labels]
    rel_types = [{"name": T, "description": s("relationships", T)[0],
                  "aliases": s("relationships", T)[1]} for T in rtypes]
    node_prop_rows = [prop_row(L, p) for L in labels for p in node_props.get(L, [])]
    rel_prop_rows = [prop_row(T, p) for T in rtypes for p in rel_props.get(T, [])]
    triples = [{"start": r["start"], "type": r["type"], "end": r["end"]} for r in rels]
    return {"ds": data_source, "version": schema_version(schema),
            "node_types": node_types, "rel_types": rel_types,
            "node_props": node_prop_rows, "rel_props": rel_prop_rows,
            "triples": triples}


LOAD_STATEMENTS = [
    "MATCH (n {data_source: $ds}) DETACH DELETE n",
    "CREATE (:Schema {data_source: $ds, version: $version, loaded_at: datetime()})",
    """UNWIND $node_types AS nt
       MATCH (s:Schema {data_source: $ds})
       CREATE (s)-[:HAS_NODE_TYPE]->(:NodeType {data_source: $ds, name: nt.name,
               description: nt.description, aliases: nt.aliases})""",
    """UNWIND $rel_types AS rt
       CREATE (:RelationshipType {data_source: $ds, name: rt.name,
               description: rt.description, aliases: rt.aliases})""",
    """UNWIND $node_props AS p
       MATCH (n:NodeType {data_source: $ds, name: p.owner})
       CREATE (n)-[:HAS_PROPERTY]->(:Property {data_source: $ds, owner: p.owner,
               name: p.name, datatype: p.datatype, description: p.description,
               aliases: p.aliases, distinct: p.distinct, non_null: p.non_null,
               min: p.min, max: p.max, top_values: p.top_values})""",
    """UNWIND $rel_props AS p
       MATCH (r:RelationshipType {data_source: $ds, name: p.owner})
       CREATE (r)-[:HAS_PROPERTY]->(:Property {data_source: $ds, owner: p.owner,
               name: p.name, datatype: p.datatype, description: p.description,
               aliases: p.aliases})""",
    """UNWIND $triples AS t
       MATCH (a:NodeType {data_source: $ds, name: t.start}),
             (r:RelationshipType {data_source: $ds, name: t.type}),
             (b:NodeType {data_source: $ds, name: t.end})
       MERGE (a)-[:HAS_RELATIONSHIP]->(r)
       CREATE (r)-[:CONNECTS {from: t.start, to: t.end}]->(b)""",
]

PATH_QUERY = """
MATCH (a:NodeType {data_source: $ds, name: $src}),
      (b:NodeType {data_source: $ds, name: $dst}),
      p = shortestPath((a)-[:HAS_RELATIONSHIP|CONNECTS*..8]-(b))
RETURN [n IN nodes(p) | n.name] AS path
"""


def ensure_database(driver, name):
    try:
        driver.execute_query(f"CREATE DATABASE `{name}` IF NOT EXISTS WAIT",
                             database_="system")
    except Exception as e:
        raise SystemExit(
            f"Could not create database '{name}' ({e}).\n"
            f"Create it yourself in Neo4j Desktop / Browser "
            f"(`CREATE DATABASE {name}`) or pass --target-database <existing db>.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-source", required=True)
    p.add_argument("--parquet", default="Benchmark/MTQ-Auto/test.parquet")
    p.add_argument("--schema-file", help="schema JSON from schema_inspect.py "
                   "(recommended: includes value statistics)")
    p.add_argument("--semantics-dir", default="semantics")
    p.add_argument("--target-database", default="schemagraph",
                   help="database the schema graph is written to")
    p.add_argument("--data-database", default=os.environ.get("DB", "neo4j"),
                   help="the benchmark database (only used to refuse writing into it)")
    p.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687"))
    p.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    p.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", ""))
    p.add_argument("--dry-run", action="store_true", help="print, don't write")
    a = p.parse_args()

    schema = load_schema(a.data_source, a.parquet, a.schema_file)
    semantics = load_semantics(a.data_source, a.semantics_dir)
    rows = build_rows(schema, semantics, a.data_source)
    print(f"{a.data_source}: schema_version {rows['version']}, "
          f"{len(rows['node_types'])} node types, {len(rows['rel_types'])} relationship "
          f"types, {len(rows['node_props']) + len(rows['rel_props'])} properties, "
          f"semantics={'yes' if semantics else 'no'}")

    if a.dry_run:
        for t in rows["triples"]:
            print(f"  (:NodeType {t['start']})-[:HAS_RELATIONSHIP]->"
                  f"(:RelationshipType {t['type']})-[:CONNECTS]->(:NodeType {t['end']})")
        return

    if a.target_database == a.data_database:
        p.error(f"refusing to write the schema graph into the benchmark database "
                f"'{a.data_database}'; pick another --target-database")
    if not a.neo4j_password:
        p.error("No Neo4j password. Set NEO4J_PASSWORD or pass --neo4j-password.")

    driver = GraphDatabase.driver(a.neo4j_uri, auth=(a.neo4j_user, a.neo4j_password))
    try:
        ensure_database(driver, a.target_database)
        with driver.session(database=a.target_database,
                            default_access_mode=WRITE_ACCESS) as session:
            def load(tx):
                for stmt in LOAD_STATEMENTS:
                    tx.run(stmt, **rows).consume()
            session.execute_write(load)

        counts = driver.execute_query(
            "MATCH (n {data_source: $ds}) RETURN labels(n)[0] AS kind, count(*) AS n "
            "ORDER BY kind", ds=a.data_source, database_=a.target_database).records
        print(f"Loaded into database '{a.target_database}': "
              + ", ".join(f"{r['kind']}={r['n']}" for r in counts))

        names = [n["name"] for n in rows["node_types"]]
        if len(names) >= 2:
            recs = driver.execute_query(PATH_QUERY, ds=a.data_source, src=names[0],
                                        dst=names[-1], database_=a.target_database).records
            if recs:
                print(f"Example: path {names[0]} -> {names[-1]}: {recs[0]['path']}")
        print(f"Explore it in Neo4j Browser:  :use {a.target_database}  then  "
              f"MATCH (n {{data_source:'{a.data_source}'}}) RETURN n")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
