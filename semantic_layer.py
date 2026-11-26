#!/usr/bin/env python3
"""
Semantic layer for a graph schema. (Roadmap Part 3.2)

For every node label, relationship type and property, store a short
description and a list of aliases (other words a user might use):

    {
      "data_source": "er",
      "schema_version": "46797f943d798546",
      "labels":        {"Movie": {"description": "...", "aliases": ["film", ...]}},
      "relationships": {"WATCHED": {"description": "...", "aliases": ["viewed", ...]}},
      "properties":    {"Movie.year": {"description": "...", "aliases": ["release year"]}}
    }

The retriever uses aliases and descriptions, so a question about "films"
can still find the `Movie` label.

Three sources, merged in this order (later wins):
    1. rules    - always on: split identifier names ("firstName" -> "first name")
    2. llm      - optional (--llm): ask the local llama-server, one call per
                  label / relationship type, with sample values if available
    3. overrides - optional: a hand-written semantics/<data_source>.overrides.json
                  in the same format, for fixing mistakes

    # schema from the MTQ parquet (works for all 11 graphs, no Neo4j needed)
    uv run semantic_layer.py --data-source er --llm

    # schema from the live graph (includes sample values -> better descriptions)
    uv run schema_inspect.py --neo4j-database $DB --out schemas/er.json
    uv run semantic_layer.py --data-source er --schema-file schemas/er.json --llm

    # every graph in the parquet
    uv run semantic_layer.py --all --llm
"""

import argparse
import json
import re
import time
from pathlib import Path

import requests

from schema_retriever import _split_ident

DEFAULT_LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions"
SEMANTICS_VERSION = "v1"


# ------------------------------------------------------------
# Loading schemas
# ------------------------------------------------------------

def load_schema(data_source, parquet=None, schema_file=None):
    """Return the schema dict (MTQ format, maybe with value_stats)."""
    if schema_file:
        return json.loads(Path(schema_file).read_text())
    import pandas as pd
    df = pd.read_parquet(parquet)
    rows = df[df["data_source"] == data_source]
    if rows.empty:
        raise SystemExit(f"No rows with data_source={data_source!r} in {parquet}")
    return json.loads(rows["schema"].iloc[0])


def schema_elements(schema):
    """Labels, relationship types and properties in a schema."""
    node_props = schema.get("node_props", {})
    rel_props = schema.get("rel_props", {})
    rels = schema.get("relationships", [])
    labels = sorted(set(node_props) | {r["start"] for r in rels} | {r["end"] for r in rels})
    rtypes = sorted({r["type"] for r in rels} | set(rel_props))
    return labels, rtypes, node_props, rel_props, rels


# ------------------------------------------------------------
# 1. Rules
# ------------------------------------------------------------

def _words(name):
    return " ".join(_split_ident(name))


def _plural(phrase):
    if phrase.endswith("y") and not phrase.endswith(("ay", "ey", "oy", "uy")):
        return phrase[:-1] + "ies"
    if phrase.endswith(("s", "x", "ch", "sh")):
        return phrase + "es"
    return phrase + "s"


def rule_semantics(schema):
    labels, rtypes, node_props, rel_props, _ = schema_elements(schema)
    out = {"labels": {}, "relationships": {}, "properties": {}}
    for L in labels:
        w = _words(L)
        out["labels"][L] = {"description": "", "aliases": sorted({w, _plural(w)} - {L.lower()})}
        for p in node_props.get(L, []):
            pw = _words(p["property"])
            out["properties"][f"{L}.{p['property']}"] = {
                "description": "", "aliases": [pw] if pw != p["property"].lower() else []}
    for T in rtypes:
        w = _words(T)
        out["relationships"][T] = {"description": "", "aliases": [w] if w != T.lower() else []}
        for p in rel_props.get(T, []):
            pw = _words(p["property"])
            out["properties"][f"{T}.{p['property']}"] = {
                "description": "", "aliases": [pw] if pw != p["property"].lower() else []}
    return out


# ------------------------------------------------------------
# 2. LLM
# ------------------------------------------------------------

SYSTEM_PROMPT = """You document graph database schemas for a search system.
Given one element of a Neo4j graph schema and its context, return JSON only:

{
  "description": "one short sentence saying what this element represents",
  "aliases": ["3 to 6 other words or short phrases a user might use for it"],
  "properties": {
    "<property name>": {"description": "one short sentence", "aliases": ["1 to 4 other words"]}
  }
}

Rules: aliases are lower-case natural words (synonyms, plurals, everyday
terms), never Cypher. Include every listed property under "properties".
Do not invent properties. Return valid JSON only."""


def _values_hint(schema, owner, prop):
    vs = schema.get("value_stats", {}).get(f"{owner}.{prop}")
    if not vs:
        return ""
    if vs.get("top_values"):
        vals = [str(v["value"]) for v in vs["top_values"][:5]]
        return f" (e.g. {', '.join(vals)})"
    if vs.get("min") is not None:
        return f" (range {vs['min']} to {vs['max']})"
    return ""


def _label_prompt(schema, L):
    _, _, node_props, _, rels = schema_elements(schema)
    props = "\n".join(
        f"  - {p['property']}: {p['datatype']}{_values_hint(schema, L, p['property'])}"
        for p in node_props.get(L, [])) or "  (none)"
    links = "\n".join(f"  ({r['start']})-[:{r['type']}]->({r['end']})"
                      for r in rels if L in (r["start"], r["end"])) or "  (none)"
    return f"Node label: {L}\nProperties:\n{props}\nRelationships:\n{links}"


def _rel_prompt(schema, T):
    _, _, _, rel_props, rels = schema_elements(schema)
    props = "\n".join(f"  - {p['property']}: {p['datatype']}"
                      for p in rel_props.get(T, [])) or "  (none)"
    ends = "\n".join(f"  ({r['start']})-[:{T}]->({r['end']})"
                     for r in rels if r["type"] == T)
    return f"Relationship type: {T}\nConnects:\n{ends}\nProperties:\n{props}"


def _ask(llama_url, prompt, timeout=120):
    payload = {
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 700,
        "response_format": {"type": "json_object"},
    }
    r = requests.post(llama_url, json=payload, timeout=timeout)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"].strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        return json.loads(text[s:e + 1]) if s != -1 and e > s else {}


def _clean_aliases(xs, own_name):
    out = []
    for a in xs if isinstance(xs, list) else []:
        a = str(a).strip().lower()
        if a and a != own_name.lower() and len(a) <= 40 and a not in out:
            out.append(a)
    return out[:8]


def llm_semantics(schema, llama_url, verbose=True):
    labels, rtypes, node_props, rel_props, _ = schema_elements(schema)
    out = {"labels": {}, "relationships": {}, "properties": {}}
    jobs = [("label", L, _label_prompt(schema, L)) for L in labels] + \
           [("rel", T, _rel_prompt(schema, T)) for T in rtypes]
    for i, (kind, name, prompt) in enumerate(jobs, 1):
        t0 = time.perf_counter()
        try:
            ans = _ask(llama_url, prompt)
        except Exception as e:  # keep going; rules still cover this element
            print(f"  [{i}/{len(jobs)}] {name}: LLM failed ({e})")
            continue
        entry = {"description": str(ans.get("description", "")).strip(),
                 "aliases": _clean_aliases(ans.get("aliases"), name)}
        out["labels" if kind == "label" else "relationships"][name] = entry
        known = {p["property"] for p in
                 (node_props.get(name, []) if kind == "label" else rel_props.get(name, []))}
        for prop, pv in (ans.get("properties") or {}).items():
            if prop in known and isinstance(pv, dict):  # drop invented properties
                out["properties"][f"{name}.{prop}"] = {
                    "description": str(pv.get("description", "")).strip(),
                    "aliases": _clean_aliases(pv.get("aliases"), prop)}
        if verbose:
            print(f"  [{i}/{len(jobs)}] {name}: {entry['aliases']} "
                  f"({time.perf_counter() - t0:.1f}s)")
    return out


# ------------------------------------------------------------
# Merge + save
# ------------------------------------------------------------

def merge(*layers):
    out = {"labels": {}, "relationships": {}, "properties": {}}
    for layer in layers:
        for section in out:
            for name, v in (layer or {}).get(section, {}).items():
                cur = out[section].setdefault(name, {"description": "", "aliases": []})
                if v.get("description"):
                    cur["description"] = v["description"]
                for a in v.get("aliases", []):
                    if a not in cur["aliases"]:
                        cur["aliases"].append(a)
    return out


def build(data_source, parquet, schema_file=None, use_llm=False,
          llama_url=DEFAULT_LLAMA_URL, out_dir="semantics"):
    from schema_inspect import schema_version
    schema = load_schema(data_source, parquet, schema_file)
    print(f"{data_source}: {len(schema_elements(schema)[0])} labels, "
          f"{len(schema_elements(schema)[1])} relationship types")
    layers = [rule_semantics(schema)]
    if use_llm:
        layers.append(llm_semantics(schema, llama_url))
    overrides = Path(out_dir) / f"{data_source}.overrides.json"
    if overrides.exists():
        print(f"  applying {overrides}")
        layers.append(json.loads(overrides.read_text()))
    sem = merge(*layers)
    sem = {"data_source": data_source,
           "schema_version": schema_version(schema),
           "semantics_version": SEMANTICS_VERSION,
           "sources": ["rules"] + (["llm"] if use_llm else []) +
                      (["overrides"] if overrides.exists() else []),
           **sem}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{data_source}.json"
    path.write_text(json.dumps(sem, indent=2, ensure_ascii=False))
    print(f"  written to {path}")
    return sem


def load_semantics(data_source, sem_dir="semantics"):
    """Used by the retriever / main.py. Returns None if there is no file."""
    if not sem_dir:
        return None
    path = Path(sem_dir) / f"{data_source}.json"
    return json.loads(path.read_text()) if path.exists() else None


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-source", help="MTQ data_source, e.g. er")
    p.add_argument("--all", action="store_true", help="every graph in the parquet")
    p.add_argument("--parquet", default="Benchmark/MTQ-Auto/test.parquet")
    p.add_argument("--schema-file", help="use this schema JSON (from schema_inspect.py)")
    p.add_argument("--llm", action="store_true", help="also ask llama-server")
    p.add_argument("--llama-url", default=DEFAULT_LLAMA_URL)
    p.add_argument("--out-dir", default="semantics")
    a = p.parse_args()

    if a.all:
        import pandas as pd
        sources = sorted(pd.read_parquet(a.parquet)["data_source"].unique())
    elif a.data_source:
        sources = [a.data_source]
    else:
        p.error("give --data-source or --all")
    if a.schema_file and len(sources) > 1:
        p.error("--schema-file works with a single --data-source")

    for ds in sources:
        build(ds, a.parquet, a.schema_file, a.llm, a.llama_url, a.out_dir)


if __name__ == "__main__":
    main()
