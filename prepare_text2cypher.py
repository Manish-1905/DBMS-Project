#!/usr/bin/env python3
"""
Convert Neo4j's Text2Cypher 2024 dataset into the same parquet format as
MTQ, so eval_retrieval.py can benchmark the retrievers on it.
(Roadmap Part 4, second primary dataset)

Dataset: https://huggingface.co/datasets/neo4j/text2cypher-2024v1

Its `schema` column is plain text (two or three different layouts), so
this script parses it into MTQ's JSON schema format:
    {"node_props": {...}, "rel_props": {...}, "relationships": [...]}

Output columns: question, schema (JSON), cypher, data_source, query_type
    data_source = the demo database the question belongs to
                  (database_reference_alias), or "other" when it has none
    query_type  = the dataset's own source column (e.g. which generator
                  wrote the question), so --by-type style tables still work

    uv run prepare_text2cypher.py                       # downloads the test split
    uv run prepare_text2cypher.py --split train
    uv run prepare_text2cypher.py --input my_download.parquet
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import requests

DATASET = "neo4j/text2cypher-2024v1"
OUT_DIR = Path("Benchmark/Text2Cypher-2024")

_NAME = r"`[^`]+`|[A-Za-z_][A-Za-z0-9_]*"
_INLINE = re.compile(rf"({_NAME})\s*\{{([^}}]*)\}}")
_INLINE_PROP = re.compile(rf"({_NAME})\s*:\s*([A-Za-z_]+)")
_BULLET_OWNER = re.compile(rf"^\s*-\s*\*\*({_NAME})\*\*")
_BULLET_PROP = re.compile(rf"^\s*-\s*`?({_NAME})`?\s*:\s*([A-Za-z_]+)")
_REL = re.compile(
    rf"\(\s*:?\s*({_NAME})\s*\)\s*-\s*\[\s*:?\s*({_NAME})\s*\]\s*->\s*\(\s*:?\s*({_NAME})\s*\)")


def _clean(name):
    return name.strip().strip("`")


def parse_schema_text(text):
    """Parse a Text2Cypher schema string into MTQ's JSON schema dict."""
    node_props, rel_props, rels = {}, {}, []
    section = None   # "node" | "rel" | "rels"
    owner = None
    for line in str(text).splitlines():
        low = line.lower()
        if "relationship properties" in low:
            section, owner = "rel", None
        elif "node properties" in low or "node labels" in low:
            section, owner = "node", None
        elif "relationships" in low and "(" not in line:
            section, owner = "rels", None

        for a, t, b in _REL.findall(line):
            r = {"start": _clean(a), "type": _clean(t), "end": _clean(b)}
            if r not in rels:
                rels.append(r)
        if section not in ("node", "rel"):
            continue
        target = node_props if section == "node" else rel_props

        # layout 1: Label {prop: TYPE, prop: TYPE}, Label2 {...}
        for name, body in _INLINE.findall(line):
            props = target.setdefault(_clean(name), [])
            for p, dtype in _INLINE_PROP.findall(body):
                props.append({"property": _clean(p), "datatype": dtype.upper()})
        # layout 2: - **Label**   then   - `prop`: TYPE ...
        m = _BULLET_OWNER.match(line)
        if m:
            owner = _clean(m.group(1))
            target.setdefault(owner, [])
            continue
        m = _BULLET_PROP.match(line)
        if m and owner is not None:
            target[owner].append({"property": _clean(m.group(1)),
                                  "datatype": m.group(2).upper()})

    # relationship types that only appear in the patterns
    for r in rels:
        node_props.setdefault(r["start"], [])
        node_props.setdefault(r["end"], [])
    return {"node_props": node_props, "rel_props": rel_props, "relationships": rels}


def download(split):
    """Return a DataFrame with the given split, via the Hugging Face parquet API."""
    urls = []
    try:
        r = requests.get(f"https://huggingface.co/api/datasets/{DATASET}/parquet/default/{split}",
                         timeout=60)
        r.raise_for_status()
        urls = r.json()
    except Exception:
        r = requests.get("https://datasets-server.huggingface.co/parquet",
                         params={"dataset": DATASET}, timeout=60)
        r.raise_for_status()
        urls = [f["url"] for f in r.json()["parquet_files"] if f["split"] == split]
    if not urls:
        raise SystemExit(f"No parquet files found for split {split!r}")
    frames = []
    for u in urls:
        print("Downloading", u)
        frames.append(pd.read_parquet(u))
    return pd.concat(frames, ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--input", help="use an already downloaded parquet/csv instead")
    p.add_argument("--out", help=f"default: {OUT_DIR}/<split>.parquet")
    a = p.parse_args()

    if a.input:
        df = pd.read_csv(a.input) if a.input.endswith(".csv") else pd.read_parquet(a.input)
    else:
        df = download(a.split)
    print(f"{len(df)} rows, columns: {list(df.columns)}")

    schemas = df["schema"].map(parse_schema_text)
    ok = schemas.map(lambda s: bool(s["node_props"]))
    alias = df["database_reference_alias"] if "database_reference_alias" in df else None
    out = pd.DataFrame({
        "question": df["question"],
        "schema": schemas.map(json.dumps),
        "cypher": df["cypher"],
        "data_source": (alias.fillna("other").astype(str) if alias is not None
                        else "other"),
        "query_type": df["data_source"].astype(str) if "data_source" in df else None,
    })[ok.values]

    dst = Path(a.out) if a.out else OUT_DIR / f"{a.split}.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst, index=False)
    print(f"Kept {len(out)} / {len(df)} rows (dropped {int((~ok).sum())} with no parsable schema)")
    print("Rows per data_source:")
    print(out["data_source"].value_counts().to_string())
    print(f"\nWritten to {dst}")


if __name__ == "__main__":
    main()
