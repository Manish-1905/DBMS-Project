#!/usr/bin/env python3
"""
Value-grounding benchmark: realistic rewrites of how users type values.
(Roadmap Part 5.4)

For every ER question whose gold query compares a property with a value that
the question mentions (e.g. 'Four Christmases', 'US', female), produce
variants where only that mention changes. The graph and the gold query stay
the same, so the correct answer stays the same:

    original   ... watched 'Four Christmases' ...
    lowercase  ... watched four christmases ...
    unquoted   ... watched Four Christmases ...
    typo       ... watched 'Four Chrsitmases' ...
    partial    ... watched 'Bill Traylor' ...           (title shortened)
    alias      ... users from 'USA' / from OH / women ...

Output: Benchmark/MTQ-Auto/value_variants_er.parquet with the usual columns
(question, schema, cypher, data_source, query_type) plus variant,
original_question, target_property, target_value, surface.

Deterministic (no randomness). Commit the parquet and don't regenerate it
once evaluation has started ("freeze the benchmark").

    uv run make_value_variants.py
"""

import argparse
import re

import pandas as pd

from gold_schema import gold_literals
from grounding import _US_STATES

# Values worth grounding (IP addresses, e-mails, phones are copied verbatim).
TARGET_PROPS = ["Movie.title", "Genre.name", "User.country", "User.state",
                "User.gender", "User.firstName", "User.lastName"]

ARTICLES = ("The ", "A ", "An ")

ALIAS = {
    ("User.country", "US"): "USA",
    ("User.gender", "Female"): "women",
    ("User.gender", "Male"): "men",
    ("Genre.name", "Science Fiction"): "Sci-Fi",
    ("Genre.name", "Sci-Fi"): "science fiction",
}


def find_mention(question, value):
    """(start, end, quoted) of the value in the question, ignoring case."""
    m = re.search(r"(['\"])" + re.escape(value) + r"\1", question, re.IGNORECASE)
    if m:
        return m.start(), m.end(), True
    m = re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", question, re.IGNORECASE)
    if m:
        return m.start(), m.end(), False
    return None


def typo(value):
    """Swap two neighbouring letters in the middle of the longest word."""
    words = value.split(" ")
    i = max(range(len(words)), key=lambda k: len(words[k]))
    w = words[i]
    if len(w) < 5:
        return None
    j = len(w) // 2
    if not (w[j].isalpha() and w[j + 1].isalpha()) or w[j] == w[j + 1]:
        return None
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def partial(value):
    if ":" in value and value.index(":") >= 3:
        return value.split(":")[0].strip()
    for a in ARTICLES:
        if value.startswith(a) and len(value) > len(a) + 3:
            return value[len(a):]
    words = value.split()
    if len(words) >= 3:
        return " ".join(words[:-1])
    return None


def alias(prop, value):
    if (prop, value) in ALIAS:
        return ALIAS[(prop, value)]
    if prop == "User.state" and value.lower() in _US_STATES:
        return _US_STATES[value.lower()].upper()
    return None


def variants_for(question, prop, value):
    pos = find_mention(question, value)
    if not pos:
        return []
    start, end, quoted = pos
    surface = question[start + 1:end - 1] if quoted else question[start:end]
    q = lambda s: f"'{s}'" if quoted else s           # keep the question's quoting
    out = [("original", question, surface)]
    if surface.lower() != surface and len(surface) > 3:   # "us" would read as a word
        out.append(("lowercase", question[:start] + surface.lower() + question[end:],
                    surface.lower()))
    if quoted:
        out.append(("unquoted", question[:start] + surface + question[end:], surface))
    t = typo(surface)
    if t:
        out.append(("typo", question[:start] + q(t) + question[end:], t))
    if prop == "Movie.title":
        pt = partial(surface)
        if pt:
            out.append(("partial", question[:start] + q(pt) + question[end:], pt))
    al = alias(prop, value)
    if al:
        out.append(("alias", question[:start] + q(al) + question[end:], al))
    return out


def build(df):
    rows = []
    for _, r in df.iterrows():
        lits = gold_literals(r["cypher"])
        target = next(((p, v) for p in TARGET_PROPS for (lp, v) in sorted(lits)
                       if lp == p and find_mention(r["question"], v)), None)
        if not target:
            continue
        prop, value = target
        for variant, question, surface in variants_for(r["question"], prop, value):
            rows.append({
                "question": question, "schema": r["schema"], "cypher": r["cypher"],
                "data_source": r["data_source"], "query_type": r["query_type"],
                "variant": variant, "original_question": r["question"],
                "target_property": prop, "target_value": value, "surface": surface,
            })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="Benchmark/MTQ-Auto/test-split.parquet")
    p.add_argument("--data-source", default="er")
    p.add_argument("--out", default="Benchmark/MTQ-Auto/value_variants_er.parquet")
    a = p.parse_args()
    df = pd.read_parquet(a.parquet)
    df = df[df["data_source"] == a.data_source]
    out = build(df)
    out.to_parquet(a.out, index=False)
    print(f"{out['original_question'].nunique()} questions -> {len(out)} rows")
    print(out.groupby(["variant"]).size().to_string())
    print()
    print(out.groupby(["target_property"]).size().to_string())
    print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
