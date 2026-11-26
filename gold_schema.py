#!/usr/bin/env python3
"""
Extract the "gold schema set" of a Cypher query: the labels, relationship
types and properties it actually uses. (Roadmap Part 4.4)

This is what a schema retriever must recover for the LLM to be able to
answer the question, so it is the ground truth for retrieval recall.

    from gold_schema import gold_schema_set
    gold_schema_set("MATCH (u:User)-[:WATCHED]->(m:Movie) RETURN m.title")
    # {'labels': {'User', 'Movie'}, 'rels': {'WATCHED'}, 'props': {'Movie.title'}}

Regex based, so it is approximate, but good enough for MTQ-style queries.
"""

import re

_IDENT = r"`[^`]+`|[A-Za-z_][A-Za-z0-9_]*"

# (var:Label:Label2 ...   or   (:Label
_NODE = re.compile(
    rf"\(\s*({_IDENT})?\s*((?::\s*(?:{_IDENT})\s*)+)"
)
# [var:TYPE|TYPE2 ...   or   [:TYPE
_REL = re.compile(
    rf"\[\s*({_IDENT})?\s*:\s*((?:{_IDENT})(?:\s*\|\s*:?\s*(?:{_IDENT}))*)"
)
# var.prop
_PROP = re.compile(rf"\b({_IDENT})\s*\.\s*({_IDENT})")
# inline map inside a node pattern: (n:Label {prop: ...})
_INLINE = re.compile(
    rf"\(\s*({_IDENT})?\s*((?::\s*(?:{_IDENT})\s*)+)\{{([^}}]*)\}}"
)
_STRINGS = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")


def _clean(name):
    return name.strip().strip("`")


def gold_schema_set(cypher):
    """Return {'labels': set, 'rels': set, 'props': set('Label.prop')}."""
    if not cypher:
        return {"labels": set(), "rels": set(), "props": set()}

    # Remove string literals so "a:b" inside quotes is not parsed.
    q = _STRINGS.sub("''", cypher)

    labels, rels, props = set(), set(), set()
    var_to_names = {}  # variable -> set of labels / rel types

    for m in _NODE.finditer(q):
        var = _clean(m.group(1)) if m.group(1) else None
        names = {_clean(x) for x in re.findall(_IDENT, m.group(2))}
        labels |= names
        if var:
            var_to_names.setdefault(var, set()).update(names)

    for m in _REL.finditer(q):
        var = _clean(m.group(1)) if m.group(1) else None
        names = {_clean(x) for x in re.findall(_IDENT, m.group(2))}
        rels |= names
        if var:
            var_to_names.setdefault(var, set()).update(names)

    for m in _PROP.finditer(q):
        var, prop = _clean(m.group(1)), _clean(m.group(2))
        for owner in var_to_names.get(var, ()):
            props.add(f"{owner}.{prop}")

    for m in _INLINE.finditer(q):
        owners = {_clean(x) for x in re.findall(_IDENT, m.group(2))}
        for key in re.findall(rf"({_IDENT})\s*:", m.group(3)):
            for owner in owners:
                props.add(f"{owner}.{_clean(key)}")

    return {"labels": labels, "rels": rels, "props": props}


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or (
        "MATCH (u:User)-[w:WATCHED]->(m:Movie)-[:HAS]->(g:Genre) "
        "WHERE u.state = 'CA' RETURN g.name, count(*)"
    )
    print(q)
    for k, v in gold_schema_set(q).items():
        print(f"  {k:6}: {sorted(v)}")
