import re

_IDENT = r"`[^`]+`|[A-Za-z_][A-Za-z0-9_]*"

_NODE = re.compile(
    rf"\(\s*({_IDENT})?\s*((?::\s*(?:{_IDENT})\s*)+)"
)
_REL = re.compile(
    rf"\[\s*({_IDENT})?\s*:\s*((?:{_IDENT})(?:\s*\|\s*:?\s*(?:{_IDENT}))*)"
)
_PROP = re.compile(rf"\b({_IDENT})\s*\.\s*({_IDENT})")
_INLINE = re.compile(
    rf"\(\s*({_IDENT})?\s*((?::\s*(?:{_IDENT})\s*)+)\{{([^}}]*)\}}"
)
_STRINGS = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")


def _clean(name):
    return name.strip().strip("`")


_CMP_BEFORE = re.compile(
    rf"({_IDENT})\s*\.\s*({_IDENT})\s*(?:=|<>)\s*$"
    rf"|({_IDENT})\s*\.\s*({_IDENT})\s+IN\s*\[[^\]]*$",
    re.IGNORECASE,
)
_INLINE_BEFORE = re.compile(
    rf"\(\s*(?:{_IDENT})?\s*((?::\s*(?:{_IDENT})\s*)+)\{{[^}}]*?({_IDENT})\s*:\s*$"
)
_LITERAL = re.compile(r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"")


def _var_map(q):
    var_to_names = {}
    for m in _NODE.finditer(q):
        if m.group(1):
            var_to_names.setdefault(_clean(m.group(1)), set()).update(
                _clean(x) for x in re.findall(_IDENT, m.group(2)))
    for m in _REL.finditer(q):
        if m.group(1):
            var_to_names.setdefault(_clean(m.group(1)), set()).update(
                _clean(x) for x in re.findall(_IDENT, m.group(2)))
    return var_to_names


def literal_spans(cypher):
    if not cypher:
        return []
    var_to_names = _var_map(_STRINGS.sub("''", cypher))
    out = []
    for m in _LITERAL.finditer(cypher):
        value = m.group(1) if m.group(1) is not None else m.group(2)
        before = _STRINGS.sub("''", cypher[: m.start()])
        c = _CMP_BEFORE.search(before)
        if c:
            var = _clean(c.group(1) or c.group(3))
            prop = _clean(c.group(2) or c.group(4))
            for owner in sorted(var_to_names.get(var, ())):
                out.append((m.start(), m.end(), f"{owner}.{prop}", value))
            continue
        i = _INLINE_BEFORE.search(before)
        if i:
            for owner in sorted({_clean(x) for x in re.findall(_IDENT, i.group(1))}):
                out.append((m.start(), m.end(), f"{owner}.{_clean(i.group(2))}", value))
    return out


def gold_literals(cypher):
    return {(prop, value) for _, _, prop, value in literal_spans(cypher)}


def gold_schema_set(cypher):
    if not cypher:
        return {"labels": set(), "rels": set(), "props": set()}

    q = _STRINGS.sub("''", cypher)

    labels, rels, props = set(), set(), set()
    var_to_names = {}

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
