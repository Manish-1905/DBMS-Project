#!/usr/bin/env python3
"""
Controlled distractors for the retrieval benchmark. (Roadmap Part 4.5)

For one question we know which schema elements are relevant (its gold
schema set). Everything else in the schema is a distractor. We rebuild the
schema so that a chosen share of its elements is irrelevant:

    level 0.40 -> 40% of the elements are distractors
    level 0.95 -> 95% of the elements are distractors

* If the real schema already has MORE distractors than asked for, irrelevant
  elements are hidden (props first, then relationships, then labels).
* If it has FEWER, real labels (with their properties) and relationships are
  borrowed from the OTHER MTQ graphs, so the distractors have realistic
  names. Some borrowed labels are linked to the real schema with a borrowed
  relationship type, so the topology step cannot simply ignore them.

Gold elements are never removed, and nothing borrowed reuses a name already
in the schema, so the gold set stays valid.

An "element" is a label, a relationship type or a property, the same units
that eval_retrieval.py scores.
"""

import json
import random


def schema_sets(schema):
    s = json.loads(schema) if isinstance(schema, str) else schema
    rels = s.get("relationships", [])
    labels = set(s.get("node_props", {})) | {r["start"] for r in rels} | {r["end"] for r in rels}
    rtypes = {r["type"] for r in rels} | set(s.get("rel_props", {}))
    props = {f"{L}.{p['property']}" for L, ps in s.get("node_props", {}).items() for p in ps}
    props |= {f"{T}.{p['property']}" for T, ps in s.get("rel_props", {}).items() for p in ps}
    return {"labels": labels, "rels": rtypes, "props": props}


def count_elements(schema):
    return sum(len(v) for v in schema_sets(schema).values())


def _gold_all(gold):
    return set(gold["labels"]) | set(gold["rels"]) | set(gold["props"])


def irrelevant_share(schema, gold):
    sets = schema_sets(schema)
    total = sum(len(v) for v in sets.values())
    relevant = sum(len(sets[k] & set(gold[k])) for k in sets)
    return (total - relevant) / total if total else 0.0


def build_pool(schemas):
    """All labels and relationships of a list of schemas, as borrowable units."""
    labels, rels = {}, []
    for sch in schemas:
        s = json.loads(sch) if isinstance(sch, str) else sch
        for L, ps in s.get("node_props", {}).items():
            labels.setdefault(L, ps)
        for r in s.get("relationships", []):
            rels.append((r, s.get("rel_props", {}).get(r["type"], [])))
    return {"labels": labels, "rels": rels}


def _copy(s):
    return json.loads(json.dumps(s))


def _hide(s, gold, target, rng):
    """Remove irrelevant elements until the irrelevant share is <= target."""
    g = {k: set(v) for k, v in gold.items()}

    def share():
        return irrelevant_share(s, g)

    # 1. irrelevant properties
    props = [(o, p["property"], kind) for kind in ("node_props", "rel_props")
             for o, ps in s.get(kind, {}).items() for p in ps
             if f"{o}.{p['property']}" not in g["props"]]
    rng.shuffle(props)
    for owner, name, kind in props:
        if share() <= target:
            return s
        s[kind][owner] = [p for p in s[kind][owner] if p["property"] != name]

    # 2. irrelevant relationship types
    rtypes = sorted({r["type"] for r in s.get("relationships", [])} - g["rels"])
    rng.shuffle(rtypes)
    for T in rtypes:
        if share() <= target:
            return s
        s["relationships"] = [r for r in s["relationships"] if r["type"] != T]
        s.get("rel_props", {}).pop(T, None)

    # 3. irrelevant labels (and any relationship touching them)
    # keep endpoints of gold relationships, so no gold relationship disappears
    gold_ends = {x for r in s["relationships"] if r["type"] in g["rels"]
                 for x in (r["start"], r["end"])}
    labs = sorted(schema_sets(s)["labels"] - g["labels"] - gold_ends)
    rng.shuffle(labs)
    for L in labs:
        if share() <= target:
            return s
        s.get("node_props", {}).pop(L, None)
        s["relationships"] = [r for r in s["relationships"] if L not in (r["start"], r["end"])]
    return s


def _add(s, gold, target, pool, rng, link_prob=0.3):
    """Borrow labels/relationships from other graphs until share >= target."""
    g = {k: set(v) for k, v in gold.items()}
    s.setdefault("node_props", {})
    s.setdefault("rel_props", {})
    s.setdefault("relationships", [])
    used = schema_sets(s)
    taken_names = used["labels"] | used["rels"]
    real_labels = sorted(used["labels"])

    cand_labels = [L for L in sorted(pool["labels"]) if L not in taken_names]
    rng.shuffle(cand_labels)
    rels_by_label = {}
    for r, rp in pool["rels"]:
        rels_by_label.setdefault(r["start"], []).append((r, rp))
        rels_by_label.setdefault(r["end"], []).append((r, rp))

    added = set()
    for L in cand_labels:
        if irrelevant_share(s, g) >= target:
            break
        s["node_props"][L] = pool["labels"][L]
        added.add(L)
        taken_names.add(L)
        # bring along relationships between borrowed labels
        for r, rp in rels_by_label.get(L, []):
            if r["start"] in added and r["end"] in added and r["type"] not in used["rels"]:
                if r not in s["relationships"]:
                    s["relationships"].append(dict(r))
                    if rp:
                        s["rel_props"].setdefault(r["type"], rp)
                taken_names.add(r["type"])
        # sometimes hook the distractor onto the real schema
        if real_labels and rng.random() < link_prob:
            borrowed = [(r, rp) for r, rp in rels_by_label.get(L, [])
                        if r["type"] not in used["rels"]]
            if borrowed:
                r, rp = rng.choice(borrowed)
                anchor = rng.choice(real_labels)
                link = ({"start": L, "type": r["type"], "end": anchor} if rng.random() < 0.5
                        else {"start": anchor, "type": r["type"], "end": L})
                if link not in s["relationships"]:
                    s["relationships"].append(link)
                    if rp:
                        s["rel_props"].setdefault(r["type"], rp)
                taken_names.add(r["type"])
    return s


def with_distractors(schema, gold, target, pool, seed=0):
    """
    Return (new_schema_dict, achieved_irrelevant_share).

    The achieved share can be below `target` when the pool runs out, so
    always report the achieved value.
    """
    rng = random.Random(seed)
    s = _copy(json.loads(schema) if isinstance(schema, str) else schema)
    s.setdefault("relationships", [])
    cur = irrelevant_share(s, gold)
    if cur > target:
        s = _hide(s, gold, target, rng)
    elif cur < target:
        s = _add(s, gold, target, pool, rng)
    return s, irrelevant_share(s, gold)
