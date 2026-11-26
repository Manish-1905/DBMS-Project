# Step 3 results: Graph Context Compiler on ER

Graph: MTQ `er` (entity-resolution-50) loaded in Neo4j. Retrieval benchmark:
all 856 ER test questions, `k_labels=3, k_rels=3`, retriever v2.

## 3.1 / 3.5 Schema extraction vs the MTQ schema

`uv run schema_inspect.py --neo4j-database $DB --out schemas/er.json --compare-mtq --data-source er`

| Check | Result |
|---|---|
| Labels / relationship types / directions / properties recall | 100% / 100% / 100% / 100% |
| Property types | match (after the type-name fix) |
| `schema_version` of the live graph vs of MTQ's schema | identical: `14b1692b201450b3` |
| Extracted extras | 9 indexes, 4 constraints, value statistics per property |

The live extraction reproduces MTQ's schema exactly.

## 3.2 Semantic layer

`semantics/er.json` = rules + local LLM (Qwen2.5-Coder-1.5B, one call per element,
with sample values) + hand-written `semantics/er.overrides.json`.
The overrides removed one harmful LLM alias (`Genre` → "type", which matches
"what type of…" questions) and added viewer/customer for `User` and clearer
aliases for `USES`.

Effect on schema retrieval (gold schema set = labels/relationships/properties
used by the gold Cypher):

| Retriever | Label recall | Rel recall | Prop recall | **All needed elements kept** | Context reduction |
|---|---|---|---|---|---|
| v1 (Step 2 runs) | 98.8% | 98.2% | 98.8% | 96.3% | 29.9% |
| v2 (fixed plural/verb matching) | 99.9% | 99.8% | 100.0% | **99.5%** | 29.1% |
| v2 + semantic layer | 100.0% | 100.0% | 100.0% | **100.0%** | 25.8% |

By query type with the semantic layer: 100% kept for every type; context
reduction 10.8% (complex retrieval) to 39.2% (simple aggregation).

Trade-off: aliases occasionally pull in an extra label, so the context is
3.3 points larger, but no question loses a needed schema element.

## 3.4 Schema graph

`uv run schema_graph.py --data-source er --schema-file schemas/er.json`

Loaded into the separate database `schemagraph`: 1 `Schema`, 4 `NodeType`,
3 `RelationshipType`, 14 `Property` nodes (with descriptions, aliases and value
statistics). Example query: shortest path `Genre → User` =
`Genre -HAS- Movie -WATCHED- User`.

## Exit criterion

`uv run pytest -q`: `schema_version` is identical for the same schema regardless
of ordering, unchanged when only values/indexes change, and changes when a label,
property, relationship, direction or property type changes.

## Caveat

ER is small (4 labels, 3 relationships), so retrieval has little to prune and
little to get wrong. The distractor experiment (roadmap 4.5) is the planned test
of retrieval under a large, noisy schema.
