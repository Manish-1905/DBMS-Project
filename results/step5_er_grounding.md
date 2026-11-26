# Step 5 results: entity and value grounding on ER

Model: Qwen2.5-Coder-1.5B (Q4_K_M), temperature 0. System 3 (`--mode grounded`) =
retrieved schema (BM25 + semantic layer) + the graph values the question refers to,
listed in the prompt.

## 5.5 Graph-state experiment (headline)

The ER graph was copied into a second database (`erstateb`) and some stored values
were recoded (state B): `User.gender` Female/Male → F/M, `User.country` US →
United States, `Genre.name` → upper case (Drama → DRAMA). Schema, nodes and
relationships are unchanged; the original database was not modified. Gold queries
were rewritten to the new values; **question text is unchanged** ("female users",
"'US'", "'Drama'").

Evaluated on the **139 questions whose gold answer depends on a recoded value**
(of 856 ER questions).

```bash
uv run graph_state.py create --source $DB --target erstateb --state B
uv run value_index.py --neo4j-database erstateb --out values/er_stateB.json
uv run graph_state.py rewrite-gold --state B --affected-only --out Benchmark/MTQ-Auto/test-split-stateB.parquet
uv run main.py ... --neo4j-database erstateb --mode retrieved_schema --sqlite state.sqlite
uv run main.py ... --neo4j-database erstateb --mode grounded --values-file values/er_stateB.json --sqlite state.sqlite
uv run evaluate.py --sqlite state.sqlite --compare retrieved_schema grounded
```

| Same 139 questions | Graph | Correct | Ran, no rows (`empty`) | Returned data |
|---|---|---|---|---|
| No schema (S0) | original | 5.8% | | |
| Retrieved schema (S2) | original | **28.1%** | | |
| Full schema (S1) | original | 28.8% | | |
| Retrieved schema (S2), no grounding | **recoded** | **0.0%** | 73% | 14% |
| **Retrieved schema + grounding (S3)** | **recoded** | **27.3%** | 54% | 33% |

Paired comparison on the recoded graph (exact McNemar): grounding answered
**38** questions the static system missed and lost **0** (p < 0.001).

Grounding alone (no LLM; `eval_grounding.py --values values/er_stateB.json`):

| | Result |
|---|---|
| New stored value written verbatim in the question | 6.5% |
| Gold value found and ranked first for its property (R@1) | **100%** |
| All gold values of the question found | **100%** |
| A wrong value proposed instead of the right one | 0% |
| Extra (unused) values proposed per question | 0.14 |

By property: Genre.name 92 questions, User.country 32, User.gender 10 — 100% each.

**Conclusion.** When the stored values changed, a system that takes values from
the question text failed on every affected question (28.1% → 0.0%; 73% of its
queries ran and returned nothing). Reading the values from the current graph
restored accuracy to 27.3%, about 97% of the original, with no retraining and no
change to the schema or the questions. Grounding found every new value, so the
remaining errors are the model's query-logic errors on these (mostly complex)
questions, the same ones it makes on the original graph.

Caveats: the "original graph" rows are from the Step 2 runs (retriever v1, no
semantic layer); the recoded-graph rows use retriever v2 + semantic layer. The
grounded system's `empty` rate stays high (54%) because many of these questions
are complex queries where the generated logic is wrong even with the right values.

## 5.4 Value-grounding benchmark

`Benchmark/MTQ-Auto/value_variants_er.parquet` (frozen): 394 ER questions whose
gold query filters on a title, genre, country, state, gender or name, rewritten as
users type them: lowercase (351), unquoted (362), typo (304), partial title (75),
alias such as USA / OH / women (55), plus the originals (394).

Results on the real catalog: run `uv run eval_grounding.py --values values/er.json`
and add the table here.
