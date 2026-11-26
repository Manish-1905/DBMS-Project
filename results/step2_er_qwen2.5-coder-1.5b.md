# Step 2 results: ER graph, Qwen2.5-Coder-1.5B (Q4_K_M)

Run on 26 Nov 2026, all 856 MTQ test questions for the `er` graph
(`Benchmark/MTQ-Auto/test-split.parquet`), temperature 0, prompt `v1`.
Gold queries: 100% executed (`gold_ok`), so every failure below is the model's.

Reproduce:

```bash
for m in no_schema full_schema retrieved_schema; do
  uv run main.py --parquet Benchmark/MTQ-Auto/test-split.parquet --neo4j-database $DB \
    --mode $m --model Qwen2.5-Coder-1.5B --sqlite results.sqlite
done
uv run evaluate.py --sqlite results.sqlite --by-type --rows-csv per_question.csv
```

## Overall

| System | Mode | Runnable | Non-null | **Result correct** | Exact match | Avg gen (s) |
|---|---|---|---|---|---|---|
| S0 | LLM only (`no_schema`) | 87% | 41% | **33%** | 0% | 3.00 |
| S1 | LLM + full schema (`full_schema`) | 86% | 61% | **52%** | 1% | 2.68 |
| S2 | LLM + retrieved schema (`retrieved_schema`) | 85% | 62% | **53%** | 1% | 4.20 |

## Result correctness by query type

| Query type | n | S0 no schema | S1 full schema | S2 retrieved schema |
|---|---|---|---|---|
| simple retrieval | 224 | 62% | 79% | 73% |
| simple aggregation | 89 | 30% | 81% | 84% |
| evaluation query | 225 | 49% | 67% | 73% |
| complex retrieval | 165 | 1% | 16% | 15% |
| complex aggregation | 153 | 1% | 10% | 16% |

Runnable rate by type (S0 / S1 / S2): complex aggregation 61% / 54% / 61%,
complex retrieval 86% / 81% / 82%, all simple types 89–97%.

## Findings

1. **Schema context is the single biggest win: +19 points result correctness**
   (33% → 52%). Non-null results rise 41% → 61%.
2. **"Runnable" barely changes (85–87%) across systems.** Without a schema the
   model writes valid Cypher against labels/relationships that do not exist, so
   it runs and returns nothing or the wrong thing. Execution success is not
   evidence of correctness; result correctness is the metric to report.
3. **Retrieved schema matches full schema (53% vs 52%) with less context**
   (about 30% fewer schema characters on ER; see `eval_retrieval.py`). ER has
   only 4 labels, so there is little to prune; larger graphs (bloom50, pole,
   gdsc, healthcare: 60–78% context reduction) are where S2 should pull ahead.
4. **Complex queries are the bottleneck (10–16% correct vs 67–84% for simple
   ones).** Complex aggregations also fail to *run* ~40–46% of the time. This is
   generation quality, not missing context: the motivation for query planning /
   IR (Part 6) and validation + repair (Part 7).
5. **Exact match (0–1%) is useless as a metric**, as expected.

## Caveats

* **Latency numbers are not comparable across runs.** The *gold* queries are
  identical in every run, yet their average time ranges 4.9–22.6 ms
  (`no_schema` ran first on a cold database cache). Don't claim speed-ups.
* S2's higher average generation time (4.20 s) comes from the simple-retrieval
  bucket (7.93 s) and is likely a few long outputs or machine load, not the
  retriever (which takes milliseconds). Check medians in `per_question.csv`.
* **These replace the midsem baseline numbers** (56% runnable, ~9% correct).
  The old script always included the schema in the prompt and was run with a
  different setup, so those numbers were not an "LLM only" baseline.
* One graph, one small model. Repeat on at least one larger-schema graph and
  ideally one bigger model before drawing final conclusions.

## Part 3 check (schema extraction, same graph)

`schema_inspect.py` recovered **100%** of MTQ's labels, relationship types,
relationship directions and properties (4 labels, 3 relationship triples,
9 indexes, 4 constraints; `schema_version` `46797f943d798546`). Type recall
showed 0% because of a type-name normalisation bug, fixed in the following
commit; re-run to confirm.
