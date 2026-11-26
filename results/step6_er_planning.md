# Step 6 results: query planning (IR → check → compiled Cypher) on ER

Model: Qwen2.5-Coder-1.5B (Q4_K_M), temperature 0. Stratified random sample of
300 ER questions (`--sample 300`, seed 0: same query-type mix as all 856).

* **S3 `grounded`**: retrieved schema + grounded values → LLM writes Cypher.
* **S4 `ir`**: same input → LLM writes a JSON query plan → checked against the
  schema and value catalog (certain mistakes auto-fixed, others sent back once
  with the reasons) → compiled to parameterised Cypher by code. If the plan is
  unsupported or still invalid, it falls back to S3.

```bash
for m in grounded ir; do uv run main.py --parquet Benchmark/MTQ-Auto/test-split.parquet \
  --neo4j-database $DB --mode $m --sample 300 --sqlite plan.sqlite; done
uv run evaluate.py --sqlite plan.sqlite --by-feature --compare grounded ir
```

## Overall

| System | Runnable | Ran, no rows | Returned data | **Correct** | Gen. time |
|---|---|---|---|---|---|
| S3 grounded | 87% | 14% | 65% | 56.3% | 2.3 s |
| **S4 plan (IR)** | 90% | **7%** | **75%** | **61.0%** | 6.2 s |

Paired (exact McNemar): both correct 156, only S3 13, **only S4 27**, neither 104;
**p = 0.038**.

## By gold-query feature (result correctness)

| Feature | n | S3 | S4 |
|---|---|---|---|
| 0 hops | 104 | 96% | 91% |
| 1 hop | 137 | 50% | 53% |
| **2 hops** | 51 | 2% | **29%** |
| **3+ hops** | 8 | 0% | **12%** |
| no filter | 56 | 43% | 32% |
| filter | 244 | 59% | **68%** |
| **no aggregation** | 201 | 68% | **79%** |
| aggregation | 99 | 33% | 25% |
| no sorting / limit | 288 | 58% | 63% |
| sorting / limit | 12 | 8% | 8% |
| single stage | 244 | 68% | **74%** |
| multi-stage (`WITH`) | 56 | 4% | 5% |

## What happened to the plans (S4)

| Plan outcome | n | Correct | Returned data |
|---|---|---|---|
| accepted first time (`ok`) | 94 (31%) | **77%** | 95% |
| accepted after one retry (`ok_after_retry`) | 20 (7%) | 5% | 60% |
| still invalid → fell back to S3 (`invalid`) | 186 (62%) | 59% | 67% |

## Findings

1. **Planning significantly improves accuracy** (56.3% → 61.0%, p = 0.038) and
   halves the empty-result rate (14% → 7%).
2. **The gain is in path structure.** Multi-hop questions go from almost never
   correct to 29% (2 hops) and 12% (3+ hops): the model only lists the path, and
   the checker corrects relationship directions and names before execution.
3. **Planning hurts aggregations** (33% → 25%) and single-label questions
   slightly (96% → 91%): the 1.5B model fills in aggregation fields less
   accurately than it writes them directly in Cypher.
4. **A plan that is valid at the first attempt is right 77% of the time**, the
   highest accuracy of any route in the project, but only 31% of plans are.
5. **Repairing plans with feedback does not work for this model**: plans
   accepted after a retry were correct 5% of the time (valid but meaningless).
   Falling back directly to S3 instead is likely better (`--ir-retries 0`;
   not yet measured).
6. **Cost**: about 2.7× the generation time (plan + sometimes retry + fallback).

## Caveats

* One model (1.5B) and one graph; a larger model would likely produce more
  valid plans.
* 62% of plans fail validation; their error types have not been analysed yet,
  so the plan format or prompt may be improvable.
* Few-shot example retrieval (roadmap 6.5) was not implemented: with only the
  ER test set, examples would leak test questions.
