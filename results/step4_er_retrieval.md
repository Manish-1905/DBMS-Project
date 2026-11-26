# Step 4 results: schema retrieval under controlled distractors (ER)

All 856 ER test questions. Retrieval only (no LLM, no Neo4j); `k_labels=3, k_rels=3`.
Distractors: fake labels/relationships added to the schema text so that `share`
of all schema elements are fake (ER has 7 real elements: 4 labels + 3 rel types).
Half of the fake labels are "hard" (share words with real ones: `UserProfile`,
`MovieReview`, `IpAddressGroup`, ...). Seed 0.

```bash
uv run eval_distractors.py --methods bm25 bm25+sem
```

| Noise | Fake elements | Method | **Gold kept** | Label R@3 | Label MRR | Rel R@3 | Rel MRR | Leak | Schema chars sent | % of full noisy schema |
|---|---|---|---|---|---|---|---|---|---|---|
| 0% | 0 | bm25 | 99.5% | 98.7% | 0.999 | 77.1% | 0.856 | 0.0% | 654 | 70.9% |
| 0% | 0 | bm25+sem | **100.0%** | 98.8% | 0.998 | 99.6% | 0.985 | 0.0% | 684 | 74.2% |
| 40% | 5 | bm25 | 97.0% | 97.6% | 0.998 | 77.1% | 0.855 | 20.7% | 745 | 43.0% |
| 40% | 5 | bm25+sem | **100.0%** | 88.5% | 0.983 | 99.6% | 0.985 | 20.4% | 899 | 51.9% |
| 60% | 10 | bm25 | 92.6% | 94.1% | 0.997 | 77.1% | 0.856 | 30.9% | 985 | 43.1% |
| 60% | 10 | bm25+sem | **100.0%** | 88.3% | 0.976 | 99.6% | 0.985 | 26.5% | 1088 | 47.7% |
| 80% | 28 | bm25 | 94.0% | 84.3% | 0.973 | 77.1% | 0.853 | 53.2% | 1027 | 21.9% |
| 80% | 28 | bm25+sem | **100.0%** | 83.5% | 0.915 | 99.3% | 0.983 | 46.6% | 1216 | 25.9% |
| 95% | 133 | bm25 | 94.3% | 76.1% | 0.918 | 77.1% | 0.848 | 54.0% | 1393 | 8.1% |
| 95% | 133 | bm25+sem | **99.3%** | 69.8% | 0.879 | 98.7% | 0.978 | 50.1% | 1485 | 8.6% |

Columns: *gold kept* = % of questions where every label/relationship/property the
gold query uses is in the retrieved schema; *R@3 / MRR* = ranking quality of the
gold labels / relationship types among all (real + fake) ones; *leak* = % of the
retrieved labels/relationships that are fake.

## Findings

1. **Retrieval survives heavy noise.** With 95% of the schema fake (140 elements,
   17,192 chars), BM25 + semantic layer still keeps everything the gold query
   needs for **99.3%** of questions while sending only **8.6%** of the schema
   (1,485 chars, about the size of the clean ER schema).
2. **The semantic layer matters most for relationships.** Plain BM25 ranks the
   right relationship type in the top 3 only 77% of the time (relationship names
   like `HAS` / `USES` rarely appear in questions); the LLM aliases raise this to
   ~99% at every noise level.
3. **Leak grows with noise (up to ~50%).** The retrieved schema keeps a few hard
   distractors (`UserClip`, `GenreNote` ...), so the LLM still sees *some* noise,
   but ~6 fake elements instead of 133.
4. **Ranking degrades gracefully**: label MRR 0.999 → 0.879 from 0% to 95% noise.

## Retrieval method comparison (BM25 vs embeddings vs hybrid)

`uv run eval_distractors.py --methods bm25+sem embed+sem hybrid+sem`
(embedding model `BAAI/bge-small-en-v1.5`, `embed_margin=0.1`; all with the semantic layer)

| Noise | Method | **Gold kept** | Label R@3 | Label MRR | Rel R@3 | Rel MRR | Leak | Chars sent | % of full |
|---|---|---|---|---|---|---|---|---|---|
| 0% | bm25+sem | 100.0% | 98.8% | 0.998 | 99.6% | 0.985 | 0.0% | 684 | 74.2% |
| 0% | embed+sem | 100.0% | 99.4% | 1.000 | 100.0% | 1.000 | 0.0% | 759 | 82.3% |
| 0% | hybrid+sem | 100.0% | 99.5% | 1.000 | 100.0% | 0.989 | 0.0% | 769 | 83.4% |
| 40% | bm25+sem | 100.0% | 88.5% | 0.983 | 99.6% | 0.985 | 20.4% | 899 | 51.9% |
| 40% | embed+sem | 99.4% | 87.0% | 0.915 | 99.7% | 0.999 | 32.6% | 1126 | 65.0% |
| 40% | hybrid+sem | 99.8% | 93.5% | 0.975 | 99.9% | 0.989 | 32.0% | 1159 | 66.9% |
| 60% | bm25+sem | 100.0% | 88.3% | 0.976 | 99.6% | 0.985 | 26.5% | 1088 | 47.7% |
| 60% | embed+sem | 99.3% | 86.3% | 0.897 | 99.5% | 0.943 | 42.9% | 1172 | 51.3% |
| 60% | hybrid+sem | 100.0% | 92.4% | 0.941 | 100.0% | 0.996 | 39.4% | 1243 | 54.5% |
| 80% | bm25+sem | **100.0%** | 83.5% | 0.915 | 99.3% | 0.983 | **46.6%** | **1216** | 25.9% |
| 80% | embed+sem | 92.2% | 64.8% | 0.708 | 89.7% | 0.899 | 69.0% | 1883 | 40.1% |
| 80% | hybrid+sem | 99.5% | 73.4% | 0.828 | 99.6% | 0.994 | 62.4% | 1835 | 39.1% |
| 95% | bm25+sem | **99.3%** | 69.8% | 0.879 | 98.7% | 0.978 | **50.1%** | **1485** | 8.6% |
| 95% | embed+sem | 87.4% | 44.1% | 0.544 | 74.9% | 0.711 | 76.3% | 2612 | 15.2% |
| 95% | hybrid+sem | 98.6% | 51.1% | 0.657 | 98.9% | 0.968 | 70.2% | 2403 | 14.0% |

**Lexical retrieval (BM25 + semantic aliases) wins under noise.** On the clean
schema all three are equal (100% kept). As noise grows, embeddings alone fall to
87.4% kept with 76% leak at 95% noise: the hard distractors (`UserProfile`,
`MovieReview` ...) mean almost the same thing as the real labels, so a semantic
model cannot tell them apart, while exact name matching can. Hybrid recovers
gold recall (98.6%) but keeps the embedding leak and sends ~60% more text than
BM25. Default stays `bm25` + semantic layer.

## LLM under noise (first run, unrepresentative sample)

Qwen2.5-Coder-1.5B, 200 questions per run, same 200 questions in every run.

| Noise | Full schema: correct | Retrieved: correct | Full: non-null | Retrieved: non-null | Full: runnable | Retrieved: runnable |
|---|---|---|---|---|---|---|
| 0% | 11% | **15%** | 38% | **42%** | 60% | 61% |
| 80% | 5% | **12%** | 25% | **34%** | 66% | 61% |
| 95% | 7% | **8%** | 24% | **33%** | 68% | 64% |

**Caveat, important:** these were the *first* 200 rows (`--end 200`), and the
parquet is sorted by query type, so the sample is 153 complex aggregations +
47 complex retrievals and no simple questions. That explains the low absolute
accuracy (Step 2 on all 856: ~52%). The direction is consistent with the
hypothesis (retrieved ≥ full at every noise level; noise costs the full schema
13–14 points of non-null answers vs 8–9 for retrieved), but at ~10% accuracy on
200 questions the gaps are a handful of questions and may be chance.
Re-run with `--sample 300` (stratified random) and check significance with
`evaluate.py --compare`.

## Caveats

* Distractors get rule-based aliases (plurals, split names) but not LLM aliases;
  real elements have LLM aliases. `bm25` (no semantics) is the fully fair row.
