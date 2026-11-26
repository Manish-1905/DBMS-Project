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

## Caveats

* Distractors get rule-based aliases (plurals, split names) but not LLM aliases;
  real elements have LLM aliases. `bm25` (no semantics) is the fully fair row.
* Embedding (`embed`) and `hybrid` retrieval are implemented but were not measured
  here (the model could not be downloaded in the build environment). Run:
  `uv run eval_distractors.py --methods bm25+sem embed+sem hybrid+sem`.
* This measures retrieval only. Whether the LLM actually does better with the
  retrieved schema under noise is the next experiment:
  `main.py --distractor-share 0.8` with `full_schema` vs `retrieved_schema`.
