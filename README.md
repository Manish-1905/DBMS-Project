# ConText2Cypher

Natural language → Cypher, evaluated by executing queries against MTQ graphs in Neo4j.

```
question (+ schema) ──► llama.cpp LLM ──► Cypher ──► Neo4j (read-only) ──► SQLite ──► evaluate.py
```

| File | What it does |
|---|---|
| `main.py` | Runs the benchmark: generates Cypher, executes LLM + gold query, saves everything to SQLite |
| `evaluate.py` | Reads the SQLite file and prints the metrics table |
| `split.py` | Makes a smaller parquet containing only one graph's questions |
| `schema_inspect.py` | **Part 3.** Extracts the schema, indexes, constraints and value statistics from the live Neo4j graph, with a `schema_version` hash |
| `gold_schema.py` | **Part 4.4.** Lists the labels / relationship types / properties a gold Cypher query uses |
| `schema_retriever.py` | **Part 4.** Picks only the part of the schema a question needs (BM25 + schema-graph shortest paths) |
| `retrievers.py` | **Part 4.2.** Ranked retrievers: BM25, semantic (embeddings), hybrid |
| `eval_retrieval.py` | **Part 4.4–4.5.** Benchmarks the retrievers on their own (Recall@k, MRR, context reduction, distractors); no LLM or Neo4j needed |
| `distractors.py` | **Part 4.5.** Builds schemas with a controlled share of distractor elements |
| `prepare_text2cypher.py` | Converts Neo4j Text2Cypher 2024 into the MTQ parquet format |
| `Benchmark/MTQ-Auto/test.parquet` | MTQ questions (`question`, `schema`, `cypher`, `data_source`, ...) |
| `Datasets/*.dump` | Neo4j database dumps, one per graph (not in git, see step 3) |

---

## 1. Install the tools (once)

| Tool | Why | How |
|---|---|---|
| Python 3.12 + **uv** | runs the scripts | `pip install uv` (or see docs.astral.sh/uv) |
| **Neo4j 5.x** | the graph database | Docker (`neo4j:5`) **or** Neo4j Desktop |
| **llama.cpp** | serves the LLM locally | prebuilt release from github.com/ggml-org/llama.cpp, or `brew install llama.cpp` |

## 2. Get the code and Python packages

```bash
git clone https://github.com/Manish-1905/DBMS-Project.git
cd DBMS-Project
uv sync            # creates .venv and installs neo4j, pandas, pyarrow, requests, duckdb
```

## 3. Load a graph into Neo4j

> The Neo4j dumps are **not** stored in git (too large). Get them from the
> Mind the Query release (or a teammate) and put them in `Datasets/`
> (ignored by git), e.g. `Datasets/entity-resolution-50.dump`.

**Option A: Docker**

```bash
mkdir -p neo4j/data neo4j/dumps
cp /path/to/entity-resolution-50.dump neo4j/dumps/neo4j.dump   # file name must be <dbname>.dump

# load the dump into the "neo4j" database
docker run --rm -v "$PWD/neo4j/data:/data" -v "$PWD/neo4j/dumps:/dumps" \
  neo4j:5 neo4j-admin database load neo4j --from-path=/dumps --overwrite-destination=true

# start the server
docker run -d --name ctx2cypher -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/choose_a_password \
  -v "$PWD/neo4j/data:/data" neo4j:5
```

**Option B: Neo4j Desktop.** Create a DBMS → ⋯ → *Import* → pick the `.dump` file → start it.

Check: open http://localhost:7474 and run `MATCH (n) RETURN labels(n), count(*)`. You should see nodes.

If loading fails with a version error, use the Neo4j version the MTQ dumps were made with.

**Switching graphs:** Neo4j Community has one user database, so stop the server, load another
dump with the same command, and start again. With Desktop/Enterprise you can keep several
databases and choose one with `--neo4j-database <name>`.

## 4. Start the LLM server

```bash
llama-server -hf Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:Q4_K_M --port 8080 -c 8192
```

`-c 8192` gives room for long schemas in the prompt. Check it's up: http://127.0.0.1:8080

## 5. Run the benchmark

```bash
export NEO4J_PASSWORD=choose_a_password      # Windows PowerShell: $env:NEO4J_PASSWORD="..."

uv run split.py --list                       # see which graphs (data_source) exist

# System A: LLM only (no schema)
uv run main.py --parquet Benchmark/MTQ-Auto/test.parquet --data-source er \
  --mode no_schema --model Qwen2.5-Coder-1.5B --sqlite results.sqlite

# System B: LLM + full schema
uv run main.py --parquet Benchmark/MTQ-Auto/test.parquet --data-source er \
  --mode full_schema --model Qwen2.5-Coder-1.5B --sqlite results.sqlite

# System 2: LLM + retrieved schema (only the relevant labels/relationships)
uv run main.py --parquet Benchmark/MTQ-Auto/test.parquet --data-source er \
  --mode retrieved_schema --model Qwen2.5-Coder-1.5B --sqlite results.sqlite
```

Each row also stores the MTQ `query_type` and, for `retrieved_schema`, which
schema elements were retrieved (`retrieved_context` column).

Useful flags: `--start 0 --end 5` (quick test on 5 questions), `--query-timeout 30`,
`--run-id my_label`.

`--data-source` must match the graph currently loaded in Neo4j, or the gold queries will fail.
`--model` is just the label saved in the results; llama-server serves whatever model it loaded.

## 6. Get the metrics

```bash
uv run evaluate.py --sqlite results.sqlite --rows-csv per_question.csv

# also break results down by query type (simple/complex retrieval/aggregation, ...)
uv run evaluate.py --sqlite results.sqlite --by-type
```

Prints one row per run with: runnable, non-null, result correct, exact match, execution times,
generation time. `per_question.csv` has every question with its error message, for error analysis.

---

## 7. Schema extraction (Part 3)

With a graph loaded in Neo4j:

```bash
uv run schema_inspect.py --out schemas/er.json --compare-mtq --data-source er
```

Prints `schema_version` (changes only when the *structure* changes) and how much
of MTQ's schema was recovered (label / relationship / direction / property / type
recall). The JSON also has `value_stats` (distinct counts, most common values,
min/max), which entity/value grounding (Part 5) will use.

## 8. Schema retrieval (Part 4)

Part 4 decides which part of the schema the LLM sees. It is benchmarked
**on its own first** (no LLM or Neo4j needed, it runs on the parquet files).

| Plan item | File |
|---|---|
| 4.2 Retriever 1, lexical (BM25) | `schema_retriever.py` (default), `retrievers.py` (`bm25`) |
| 4.2 Retriever 2, semantic (embeddings) | `retrievers.py` (`semantic`, and `hybrid` = BM25 + semantic fused) |
| 4.2 Retriever 3, topology (smallest connected subgraph) | `schema_retriever.py`, applied after any ranker |
| 4.4 Gold schema sets | `gold_schema.py` |
| 4.4 Recall@1/3/5/10, MRR, recall of the pruned schema | `eval_retrieval.py` |
| 4.5 Controlled distractors | `distractors.py`, `eval_retrieval.py --distractors` |
| Text2Cypher 2024 dataset | `prepare_text2cypher.py` |

The semantic retriever needs one extra install (PyTorch + sentence-transformers,
about 1 GB; the embedding model, all-MiniLM-L6-v2, downloads on first use):

```bash
uv sync --extra semantic
```

### Run the benchmarks

```bash
# BM25 on all 11 MTQ graphs
uv run eval_retrieval.py --csv retrieval_rows.csv

# compare the three retrievers
uv run eval_retrieval.py --retriever bm25 semantic hybrid

# 4.5: 40 / 60 / 80 / 95 % of the schema is distractors
uv run eval_retrieval.py --retriever bm25 semantic hybrid --distractors 0.4 0.6 0.8 0.95

# inspect one question
uv run schema_retriever.py --data-source er --row 3 --retriever semantic
```

Two tables per retriever:

* **Pruned schema** (what the LLM would see): recall per element kind,
  `full_gold_recall` (every label, relationship and property the gold query
  needs was kept) and `context_reduction` (how much smaller the schema text is).
* **Ranking:** Recall@k and MRR over the ranked list of all schema elements
  (labels, relationship types, properties). A gold set usually has 4–6
  elements, so Recall@1 can be at most 1/|gold|; look at R@5 / R@10 and MRR.

Useful flags: `--k-labels/--k-rels` (BM25 selection), `--k-elements`
(semantic/hybrid: how many top-ranked elements seed the topology step),
`--sample 1000` (fixed random sample, faster), `--data-source er`.

**Distractors (4.5).** For each question, `distractors.py` rebuilds its
schema so a chosen share of the elements is irrelevant to the gold query. If
the real schema has more distractors than asked for, irrelevant elements are
hidden; if fewer, real labels and relationships are borrowed from the *other*
MTQ graphs (some linked to the real schema), so distractors look realistic.
Gold elements are never removed. The table reports the share actually
achieved.

### BM25 results on all 8,284 MTQ test questions

| distractors | avg schema elements | all needed elements kept | context reduction | R@5 | R@10 | MRR |
|---|---|---|---|---|---|---|
| real schema (89% irrelevant on average) | 81 | 95.4% | 41.0% | 62.3% | 80.5% | 0.944 |
| 40% | 8 | 96.9% | 14.9% | 84.7% | 97.4% | 0.994 |
| 60% | 13 | 96.3% | 31.8% | 80.6% | 94.1% | 0.984 |
| 80% | 27 | 95.8% | 52.9% | 72.1% | 88.4% | 0.964 |
| 95% | 126 | 95.4% | 70.1% | 63.0% | 79.1% | 0.946 |

So recall stays around 95% while the share of distractors goes from 40% to
95%, and the context saving grows with it.

### Text2Cypher 2024

```bash
uv run prepare_text2cypher.py          # downloads the test split from Hugging Face
uv run eval_retrieval.py --parquet Benchmark/Text2Cypher-2024/test.parquet --retriever bm25 semantic hybrid
```

The script turns the dataset's text schemas into MTQ's JSON format. Rows are
grouped by demo database (`graph=`) and by the dataset's source column (`type=`).

### System 2 with the LLM

`--retriever` also works in `main.py` (needs Neo4j and llama-server, as in step 5):

```bash
uv run main.py --parquet Benchmark/MTQ-Auto/test-split.parquet --mode retrieved_schema --retriever semantic --model Qwen2.5-Coder-1.5B --sqlite results.sqlite
```

The default run ID includes the retriever, so `evaluate.py` shows each one as its own row.

## Safety notes

* LLM queries run in a **read-only session** with a timeout, so a generated `DELETE` cannot wipe the graph.
* The password comes from `NEO4J_PASSWORD`; don't hard-code it.

## Adding a new system (Parts 3–8 of the roadmap)

1. Add a name to `MODES` in `main.py` (e.g. `grounded`). See how
   `retrieved_schema` is handled in `run_benchmark()` for an example.
2. Add a branch in `build_user_prompt()` that builds that system's prompt.
3. If you change the prompt text, bump `PROMPT_VERSION`.
4. Run with `--mode <name>`; `evaluate.py` compares it with the others automatically.
