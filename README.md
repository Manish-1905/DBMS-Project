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
| `semantic_layer.py` | **Part 3.2.** Descriptions + aliases for every label / relationship / property (rules, local LLM, hand-written overrides) → `semantics/<graph>.json` |
| `schema_graph.py` | **Part 3.4.** Stores the schema itself as a graph (`NodeType`, `RelationshipType`, `Property`) in a separate Neo4j database |
| `tests/` | `uv run pytest`: schema versioning, type names, gold-schema parser, retriever, semantic layer |
| `eval_retrieval.py` | **Part 4.4.** Benchmarks the retriever on its own (recall, Recall@k, MRR, context reduction); no LLM or Neo4j needed |
| `embedder.py` | **Part 4.2.** Embedding model for the `embed` / `hybrid` retrievers (fastembed, no PyTorch) |
| `distractors.py` | **Part 4.5.** Adds fake-but-plausible labels/relationships to a schema at a chosen noise level |
| `eval_distractors.py` | **Part 4.5.** Retrieval quality vs noise level, per retrieval method |
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

### Semantic layer (Part 3.2)

Aliases let the retriever match words that aren't in the schema ("films" → `Movie`).
Needs llama-server running (same command as step 4):

```bash
uv run schema_inspect.py --neo4j-database $DB --out schemas/er.json      # adds sample values
uv run semantic_layer.py --data-source er --schema-file schemas/er.json --llm
uv run semantic_layer.py --all --llm                                     # all 11 graphs, from the parquet
```

Review `semantics/er.json`. To fix it, edit `semantics/<graph>.overrides.json`
and re-run: per element, `"aliases"` replaces the list (use it to drop bad ones),
`"add_aliases"` adds to it, `"description"` replaces it. See `semantics/er.overrides.json`.
Commit the `semantics/` folder: it is part of the benchmark setup.
`main.py --mode retrieved_schema` uses `semantics/<graph>.json` automatically when it exists.

### Schema graph (Part 3.4)

```bash
uv run schema_graph.py --data-source er --schema-file schemas/er.json
```

Writes to a separate database `schemagraph` (created automatically in Neo4j Desktop),
never to the benchmark database. Explore it in Neo4j Browser with `:use schemagraph`.

### Tests

```bash
uv run pytest -q
```

## 8. Schema retrieval benchmark (Part 4)

Runs offline on the parquet; no LLM or Neo4j needed:

```bash
uv run eval_retrieval.py                          # all 11 graphs
uv run eval_retrieval.py --data-source er --csv retrieval_rows.csv
uv run schema_retriever.py --data-source er --row 3   # inspect one question
```

Results on all 8,284 MTQ test questions (k_labels=3, k_rels=3):

| Retriever | label recall | rel recall | prop recall | all needed elements kept | context reduction |
|---|---|---|---|---|---|
| v1 | 98.5% | 97.9% | 98.8% | 95.3% | 41.0% |
| v2 (fixed plural/verb matching) | 99.1% | 98.2% | 99.3% | 96.3% | 39.3% |

Compare with the semantic layer: `uv run eval_retrieval.py --semantics-dir semantics`.
The table also has ranked metrics (label/relationship Recall@k and MRR).

### Retrieval methods (Part 4.2)

`--method bm25` (keywords, default), `embed` (embedding similarity with
`BAAI/bge-small-en-v1.5` via fastembed; ~130 MB downloaded on first use) or
`hybrid` (both, fused by reciprocal rank). `--embed-margin` (default 0.1) keeps
only elements whose similarity is within that of the best match.

### Distractor experiment (Part 4.5)

Pads the schema with fake labels/relationships (only the schema text; Neo4j is
untouched) at 40/60/80/95% noise and measures retrieval:

```bash
uv run distractors.py --share 0.8                  # see what gets added
uv run eval_distractors.py --methods bm25 bm25+sem embed+sem hybrid+sem
```

With the LLM (full noisy schema vs retrieved schema), e.g. 300 questions at 80% noise.
Use `--sample N`, not `--end N`: the parquet is sorted by query type, so the first
N rows are all complex questions. `--sample` keeps the query-type mix and always
picks the same questions for the same N and `--sample-seed`.

```bash
uv run main.py --parquet Benchmark/MTQ-Auto/test-split.parquet --neo4j-database $DB \
  --mode full_schema --distractor-share 0.8 --sample 300 --sqlite noise.sqlite
uv run main.py --parquet Benchmark/MTQ-Auto/test-split.parquet --neo4j-database $DB \
  --mode retrieved_schema --distractor-share 0.8 --sample 300 --sqlite noise.sqlite
uv run evaluate.py --sqlite noise.sqlite --compare full_schema_noise80 retrieved_schema_noise80
```

`--compare` pairs the two runs question by question and gives a McNemar p-value
(is the difference more than chance?).

At 95% noise the full schema is ~17k characters: start llama-server with `-c 16384`.
Results: `results/step4_er_retrieval.md`.

Per-graph and per-query-type rows are printed by the script. The `missing`
column in the CSV shows what the retriever dropped; use it to improve it.

## Safety notes

* LLM queries run in a **read-only session** with a timeout, so a generated `DELETE` cannot wipe the graph.
* The password comes from `NEO4J_PASSWORD`; don't hard-code it.

## Adding a new system (Parts 3–8 of the roadmap)

1. Add a name to `MODES` in `main.py` (e.g. `grounded`). See how
   `retrieved_schema` is handled in `run_benchmark()` for an example.
2. Add a branch in `build_user_prompt()` that builds that system's prompt.
3. If you change the prompt text, bump `PROMPT_VERSION`.
4. Run with `--mode <name>`; `evaluate.py` compares it with the others automatically.
