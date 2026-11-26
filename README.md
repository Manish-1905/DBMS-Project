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
| `Benchmark/MTQ-Auto/test.parquet` | MTQ questions (`question`, `schema`, `cypher`, `data_source`, ...) |
| `Datasets/*.dump` | Neo4j database dumps (one per graph) |

---

## 1. Install the tools (once)

| Tool | Why | How |
|---|---|---|
| Python 3.12 + **uv** | runs the scripts | `pip install uv` (or see docs.astral.sh/uv) |
| **Neo4j 5.x** | the graph database | Docker (`neo4j:5`) **or** Neo4j Desktop |
| **llama.cpp** | serves the LLM locally | prebuilt release from github.com/ggml-org/llama.cpp, or `brew install llama.cpp` |

## 2. Get the code and Python packages

```bash
git clone https://github.com/TrimanSingh/ConText2Cypher.git
cd ConText2Cypher
uv sync            # creates .venv and installs neo4j, pandas, pyarrow, requests, duckdb
```

## 3. Load a graph into Neo4j

> ⚠️ `Datasets/entity-resolution-50.dump` in the repo is **0 bytes** (the upload failed).
> Get the real dump again (from the MTQ release or a teammate) before this step.
> Don't commit large dumps to git; put a download link here instead (or use Git LFS).

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
```

Useful flags: `--start 0 --end 5` (quick test on 5 questions), `--query-timeout 30`,
`--run-id my_label`.

`--data-source` must match the graph currently loaded in Neo4j, or the gold queries will fail.
`--model` is just the label saved in the results; llama-server serves whatever model it loaded.

## 6. Get the metrics

```bash
uv run evaluate.py --sqlite results.sqlite --rows-csv per_question.csv
```

Prints one row per run with: runnable, non-null, result correct, exact match, execution times,
generation time. `per_question.csv` has every question with its error message, for error analysis.

---

## Safety notes

* LLM queries run in a **read-only session** with a timeout, so a generated `DELETE` cannot wipe the graph.
* The password comes from `NEO4J_PASSWORD`; don't hard-code it.

## Adding a new system (Parts 3–8 of the roadmap)

1. Add a name to `MODES` in `main.py` (e.g. `retrieved_schema`).
2. Add a branch in `build_user_prompt()` that builds that system's prompt.
3. If you change the prompt text, bump `PROMPT_VERSION`.
4. Run with `--mode <name>`; `evaluate.py` compares it with the others automatically.
