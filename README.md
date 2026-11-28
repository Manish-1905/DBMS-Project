# ConText2Cypher

Our DBMS course project. The idea is simple: you ask a question about a graph database in plain English, and the system writes the Cypher query for you and runs it on Neo4j.

The catch is that language models don't know your graph. They guess label names, get relationship directions backwards and use values that aren't in the data. So instead of just asking the model, we give it the parts of the graph it needs for each question (the relevant schema and the real stored values) and we check every query before it runs.

Everything runs locally on a laptop: Neo4j for the database and a small model (Qwen2.5-Coder 1.5B) through llama.cpp.

The full write-up is in [`report/report.pdf`](report/report.pdf).

**Team:** Saurabh Sharma (2024MCB1311), Triman Singh Tuteja (2023EPB1283), Manish Kumar Chatakunda (2024MCB1302), Kushagra Gupta (2024MCB1299), Ipsita Pandey (2023EEB1210)

## Results in short

We tested on the ER graph from the Mind the Query benchmark (856 questions with known correct answers).

| System | What the model gets | Correct answers |
|---|---|---|
| S0 | just the question | 32.8% |
| S1 | question + whole schema | 51.5% |
| S2 | question + only the relevant schema | 55.1% |
| S3 | S2 + the actual values from the graph | 56.2% |
| S5 | S3, but the model writes a plan that we check and turn into Cypher | 53.9% |
| S7 | S5 + checking and repairing queries before they run | 54.7% |

The more interesting part is what happens when the graph changes. When we changed stored values (like `'Female'` to `'F'`) or renamed parts of the schema, a system that doesn't look at the graph dropped to 0%, while ours stayed around 28%. Our checking layer also blocked every harmful query we tried (deletes, drops, admin commands), even when the model was happy to write them.

More numbers are in the [`results/`](results/) folder.

## Setup

You need Python 3.12 with [uv](https://docs.astral.sh/uv/), Neo4j 5 (we used Neo4j Desktop) and llama.cpp (`brew install llama.cpp` on a Mac).

```bash
git clone https://github.com/Manish-1905/DBMS-Project.git
cd DBMS-Project
uv sync
```

**Load the graph.** The database dump is too big for git, so get `entity-resolution-50.dump` from the Mind the Query release (or one of us) and put it in `Datasets/`. In Neo4j Desktop, create a DBMS, import the dump and start it. If the data ends up in a database that isn't called `neo4j`, pass its name with `--neo4j-database`.

**Start the model** in its own terminal and leave it running:

```bash
llama-server -hf Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:Q4_K_M --port 8080 -c 8192
```

Use `-c 16384` if you run the noisy-schema experiments, since those prompts get long.

**Set your Neo4j details** in the terminal you'll run things from:

```bash
export NEO4J_PASSWORD=your_password
export DB=neo4j
```

## Running it

`main.py` runs one system over the questions and saves everything (generated query, results, timings) to a SQLite file. The `--mode` flag picks the system:

| Mode | System |
|---|---|
| `no_schema` | S0 |
| `full_schema` | S1 |
| `retrieved_schema` | S2 |
| `grounded` | S3 (needs the value catalog, see below) |
| `ir` | S5 |
| add `--repair 2` | S7 |

A quick test on 5 questions:

```bash
uv run main.py --parquet Benchmark/MTQ-Auto/test-split.parquet --neo4j-database $DB --mode retrieved_schema --sample 5 --sqlite test.sqlite
uv run evaluate.py --sqlite test.sqlite
```

Drop `--sample` to run all 856 questions. If you want a smaller set, use `--sample N` rather than `--end N`, because the file is sorted by question type and `--end` would only give you the hardest ones.

For `grounded` and `ir` you first need the list of values stored in the graph:

```bash
uv run value_index.py --neo4j-database $DB --out values/er.json
```

### Looking at results

```bash
uv run evaluate.py --sqlite results.sqlite --by-type
uv run evaluate.py --sqlite results.sqlite --compare grounded ir
uv run error_taxonomy.py --sqlite results.sqlite --values values/er.json
```

`--compare` checks whether the difference between two runs is real or just luck (a paired McNemar test). `error_taxonomy.py` sorts the wrong answers by what went wrong (syntax error, wrong direction, made-up property and so on).

`final_report.py` puts several runs side by side in one table, which is what we used for the report.

## Other experiments

* **Schema retrieval on its own:** `eval_retrieval.py` checks whether the retriever keeps everything a question needs. It doesn't need Neo4j or the model.
* **Noisy schema:** `eval_distractors.py` adds fake labels and relationships to the schema to see how retrieval holds up. `main.py --distractor-share 0.8` does the same for a full run.
* **Changed graph:** `graph_state.py` copies the graph into a new database and changes its values or renames parts of its schema, then rewrites the correct answers to match. Your original database isn't touched.
* **Safety:** `security_bench.py` throws harmful queries and requests at the system and checks that none of them get through. Add `--offline` to skip the model and database.

## What's where

* `main.py`: runs a system over the questions
* `evaluate.py`, `final_report.py`, `error_taxonomy.py`: scoring and comparing runs
* `schema_inspect.py`, `semantic_layer.py`, `schema_graph.py`: reading the schema from Neo4j and adding descriptions and other names for labels
* `schema_retriever.py`, `embedder.py`: picking the relevant part of the schema
* `value_index.py`, `grounding.py`: matching words in the question to values in the graph
* `query_ir.py`: the query plan format, its checks and the plan-to-Cypher compiler
* `cypher_guard.py`: safety and correctness checks before a query runs, plus repair
* `graph_state.py`, `distractors.py`: building the changed-graph and noisy-schema setups
* `Benchmark/MTQ-Auto/`: the questions, plus the variants we made
* `semantics/`: descriptions and other names for the ER schema
* `results/`: results from each stage of the project
* `report/`: the final report and the script that draws its charts
* `tests/`: run them with `uv run pytest`

## Notes

Generated queries always run in a read-only session with a time limit, so a bad query can't change the database. With `--repair`, the checking layer also rejects anything that writes, drops or reads files before it reaches Neo4j.

Keep your password in `NEO4J_PASSWORD` rather than in the code.
