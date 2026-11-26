#!/usr/bin/env python3

import argparse
import json
import os
import sqlite3
import time
import re
from typing import Any, Dict, Optional

import pandas as pd
import requests
from neo4j import GraphDatabase, Query, READ_ACCESS

from schema_retriever import retrieve as retrieve_schema
from semantic_layer import load_semantics, merge, rule_semantics
from distractors import add_distractors


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions"

DEFAULT_NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687")
DEFAULT_NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
# Set NEO4J_PASSWORD in your shell instead of committing it to git.
DEFAULT_NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
DEFAULT_DB_PATH = "cypher_benchmark.sqlite"

# Change this to whatever model name you expose from llama.cpp.
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"

# Max seconds a single Cypher query may run before Neo4j kills it.
DEFAULT_QUERY_TIMEOUT = 30

# Which system is being tested (roadmap Part 2 / Part 8):
#   no_schema   -> System 0 / A : question only
#   full_schema -> System 1 / B : question + full graph schema
#   retrieved_schema -> System 2 : question + only the relevant part of
#                       the schema (see schema_retriever.py)
# Later parts add new modes here (grounded, ...).
MODES = ["no_schema", "full_schema", "retrieved_schema"]

# Bump this whenever you change SYSTEM_PROMPT or build_user_prompt,
# so results from different prompts are never mixed up.
PROMPT_VERSION = "v1"


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = r"""
Imagine you are a user at a company that needs to present queries to
corresponding question and graph schema.

You have to be very diligent at your job.

Task Nuances-

- while making cypher, Be careful while inducing node values and
  relationship values. They should match letter by letter since they
  will be directly used in the query.

- while making cypher, Be careful while using Node and relationship
  names. They should match letter by letter.

- while making cypher, Be careful while using Node and relationship
  properties. They should match letter by letter.

- While making queries involving dates, make sure to follow these
  formats and principles:

  (i) When comparing dates, convert the date property using a generic
  date string format, for example:
      date(dob) = date("YYYY-MM-DD")

  (ii) To extract date components, use dot notation:
      player.dob.year
      player.dob.month
      player.dob.day

      Do not use functions such as:
      year(player.dob)

  (iii) Some properties can be NULL. Check for not null in queries.

Just give a JSON for the query in question and nothing else.

The JSON must have exactly this structure:

{
    "Cypher": "Cypher statement corresponding to that question",
    "Reason for cypher": "Provide reason why your cypher is correct logically for the natural language question"
}

Return valid JSON only.
"""


# ============================================================
# SQLITE
# ============================================================

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS benchmark_results (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    question TEXT NOT NULL,

    LLM_cypher TEXT,

    LLM_cypher_result TEXT,

    LLM_cypher_latency REAL,

    LLM_cypher_reason TEXT,

    benchmark_cypher TEXT,

    benchmark_cypher_result TEXT,

    benchmark_cypher_latency REAL,

    LLM_model TEXT,

    time_to_generate REAL,

    dataset TEXT,

    mode TEXT,

    prompt_version TEXT,

    run_id TEXT,

    raw_response TEXT,

    query_type TEXT,

    retrieved_context TEXT

);
"""

# Columns added after the first version of this script. Older .sqlite
# files are upgraded automatically so old and new runs live together.
NEW_COLUMNS = {
    "mode": "TEXT",
    "prompt_version": "TEXT",
    "run_id": "TEXT",
    "raw_response": "TEXT",
    "query_type": "TEXT",
    "retrieved_context": "TEXT",
}


def initialize_sqlite(db_path: str):

    conn = sqlite3.connect(db_path)

    conn.execute(CREATE_TABLE_SQL)

    existing = {
        r[1] for r in conn.execute("PRAGMA table_info(benchmark_results)")
    }
    for col, typ in NEW_COLUMNS.items():
        if col not in existing:
            conn.execute(
                f"ALTER TABLE benchmark_results ADD COLUMN {col} {typ}"
            )

    conn.commit()

    return conn


# ============================================================
# LLAMA.CPP
# ============================================================

def build_user_prompt(question: str, schema: str, mode: str) -> str:
    """
    Build the actual prompt sent to the LLM.

    no_schema        : only the natural-language question.
    full_schema      : question + the full graph schema.
    retrieved_schema : question + the pruned schema (the caller passes the
                       already-pruned schema in `schema`).
    """

    if mode == "no_schema":
        return f"""
Question:
{question}

Generate the Cypher query according to the instructions.
Return JSON only.
"""

    return f"""
Graph Schema:
{schema}

Question:
{question}

Generate the Cypher query according to the instructions.
Return JSON only.
"""


def extract_json(text: str) -> Dict[str, Any]:
    """
    Attempts to extract valid JSON from the LLM response.

    Handles cases where the model accidentally returns:

    ```json
    {...}
    ```

    or some extra text around the JSON.
    """

    text = text.strip()

    # Remove markdown code fences.
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    text = text.strip()

    # First attempt: entire response is JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Second attempt: find the first JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:

        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not parse JSON from LLM response:\n{text}"
    )


def generate_cypher(
    llama_url: str,
    model_name: str,
    question: str,
    schema: str,
    mode: str = "full_schema",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: int = 300,
):
    """
    Send question (+ schema, depending on mode) to llama.cpp and return:

        cypher
        reason
        generation_time
        raw_response
    """

    user_prompt = build_user_prompt(
        question=question,
        schema=schema,
        mode=mode
    )

    payload = {
        "model": model_name,

        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        "temperature": temperature,

        "max_tokens": max_tokens,

        # This is supported by llama.cpp's OpenAI-compatible server
        # for models/configurations that support structured output.
        "response_format": {
            "type": "json_object"
        }
    }

    start_time = time.perf_counter()

    response = requests.post(
        llama_url,
        json=payload,
        timeout=timeout
    )

    generation_time = time.perf_counter() - start_time

    response.raise_for_status()

    data = response.json()

    raw_content = data["choices"][0]["message"]["content"]

    try:
        parsed = extract_json(raw_content)
    except ValueError:
        # Keep the raw text so you can see *why* parsing failed.
        return {
            "cypher": None,
            "reason": None,
            "generation_time": generation_time,
            "raw_response": raw_content,
        }

    cypher = parsed.get("Cypher")
    reason = parsed.get("Reason for cypher")

    if cypher is not None:
        cypher = str(cypher).strip()

    if reason is not None:
        reason = str(reason).strip()

    return {
        "cypher": cypher,
        "reason": reason,
        "generation_time": generation_time,
        "raw_response": raw_content,
    }


# ============================================================
# NEO4J
# ============================================================

class Neo4jExecutor:

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        query_timeout: float = DEFAULT_QUERY_TIMEOUT
    ):

        self.driver = GraphDatabase.driver(
            uri,
            auth=(username, password)
        )

        self.database = database
        self.query_timeout = query_timeout

    def verify_connection(self):

        self.driver.verify_connectivity()

    def execute_query(self, cypher: str):

        if not cypher:
            return {
                "success": False,
                "error": "Empty Cypher query",
                "result": None,
                "latency": None
            }

        start_time = time.perf_counter()

        try:

            # READ_ACCESS: Neo4j refuses CREATE / DELETE / SET / MERGE,
            # so an LLM-generated query can never damage the benchmark DB.
            # timeout: runaway queries are killed instead of hanging.
            with self.driver.session(
                database=self.database,
                default_access_mode=READ_ACCESS
            ) as session:

                result = session.run(
                    Query(cypher, timeout=self.query_timeout)
                )

                # Consume the complete result.
                records = result.data()

            latency = time.perf_counter() - start_time

            return {
                "success": True,
                "error": None,
                "result": records,
                "latency": latency
            }

        except Exception as e:

            latency = time.perf_counter() - start_time

            return {
                "success": False,
                "error": str(e),
                "result": None,
                "latency": latency
            }

    def close(self):

        self.driver.close()


# ============================================================
# RESULT SERIALIZATION
# ============================================================

def serialize_result(result):

    """
    Convert Neo4j result into JSON-safe string.

    SQLite stores it as TEXT.
    """

    try:
        return json.dumps(
            result,
            ensure_ascii=False,
            default=str
        )

    except Exception:

        return json.dumps(
            str(result),
            ensure_ascii=False
        )


# ============================================================
# SQLITE INSERT
# ============================================================

INSERT_SQL = """
INSERT INTO benchmark_results (

    question,

    LLM_cypher,
    LLM_cypher_result,
    LLM_cypher_latency,
    LLM_cypher_reason,

    benchmark_cypher,
    benchmark_cypher_result,
    benchmark_cypher_latency,

    LLM_model,
    time_to_generate,
    dataset,

    mode,
    prompt_version,
    run_id,
    raw_response,
    query_type,
    retrieved_context

)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


def save_result(
    conn,
    question,
    llm_cypher,
    llm_result,
    llm_latency,
    llm_reason,
    benchmark_cypher,
    benchmark_result,
    benchmark_latency,
    llm_model,
    generation_time,
    dataset,
    mode=None,
    prompt_version=None,
    run_id=None,
    raw_response=None,
    query_type=None,
    retrieved_context=None
):

    conn.execute(
        INSERT_SQL,
        (
            question,

            llm_cypher,
            serialize_result(llm_result),
            llm_latency,
            llm_reason,

            benchmark_cypher,
            serialize_result(benchmark_result),
            benchmark_latency,

            llm_model,
            generation_time,
            dataset,

            mode,
            prompt_version,
            run_id,
            raw_response,
            query_type,
            retrieved_context
        )
    )

    conn.commit()


# ============================================================
# MAIN BENCHMARK
# ============================================================

def run_benchmark(
    parquet_path: str,
    sqlite_path: str,
    llama_url: str,
    model_name: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str,
    start_index: int = 0,
    end_index: Optional[int] = None,
    mode: str = "full_schema",
    data_source: Optional[str] = None,
    query_timeout: float = DEFAULT_QUERY_TIMEOUT,
    run_id: Optional[str] = None,
    semantics_dir: Optional[str] = "semantics",
    distractor_share: float = 0.0,
    distractor_seed: int = 0,
    retriever: str = "bm25",
    embed_model: str = "BAAI/bge-small-en-v1.5",
    embed_margin: float = 0.1
):

    if run_id is None:
        tag = mode
        if mode == "retrieved_schema" and retriever != "bm25":
            tag += f"-{retriever}"
        if distractor_share and mode != "no_schema":
            tag += f"_noise{int(round(distractor_share * 100))}"
        run_id = f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}"

    embedder = None
    if mode == "retrieved_schema" and retriever != "bm25":
        from embedder import get_embedder
        embedder = get_embedder(embed_model)
    noisy_cache = {}

    print("=" * 70)
    print("TEXT-TO-CYPHER BENCHMARK")
    print(f"mode={mode}  prompt={PROMPT_VERSION}  run_id={run_id}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load Parquet
    # --------------------------------------------------------

    print("\nLoading dataset:")
    print(f"  {parquet_path}")

    df = pd.read_parquet(parquet_path)

    required_columns = [
        "question",
        "schema",
        "cypher",
        "data_source",
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in parquet: {missing}"
        )

    print(f"Loaded {len(df)} rows.")

    # Only keep questions for the graph currently loaded in Neo4j,
    # otherwise gold queries for other graphs will "fail".
    if data_source:
        df = df[df["data_source"] == data_source].reset_index(drop=True)
        print(f"Kept {len(df)} rows with data_source='{data_source}'.")

    # --------------------------------------------------------
    # Optional subset
    # --------------------------------------------------------

    if end_index is None:
        end_index = len(df)

    df = df.iloc[start_index:end_index]

    print(
        f"Processing rows "
        f"{start_index} -> {end_index - 1}"
    )

    # --------------------------------------------------------
    # SQLite
    # --------------------------------------------------------

    sqlite_conn = initialize_sqlite(sqlite_path)

    # --------------------------------------------------------
    # Neo4j
    # --------------------------------------------------------

    neo4j = Neo4jExecutor(
        uri=neo4j_uri,
        username=neo4j_user,
        password=neo4j_password,
        database=neo4j_database,
        query_timeout=query_timeout
    )

    print("\nChecking Neo4j connection...")

    neo4j.verify_connection()

    print("Neo4j connection successful.")

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    total = len(df)

    for local_idx, (_, row) in enumerate(df.iterrows(), start=1):

        print("\n" + "=" * 70)
        print(
            f"QUESTION {local_idx}/{total}"
        )
        print("=" * 70)

        question = str(row["question"])
        schema = str(row["schema"])

        benchmark_cypher = str(row["cypher"])
        dataset = str(row["data_source"])
        query_type = (
            str(row["query_type"]) if "query_type" in row.index else None
        )

        # Part 4.5: pad the schema with fake labels/relationships. Only the
        # schema text changes; the graph and the gold query don't.
        semantics = load_semantics(dataset, semantics_dir)
        if distractor_share and mode != "no_schema":
            if schema not in noisy_cache:
                noisy, _ = add_distractors(schema, distractor_share, distractor_seed)
                noisy_sem = (merge(rule_semantics(noisy), semantics)
                             if semantics else None)
                noisy_cache[schema] = (json.dumps(noisy), noisy_sem)
            schema, semantics = noisy_cache[schema]

        # What schema the LLM actually sees depends on the mode.
        retrieved_context = None
        if mode == "retrieved_schema":
            pruned, info = retrieve_schema(
                question, schema,
                semantics=semantics,
                method=retriever,
                embedder=embedder,
                embed_margin=embed_margin
            )
            info["distractor_share"] = distractor_share
            schema = json.dumps(pruned)
            retrieved_context = json.dumps(info)
            print("\nRetrieved schema elements:")
            print(f"  labels={info['labels']}  rels={info['rels']}"
                  f"  semantics={info['semantics']}  method={retriever}")
        elif distractor_share and mode == "full_schema":
            retrieved_context = json.dumps({"distractor_share": distractor_share,
                                            "seed": distractor_seed})

        print("\nQuestion:")
        print(question)

        print("\nDataset:")
        print(dataset)

        # ====================================================
        # 1. LLM GENERATION
        # ====================================================

        print("\nGenerating Cypher...")

        try:

            generated = generate_cypher(
                llama_url=llama_url,
                model_name=model_name,
                question=question,
                schema=schema,
                mode=mode
            )

            llm_cypher = generated["cypher"]
            llm_reason = generated["reason"]
            generation_time = generated["generation_time"]
            raw_response = generated["raw_response"]

            print(
                f"Generation time: "
                f"{generation_time:.4f} seconds"
            )

            print("\nLLM Cypher:")
            print(llm_cypher)

        except Exception as e:

            print(
                "\nLLM generation failed:"
            )
            print(str(e))

            llm_cypher = None
            llm_reason = None
            generation_time = None
            raw_response = f"ERROR: {e}"

        # ====================================================
        # 2. EXECUTE LLM CYPHER
        # ====================================================

        print("\nExecuting LLM Cypher...")

        if llm_cypher:

            llm_execution = neo4j.execute_query(
                llm_cypher
            )

        else:

            llm_execution = {
                "success": False,
                "error": "LLM did not generate Cypher",
                "result": None,
                "latency": None
            }

        if llm_execution["success"]:

            print(
                "LLM query executed successfully."
            )

            print(
                f"Latency: "
                f"{llm_execution['latency']:.4f} seconds"
            )

        else:

            print(
                "LLM query failed:"
            )

            print(
                llm_execution["error"]
            )

        # ====================================================
        # 3. EXECUTE BENCHMARK CYPHER
        # ====================================================

        print("\nExecuting benchmark Cypher...")

        benchmark_execution = neo4j.execute_query(
            benchmark_cypher
        )

        if benchmark_execution["success"]:

            print(
                "Benchmark query executed successfully."
            )

            print(
                f"Latency: "
                f"{benchmark_execution['latency']:.4f} seconds"
            )

        else:

            print(
                "Benchmark query failed:"
            )

            print(
                benchmark_execution["error"]
            )

        # ====================================================
        # 4. SAVE EVERYTHING
        # ====================================================

        save_result(

            conn=sqlite_conn,

            question=question,

            llm_cypher=llm_cypher,

            llm_result={
                "success": llm_execution["success"],
                "result": llm_execution["result"],
                "error": llm_execution["error"]
            },

            llm_latency=llm_execution["latency"],

            llm_reason=llm_reason,

            benchmark_cypher=benchmark_cypher,

            benchmark_result={
                "success": benchmark_execution["success"],
                "result": benchmark_execution["result"],
                "error": benchmark_execution["error"]
            },

            benchmark_latency=benchmark_execution["latency"],

            llm_model=model_name,

            generation_time=generation_time,

            dataset=dataset,

            mode=mode,

            prompt_version=PROMPT_VERSION,

            run_id=run_id,

            raw_response=raw_response,

            query_type=query_type,

            retrieved_context=retrieved_context
        )

        print("\nSaved to SQLite.")

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    neo4j.close()
    sqlite_conn.close()

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    print(
        f"\nSQLite database:"
        f" {sqlite_path}"
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a llama.cpp Text-to-Cypher model "
            "against benchmark Cypher queries."
        )
    )

    # Dataset

    parser.add_argument(
        "--parquet",
        required=True,
        help="Path to input parquet file"
    )

    parser.add_argument(
        "--sqlite",
        default=DEFAULT_DB_PATH,
        help="Output SQLite database"
    )

    # llama.cpp

    parser.add_argument(
        "--llama-url",
        default=DEFAULT_LLAMA_URL,
        help=(
            "llama.cpp OpenAI-compatible endpoint"
        )
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Model name used by llama.cpp"
    )

    # Neo4j

    parser.add_argument(
        "--neo4j-uri",
        default=DEFAULT_NEO4J_URI
    )

    parser.add_argument(
        "--neo4j-user",
        default=DEFAULT_NEO4J_USER
    )

    parser.add_argument(
        "--neo4j-password",
        default=DEFAULT_NEO4J_PASSWORD
    )

    parser.add_argument(
        "--neo4j-database",
        default="neo4j"
    )

    # Dataset subset

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting row index"
    )

    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Ending row index (exclusive)"
    )

    # Experiment settings

    parser.add_argument(
        "--mode",
        choices=MODES,
        default="full_schema",
        help="no_schema = LLM only (System 0), "
             "full_schema = LLM + full schema (System 1), "
             "retrieved_schema = LLM + retrieved schema (System 2)"
    )

    parser.add_argument(
        "--data-source",
        default=None,
        help="Only run questions for this graph, e.g. 'er' "
             "(must match the database loaded in Neo4j)"
    )

    parser.add_argument(
        "--query-timeout",
        type=float,
        default=DEFAULT_QUERY_TIMEOUT,
        help="Seconds before a Cypher query is killed"
    )

    parser.add_argument(
        "--semantics-dir",
        default="semantics",
        help="retrieved_schema mode uses <dir>/<data_source>.json aliases "
             "if present (see semantic_layer.py); pass '' to disable"
    )

    parser.add_argument(
        "--distractor-share",
        type=float,
        default=0.0,
        help="Part 4.5: fraction of the schema that is fake (e.g. 0.8); "
             "applies to full_schema and retrieved_schema"
    )

    parser.add_argument("--distractor-seed", type=int, default=0)

    parser.add_argument(
        "--retriever",
        choices=["bm25", "embed", "hybrid"],
        default="bm25",
        help="retrieved_schema mode: lexical, embedding or hybrid retrieval"
    )

    parser.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")

    parser.add_argument("--embed-margin", type=float, default=0.1)

    parser.add_argument(
        "--run-id",
        default=None,
        help="Label for this run (default: <mode>_<timestamp>)"
    )

    args = parser.parse_args()

    if not args.neo4j_password:
        parser.error(
            "No Neo4j password. Set NEO4J_PASSWORD or pass --neo4j-password."
        )

    run_benchmark(

        parquet_path=args.parquet,

        sqlite_path=args.sqlite,

        llama_url=args.llama_url,

        model_name=args.model,

        neo4j_uri=args.neo4j_uri,

        neo4j_user=args.neo4j_user,

        neo4j_password=args.neo4j_password,

        neo4j_database=args.neo4j_database,

        start_index=args.start,

        end_index=args.end,

        mode=args.mode,

        data_source=args.data_source,

        query_timeout=args.query_timeout,

        run_id=args.run_id,

        semantics_dir=args.semantics_dir,

        distractor_share=args.distractor_share,

        distractor_seed=args.distractor_seed,

        retriever=args.retriever,

        embed_model=args.embed_model,

        embed_margin=args.embed_margin
    )


if __name__ == "__main__":
    main()