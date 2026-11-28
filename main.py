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
from grounding import Grounder, prompt_text as grounding_prompt
from value_index import load_catalog
from cypher_guard import Guard, repair_feedback
from query_ir import (IR_SYSTEM_PROMPT, SchemaInfo, compile_ir, feedback_text,
                      ir_user_prompt, parse_plan, validate)


DEFAULT_LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions"

DEFAULT_NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687")
DEFAULT_NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
DEFAULT_NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
DEFAULT_DB_PATH = "cypher_benchmark.sqlite"

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"

DEFAULT_QUERY_TIMEOUT = 30

MODES = ["no_schema", "full_schema", "retrieved_schema", "grounded", "ir"]

PROMPT_VERSION = "v1"


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

    retrieved_context TEXT,

    attempts TEXT,

    prompt_tokens INTEGER,

    completion_tokens INTEGER

);
"""

NEW_COLUMNS = {
    "mode": "TEXT",
    "prompt_version": "TEXT",
    "run_id": "TEXT",
    "raw_response": "TEXT",
    "query_type": "TEXT",
    "retrieved_context": "TEXT",
    "attempts": "TEXT",
    "prompt_tokens": "INTEGER",
    "completion_tokens": "INTEGER",
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


def build_user_prompt(question: str, schema: str, mode: str,
                      grounding: str = "") -> str:

    if mode == "no_schema":
        return f"""
Question:
{question}

Generate the Cypher query according to the instructions.
Return JSON only.
"""

    values_block = f"\n{grounding}\n" if (mode == "grounded" and grounding) else ""

    return f"""
Graph Schema:
{schema}
{values_block}
Question:
{question}

Generate the Cypher query according to the instructions.
Return JSON only.
"""


def extract_json(text: str) -> Dict[str, Any]:

    text = text.strip()

    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

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


TOKENS = {"prompt": 0, "completion": 0}


def _count_tokens(data):
    usage = (data or {}).get("usage") or {}
    TOKENS["prompt"] += int(usage.get("prompt_tokens") or 0)
    TOKENS["completion"] += int(usage.get("completion_tokens") or 0)


def call_llm_json(llama_url, model_name, system, user, temperature=0.0,
                  max_tokens=1024, timeout=300):
    payload = {
        "model": model_name,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    t0 = time.perf_counter()
    response = requests.post(llama_url, json=payload, timeout=timeout)
    seconds = time.perf_counter() - t0
    response.raise_for_status()
    data = response.json()
    _count_tokens(data)
    raw = data["choices"][0]["message"]["content"]
    try:
        return extract_json(raw), raw, seconds
    except ValueError:
        return None, raw, seconds


def generate_with_plan(llama_url, model_name, question, prompt_schema, full_schema,
                       grounding_block, catalog, retries=1):
    sch = SchemaInfo(full_schema)
    feedback, previous = "", None
    calls, seconds, raws = 0, 0.0, []
    status, errors, warnings, plan = None, [], [], None
    for attempt in range(retries + 1):
        user = ir_user_prompt(question, prompt_schema, grounding_block, feedback)
        obj, raw, sec = call_llm_json(llama_url, model_name, IR_SYSTEM_PROMPT, user)
        calls += 1
        seconds += sec
        raws.append(raw)
        if obj is None:
            status = "parse_error"
            break
        plan = parse_plan(obj)
        if plan is None:
            status = "unsupported"
            break
        plan, errors, warnings = validate(plan, sch, catalog)
        if not errors:
            status = "ok" if attempt == 0 else "ok_after_retry"
            break
        status = "invalid"
        previous = plan
        feedback = feedback_text(errors, previous)
    result = {"status": status, "plan": plan, "errors": errors, "warnings": warnings,
              "llm_calls": calls, "seconds": seconds, "raw": raws,
              "cypher": None, "params": {}}
    if status in ("ok", "ok_after_retry"):
        result["cypher"], result["params"] = compile_ir(plan, use_params=True)
    return result


def run_with_repair(guard, cypher, params, max_repairs, repair_empty, llama_url,
                    model_name, question, prompt_schema, grounding_block, grounded,
                    full_schema, catalog):
    attempts, seconds = [], 0.0
    cur, cur_params = cypher, params or {}
    for k in range(max_repairs + 1):
        out = guard.run(cur, cur_params)
        attempts.append({
            "attempt": k, "cypher": cur, "params": cur_params, "stage": out["stage"],
            "ok": out["ok"], "empty": out["empty"], "success": out["success"],
            "error": out["error"], "failures": out["failures"],
            "warnings": out["warnings"], "result": out["result"],
            "n_rows": len(out["result"]) if out["result"] else 0,
            "truncated": out["truncated"], "latency": out["latency"],
        })
        codes = " ".join(f["code"] for f in out["failures"]) or "-"
        print(f"  attempt {k}: stage={out['stage']} ok={out['ok']} empty={out['empty']} "
              f"failures={codes}")
        needs = (not out["success"]) or bool(out["failures"]) or \
            (out["empty"] and repair_empty)
        if not needs or k == max_repairs:
            break
        feedback = repair_feedback(cur, out, full_schema, catalog, grounded)
        user = build_user_prompt(question, prompt_schema,
                                 "grounded" if grounding_block else "full_schema",
                                 grounding=grounding_block) + "\n" + feedback
        try:
            obj, raw, sec = call_llm_json(llama_url, model_name, SYSTEM_PROMPT, user)
        except Exception as e:
            attempts[-1]["repair_note"] = f"repair call failed: {e}"
            break
        seconds += sec
        new = str((obj or {}).get("Cypher") or "").strip()
        if not new:
            attempts[-1]["repair_note"] = "LLM gave no new query"
            break
        cur, cur_params = new, {}
    last = attempts[-1]
    execution = {"success": last["success"], "error": last["error"],
                 "result": last["result"], "latency": last["latency"] or 0.0}
    return execution, attempts, cur, seconds


def generate_cypher(
    llama_url: str,
    model_name: str,
    question: str,
    schema: str,
    mode: str = "full_schema",
    grounding: str = "",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: int = 300,
):

    user_prompt = build_user_prompt(
        question=question,
        schema=schema,
        mode=mode,
        grounding=grounding
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

    _count_tokens(data)
    raw_content = data["choices"][0]["message"]["content"]

    try:
        parsed = extract_json(raw_content)
    except ValueError:
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

    def execute_query(self, cypher: str, params: Optional[Dict[str, Any]] = None):

        if not cypher:
            return {
                "success": False,
                "error": "Empty Cypher query",
                "result": None,
                "latency": None
            }

        start_time = time.perf_counter()

        try:

            with self.driver.session(
                database=self.database,
                default_access_mode=READ_ACCESS
            ) as session:

                result = session.run(
                    Query(cypher, timeout=self.query_timeout),
                    params or {}
                )

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


def serialize_result(result):


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
    retrieved_context,
    attempts,
    prompt_tokens,
    completion_tokens

)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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
    retrieved_context=None,
    attempts=None,
    prompt_tokens=None,
    completion_tokens=None
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
            retrieved_context,
            attempts,
            prompt_tokens,
            completion_tokens
        )
    )

    conn.commit()


def stratified_sample(df, n, seed=0):
    if n >= len(df):
        return df
    if "query_type" not in df.columns:
        return df.sample(n=n, random_state=seed).sort_index()
    sizes = df["query_type"].value_counts()
    quota = (sizes / sizes.sum() * n).round().astype(int)
    while quota.sum() != n:
        diff = n - quota.sum()
        k = (sizes / sizes.sum() * n - quota).idxmax() if diff > 0 else quota.idxmax()
        quota[k] += 1 if diff > 0 else -1
    parts = [df[df["query_type"] == t].sample(n=int(q), random_state=seed)
             for t, q in quota.items() if q > 0]
    return pd.concat(parts).sort_index()


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
    embed_margin: float = 0.1,
    sample: Optional[int] = None,
    sample_seed: int = 0,
    values_file: Optional[str] = None,
    ir_retries: int = 1,
    repair: Optional[int] = None,
    repair_empty: bool = True
):

    if run_id is None:
        tag = mode
        if mode in ("retrieved_schema", "grounded", "ir") and retriever != "bm25":
            tag += f"-{retriever}"
        if distractor_share and mode != "no_schema":
            tag += f"_noise{int(round(distractor_share * 100))}"
        if repair is not None:
            tag += f"_repair{repair}"
        run_id = f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}"

    embedder = None
    if mode in ("retrieved_schema", "grounded", "ir") and retriever != "bm25":
        from embedder import get_embedder
        embedder = get_embedder(embed_model)
    noisy_cache = {}

    grounders, grounder_states, grounder_catalogs = {}, {}, {}

    def get_grounder(ds):
        if ds not in grounders:
            path = values_file or f"values/{ds}.json"
            if not os.path.exists(path):
                raise SystemExit(
                    f"--mode grounded needs a value catalog: {path} not found.\n"
                    f"Build it with: uv run value_index.py --neo4j-database "
                    f"<db> --out {path}")
            cat = load_catalog(path)
            grounders[ds] = Grounder(cat)
            grounder_catalogs[ds] = cat
            grounder_states[ds] = cat.get("state_version")
            print(f"Value catalog {path}: state_version {cat.get('state_version')}, "
                  f"database {cat.get('database')}")
        return grounders[ds]

    print("=" * 70)
    print("TEXT-TO-CYPHER BENCHMARK")
    print(f"mode={mode}  prompt={PROMPT_VERSION}  run_id={run_id}")
    print("=" * 70)


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

    if data_source:
        df = df[df["data_source"] == data_source].reset_index(drop=True)
        print(f"Kept {len(df)} rows with data_source='{data_source}'.")


    if sample:
        df = stratified_sample(df, sample, sample_seed)
        print(f"Stratified random sample: {len(df)} questions (seed {sample_seed})")
        if "query_type" in df.columns:
            print("  " + ", ".join(f"{k}={v}" for k, v in
                                   df["query_type"].value_counts().items()))
    else:
        if end_index is None:
            end_index = len(df)

        df = df.iloc[start_index:end_index]

        print(
            f"Processing rows "
            f"{start_index} -> {end_index - 1}"
        )


    sqlite_conn = initialize_sqlite(sqlite_path)


    neo4j = Neo4jExecutor(
        uri=neo4j_uri,
        username=neo4j_user,
        password=neo4j_password,
        database=neo4j_database,
        query_timeout=query_timeout
    )

    print("\nChecking Neo4j connection...")

    neo4j.verify_connection()

    guards = {}

    def guard_for(ds, schema_text):
        if ds not in guards:
            guards[ds] = Guard(neo4j.driver, neo4j_database, schema_text,
                               guard_catalog(ds), budget={"timeout_s": query_timeout})
        return guards[ds]

    def guard_catalog(ds):
        path = values_file or f"values/{ds}.json"
        if ds in grounder_catalogs:
            return grounder_catalogs[ds]
        return load_catalog(path) if os.path.exists(path) else None

    print("Neo4j connection successful.")


    total = len(df)

    for local_idx, (_, row) in enumerate(df.iterrows(), start=1):

        print("\n" + "=" * 70)
        print(
            f"QUESTION {local_idx}/{total}"
        )
        print("=" * 70)

        TOKENS["prompt"] = TOKENS["completion"] = 0
        question = str(row["question"])
        schema = str(row["schema"])
        real_schema = schema
        info = None

        benchmark_cypher = str(row["cypher"])
        dataset = str(row["data_source"])
        query_type = (
            str(row["query_type"]) if "query_type" in row.index else None
        )

        semantics = load_semantics(dataset, semantics_dir)
        if distractor_share and mode != "no_schema":
            if schema not in noisy_cache:
                noisy, _ = add_distractors(schema, distractor_share, distractor_seed)
                noisy_sem = (merge(rule_semantics(noisy), semantics)
                             if semantics else None)
                noisy_cache[schema] = (json.dumps(noisy), noisy_sem)
            schema, semantics = noisy_cache[schema]

        retrieved_context = None
        grounding_block = ""
        full_schema_text = schema
        if mode in ("retrieved_schema", "grounded", "ir"):
            pruned, info = retrieve_schema(
                question, schema,
                semantics=semantics,
                method=retriever,
                embedder=embedder,
                embed_margin=embed_margin
            )
            info["distractor_share"] = distractor_share
            schema = json.dumps(pruned)
            print("\nRetrieved schema elements:")
            print(f"  labels={info['labels']}  rels={info['rels']}"
                  f"  semantics={info['semantics']}  method={retriever}")
            if mode in ("grounded", "ir"):
                grounder = get_grounder(dataset)
                grounded = grounder.ground(question, schema_labels=set(info["labels"]))
                grounding_block = grounding_prompt(grounded)
                info["grounding"] = grounded
                info["values_state"] = grounder_states.get(dataset)
                print("\nGrounded values:")
                for gv in grounded:
                    print(f"  \"{gv['surface']}\" -> {gv['property']} = "
                          f"{gv['value']!r} ({gv['method']}, {gv['score']})")
            if mode != "ir":
                retrieved_context = json.dumps(info)
        elif distractor_share and mode == "full_schema":
            retrieved_context = json.dumps({"distractor_share": distractor_share,
                                            "seed": distractor_seed})

        print("\nQuestion:")
        print(question)

        print("\nDataset:")
        print(dataset)


        print("\nGenerating Cypher...")

        llm_params = {}
        try:
            plan_result = None
            if mode == "ir":
                plan_result = generate_with_plan(
                    llama_url, model_name, question, schema, full_schema_text,
                    grounding_block, grounder_catalogs.get(dataset), retries=ir_retries)
                print(f"Plan: {plan_result['status']} "
                      f"({plan_result['llm_calls']} LLM call(s))")
                for wmsg in plan_result["warnings"]:
                    print(f"  auto-fix: {wmsg}")
                for emsg in plan_result["errors"]:
                    print(f"  error: {emsg}")

            if plan_result and plan_result["cypher"]:
                generated = {"cypher": plan_result["cypher"], "reason": "compiled from plan",
                             "generation_time": plan_result["seconds"],
                             "raw_response": json.dumps(plan_result["raw"])}
                llm_params = plan_result["params"]
            else:
                generated = generate_cypher(
                    llama_url=llama_url,
                    model_name=model_name,
                    question=question,
                    schema=schema,
                    mode="grounded" if mode == "ir" else mode,
                    grounding=grounding_block
                )
                if plan_result:
                    generated["generation_time"] += plan_result["seconds"]
                    generated["raw_response"] = json.dumps(
                        plan_result["raw"] + [generated["raw_response"]])

            if plan_result:
                info["ir"] = {k: plan_result[k] for k in
                              ("status", "plan", "errors", "warnings", "llm_calls")}
                info["ir"]["fallback"] = not plan_result["cypher"]
                info["ir"]["params"] = llm_params
                retrieved_context = json.dumps(info, default=str)

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
            if llm_params:
                print(f"Parameters: {llm_params}")

        except Exception as e:

            print(
                "\nLLM generation failed:"
            )
            print(str(e))

            llm_cypher = None
            llm_reason = None
            generation_time = None
            raw_response = f"ERROR: {e}"


        print("\nExecuting LLM Cypher...")

        attempts = None
        if llm_cypher and repair is not None:
            llm_execution, attempts, llm_cypher, extra_s = run_with_repair(
                guard_for(dataset, real_schema), llm_cypher, llm_params, repair,
                repair_empty, llama_url, model_name, question, schema,
                grounding_block, info.get("grounding") if info else None,
                real_schema, guard_catalog(dataset))
            if generation_time is not None:
                generation_time += extra_s

        elif llm_cypher:

            llm_execution = neo4j.execute_query(
                llm_cypher, llm_params
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

            retrieved_context=retrieved_context,

            attempts=json.dumps(attempts, default=str) if attempts is not None else None,

            prompt_tokens=TOKENS["prompt"],

            completion_tokens=TOKENS["completion"]
        )

        print("\nSaved to SQLite.")


    neo4j.close()
    sqlite_conn.close()

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    print(
        f"\nSQLite database:"
        f" {sqlite_path}"
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a llama.cpp Text-to-Cypher model "
            "against benchmark Cypher queries."
        )
    )


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
        help="Ending row index (exclusive). Rows are sorted by query type, "
             "so use --sample for a representative subset"
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Run N randomly chosen questions with the same query-type mix "
             "as the full set (overrides --start/--end)"
    )

    parser.add_argument("--sample-seed", type=int, default=0)

    parser.add_argument(
        "--repair",
        type=int,
        default=None,
        help="Part 7: check every query (policy, schema, values, EXPLAIN) before "
             "running it, and repair failures up to N times (0 = check only)"
    )

    parser.add_argument(
        "--no-repair-empty",
        action="store_true",
        help="with --repair: don't treat 'ran but returned no rows' as a failure"
    )

    parser.add_argument(
        "--ir-retries",
        type=int,
        default=1,
        help="--mode ir: how often a rejected plan is sent back with the errors"
    )

    parser.add_argument(
        "--values-file",
        default=None,
        help="--mode grounded: value catalog from value_index.py "
             "(default values/<data_source>.json)"
    )


    parser.add_argument(
        "--mode",
        choices=MODES,
        default="full_schema",
        help="no_schema = LLM only (System 0), "
             "full_schema = LLM + full schema (System 1), "
             "retrieved_schema = LLM + retrieved schema (System 2), "
             "grounded = retrieved schema + graph values (System 3), "
             "ir = plan -> check -> compiled Cypher (System 4)"
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

        embed_margin=args.embed_margin,

        sample=args.sample,

        sample_seed=args.sample_seed,

        values_file=args.values_file,

        ir_retries=args.ir_retries,

        repair=args.repair,

        repair_empty=not args.no_repair_empty
    )


if __name__ == "__main__":
    main()