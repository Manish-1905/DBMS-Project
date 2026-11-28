import json

import pytest

import cypher_guard as cg
import main
from error_taxonomy import labels_for
from query_ir import SchemaInfo
from security_bench import BENIGN_CYPHER, MALICIOUS_CYPHER

CAT = {"properties": {"User.gender": {"values": {"Female": 1, "Male": 1}},
                      "Movie.title": {"values": {**{f"T{i}": 1 for i in range(300)},
                                                 "Four Christmases": 1}}}}


def codes(fails):
    return sorted({f["code"] for f in fails})


@pytest.mark.parametrize("name,q", MALICIOUS_CYPHER)
def test_every_harmful_query_is_rejected(name, q):
    assert cg.policy_check(q), name


@pytest.mark.parametrize("q", BENIGN_CYPHER)
def test_harmless_queries_pass_policy(q):
    assert cg.policy_check(q) == []


@pytest.mark.parametrize("q,expected", [
    ("MATCH (u:Users) RETURN u", ["F2"]),
    ("MATCH (u:User)-[:WATCH]->(m:Movie) RETURN m", ["F3"]),
    ("MATCH (m:Movie)-[:WATCHED]->(u:User) RETURN u", ["F4"]),
    ("MATCH (u:User)<-[:WATCHED]-(m:Movie) RETURN u", ["F4"]),
    ("MATCH (u:User)-[:WATCHED]->(m:Movie) RETURN m.releaseYear", ["F5"]),
    ("MATCH (m:Movie) WHERE m.year = '2000' RETURN m", ["F6"]),
    ("MATCH (u:User)-[:WATCHED]->(m:Movie) RETURN count(*)", []),
    ("MATCH (u:User)-[:WATCHED]-(m:Movie) RETURN count(*)", []),
])
def test_schema_check(er_schema, q, expected):
    assert codes(cg.schema_check(q, er_schema)) == expected


def test_value_check():
    f = cg.value_check("MATCH (u:User {gender: 'female'}) RETURN u", CAT)
    assert codes(f) == ["F8"] and "the graph stores 'Female'" in f[0]["message"]
    f = cg.value_check("MATCH (m:Movie) WHERE m.title = 'Four Chrismases' RETURN m", CAT)
    assert codes(f) == ["F7"] and "'Four Christmases'" in f[0]["message"]
    assert cg.value_check("MATCH (u:User {gender: 'Male'}) RETURN u", CAT) == []


@pytest.mark.parametrize("msg,code", [
    ("Invalid input 'RETRUN': expected", "F1"),
    ("The transaction has been terminated. Transaction timed out", "F12"),
    ("Writing in read access mode not allowed", "F13"),
    ("Type mismatch: expected Integer but was String", "F6"),
])
def test_db_errors(msg, code):
    assert cg.classify_db_error(msg) == code


def test_repair_feedback_says_why(er_schema):
    q = "MATCH (m:Movie)-[:WATCHED]->(u:User {gender: 'female'}) RETURN count(m)"
    out = {"failures": cg.schema_check(q, er_schema) + cg.value_check(q, CAT), "empty": True}
    fb = cg.repair_feedback(q, out, er_schema, CAT)
    assert "the graph has (User)-[:WATCHED]->(Movie)" in fb
    assert "the graph stores 'Female'" in fb
    assert "(User)-[:WATCHED]->(Movie)" in fb.split("Valid relationships")[1]
    empty = cg.repair_feedback("MATCH (u:User) RETURN u", {"failures": [], "empty": True},
                               er_schema, grounded=[{"property": "User.state", "value": "Ohio"}])
    assert "returned NO rows" in empty and "User.state = 'Ohio'" in empty


class FakeGuard(cg.Guard):
    def __init__(self, schema, catalog=None):
        self.schema, self.catalog, self.budget = SchemaInfo(schema), catalog, dict(cg.DEFAULT_BUDGET)

    def explain(self, c, p=None):
        return [], [], 10

    def execute(self, c, p=None):
        rows = [{"n": 7}] if "'Female'" in c and c.startswith("MATCH (u:User") else []
        return {"success": True, "error": None, "result": rows, "truncated": False, "latency": 0.0}


def test_guard_blocks_before_executing(er_schema):
    out = FakeGuard(er_schema).run("MATCH (n) DETACH DELETE n")
    assert out["stage"] == "static" and not out["success"] and codes(out["failures"]) == ["F13"]


def test_repair_loop_fixes_query(er_schema, monkeypatch):
    good = "MATCH (u:User {gender: 'Female'})-[:WATCHED]->(m:Movie) RETURN count(m) AS n"
    seen = []

    def fake_llm(url, model, system, user, **kw):
        seen.append(user)
        return {"Cypher": good}, "", 0.1

    monkeypatch.setattr(main, "call_llm_json", fake_llm)
    bad = "MATCH (m:Movie)-[:WATCHED]->(u:User {gender: 'female'}) RETURN count(m) AS n"
    ex, attempts, final, sec = main.run_with_repair(
        FakeGuard(er_schema, CAT), bad, {}, 2, True, "x", "m", "q?", json.dumps(er_schema),
        "", None, er_schema, CAT)
    assert [a["ok"] for a in attempts] == [False, True]
    assert codes(attempts[0]["failures"]) == ["F4", "F8"]
    assert final == good and ex["result"] == [{"n": 7}]
    assert "the graph stores 'Female'" in seen[0]


def test_no_repair_when_first_attempt_is_fine(er_schema, monkeypatch):
    monkeypatch.setattr(main, "call_llm_json", lambda *a, **k: pytest.fail("no repair expected"))
    good = "MATCH (u:User {gender: 'Female'})-[:WATCHED]->(m:Movie) RETURN count(m) AS n"
    _, attempts, _, _ = main.run_with_repair(FakeGuard(er_schema, CAT), good, {}, 2, True,
                                             "x", "m", "q?", "S", "", None, er_schema, CAT)
    assert len(attempts) == 1 and attempts[0]["ok"]


@pytest.mark.parametrize("cy,runnable,error,expected", [
    ("MATCH (u:User RETURN u", False, "Invalid input 'RETURN'", ["F1"]),
    ("MATCH (u:User {gender: 'female'})-[:WATCHED]->(m:Movie) RETURN count(m)", True, None, ["F8"]),
    ("MATCH (u:User {gender: 'Female'}) RETURN count(u)", True, None, ["F9"]),
    ("MATCH (u:User {gender: 'Female'})-[:WATCHED]->(m:Movie) RETURN avg(m.year)", True, None, ["F10"]),
    ("MATCH (u:User {gender: 'Female'})-[:WATCHED]->(m:Movie) RETURN count(m)", True, None, ["F14"]),
])
def test_taxonomy(er_schema, cy, runnable, error, expected):
    row = {"llm_cypher": cy, "runnable": runnable, "llm_error": error,
           "gold_cypher": "MATCH (u:User {gender: 'Female'})-[:WATCHED]->(m:Movie) RETURN count(m)"}
    got, _ = labels_for(row, SchemaInfo(er_schema), CAT)
    assert got == expected
