from evaluate import results_match
from main import build_user_prompt
from schema_graph import build_rows


def test_schema_graph_rows(er_schema):
    er_schema["value_stats"] = {"User.state": {"distinct": 2, "non_null": 9,
                                               "top_values": [{"value": "CA", "freq": 5}]}}
    sem = {"labels": {"User": {"description": "A viewer", "aliases": ["viewer"]}}}
    rows = build_rows(er_schema, sem, "er")
    assert {n["name"] for n in rows["node_types"]} == {"User", "Movie", "Genre", "IpAddress"}
    assert len(rows["triples"]) == 3
    user = next(n for n in rows["node_types"] if n["name"] == "User")
    assert user["aliases"] == ["viewer"]
    state = next(p for p in rows["node_props"] if p["name"] == "state")
    assert state["distinct"] == 2 and "CA" in state["top_values"]
    assert rows["rel_props"][0]["owner"] == "WATCHED"


def test_prompt_modes():
    assert "Graph Schema" not in build_user_prompt("q?", "S", "no_schema")
    assert "Graph Schema:\nS" in build_user_prompt("q?", "S", "full_schema")
    assert "Graph Schema:\nS" in build_user_prompt("q?", "S", "retrieved_schema")


def test_results_match_ignores_column_names_and_order():
    gold = [{"name": "A", "c": 1}, {"name": "B", "c": 2}]
    llm = [{"x": 2, "y": "B"}, {"x": 1, "y": "A"}]
    assert results_match(llm, gold, ordered=False)
    assert not results_match(llm, gold, ordered=True)
    assert not results_match(llm[:1], gold, ordered=False)
