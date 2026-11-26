"""Part 6: query plan (IR) validation, auto-fixes, compilation."""
import pytest

from query_ir import compile_ir, feedback_text, parse_plan, validate

CAT = {"properties": {"User.gender": {"values": {"Female": 1, "Male": 1}},
                      "Genre.name": {"values": {"Drama": 1, "Comedy": 1}}}}


def plan(**kw):
    base = {"nodes": [{"var": "u", "label": "User"}, {"var": "m", "label": "Movie"}],
            "paths": [{"from": "u", "rel": "WATCHED", "to": "m", "var": "w"}],
            "return": [{"agg": "count", "expr": "m", "as": "n"}]}
    base.update(kw)
    return base


def test_valid_plan_compiles_with_parameters(er_schema):
    ir, errs, warns = validate(plan(filters=[{"expr": "u.gender", "op": "=", "value": "Female"}]),
                               er_schema, CAT)
    assert errs == [] and warns == []
    cypher, params = compile_ir(ir)
    assert cypher == ("MATCH (u:User)-[w:WATCHED]->(m:Movie) WHERE u.gender = $p0 "
                      "RETURN count(m) AS n")
    assert params == {"p0": "Female"}


def test_certain_mistakes_are_fixed(er_schema):
    p = plan(nodes=[{"var": "u", "label": "user"}, {"var": "m", "label": "Movie"}],
             paths=[{"from": "m", "rel": "watched", "to": "u"}],
             filters=[{"expr": "u.Gender", "op": "=", "value": "female"}])
    ir, errs, warns = validate(p, er_schema, CAT)
    assert errs == []
    assert len(warns) == 5          # label, rel, direction, property, value
    cypher, params = compile_ir(ir)
    assert "(u:User)-[:WATCHED]->(m:Movie)" in cypher and params == {"p0": "Female"}


@pytest.mark.parametrize("bad,message", [
    ({"nodes": [{"var": "u", "label": "Users"}], "return": [{"expr": "u"}]},
     "label 'Users' does not exist; did you mean User?"),
    (plan(filters=[{"expr": "m.releaseYear", "op": ">", "value": 2000}]),
     "Movie has no property 'releaseYear'; did you mean year?"),
    (plan(nodes=[{"var": "u", "label": "User"}, {"var": "m", "label": "Movie"},
                 {"var": "g", "label": "Genre"}]),
     "the pattern is not connected"),
    (plan(paths=[{"from": "u", "rel": "HAS", "to": "m"}]),
     "HAS connects (Movie)-[:HAS]->(Genre)"),
    (plan(filters=[{"expr": "u.gender", "op": "=", "value": "Woman"}]),
     "User.gender has no value 'Woman'; existing values: 'Female', 'Male'"),
    (plan(filters=[{"expr": "u.gender", "op": "LIKE", "value": "F"}]), "operator 'LIKE' not allowed"),
    (plan(**{"return": [{"agg": "median", "expr": "m", "as": "x"}]}), "aggregate 'median' not allowed"),
])
def test_errors_explain_why(er_schema, bad, message):
    _, errs, _ = validate(bad, er_schema, CAT)
    assert any(message in e for e in errs), errs


def test_two_stage_aggregation(er_schema):
    p = plan(**{"with": {"group_by": [{"expr": "u.gender", "as": "gender"}, {"expr": "u"}],
                         "aggregates": [{"agg": "count", "expr": "m", "distinct": True,
                                         "as": "nMovies"}],
                         "having": [{"expr": "nMovies", "op": ">", "value": 2}]},
                "return": [{"expr": "u.gender"}, {"agg": "avg", "expr": "nMovies", "as": "avgMovies"}],
                "order_by": [{"expr": "avgMovies", "desc": True}], "limit": 3})
    ir, errs, _ = validate(p, er_schema)
    assert errs == []
    cypher, params = compile_ir(ir)
    assert cypher == ("MATCH (u:User)-[w:WATCHED]->(m:Movie) "
                      "WITH u.gender AS gender, u, count(DISTINCT m) AS nMovies WHERE nMovies > $p0 "
                      "RETURN gender, avg(nMovies) AS avgMovies ORDER BY avgMovies DESC LIMIT 3")
    assert params == {"p0": 2}


def test_after_with_only_kept_things_exist(er_schema):
    p = plan(**{"with": {"group_by": [{"expr": "u"}],
                         "aggregates": [{"agg": "count", "expr": "m", "as": "c"}]},
                "return": [{"expr": "m.title"}]})
    _, errs, _ = validate(p, er_schema)
    assert any("not available after 'with'" in e for e in errs)


def test_or_filters_and_null_checks(er_schema):
    p = plan(filters=[{"or": [{"expr": "m.year", "op": "<", "value": 1990},
                              {"expr": "m.title", "op": "STARTS WITH", "value": "The"}]},
                      {"expr": "u.state", "op": "is not null"}],
             **{"return": [{"expr": "m.title", "as": "title"}], "distinct": True})
    ir, errs, _ = validate(p, er_schema)
    assert errs == []
    cypher, params = compile_ir(ir)
    assert "WHERE (m.year < $p0 OR m.title STARTS WITH $p1) AND u.state IS NOT NULL" in cypher
    assert "RETURN DISTINCT m.title AS title" in cypher


def test_literal_mode(er_schema):
    ir, _, _ = validate(plan(filters=[{"expr": "u.state", "op": "IN", "value": ["Ohio", "Texas"]}]),
                        er_schema)
    cypher, params = compile_ir(ir, use_params=False)
    assert "u.state IN ['Ohio', 'Texas']" in cypher and params == {}


def test_parse_plan_and_feedback():
    assert parse_plan({"unsupported": True}) is None
    assert parse_plan({"Cypher": "MATCH ..."}) is None
    assert parse_plan({"plan": {"nodes": [1], "return": [1]}}) == {"nodes": [1], "return": [1]}
    assert "- label 'X' does not exist" in feedback_text(["label 'X' does not exist"], {})


def test_label_names_used_as_variables_are_fixed(er_schema):
    p = plan(paths=[{"from": "u", "rel": "WATCHED", "to": "m"},
                    {"from": "m", "rel": "HAS", "to": "Genre"},
                    {"from": "User", "rel": "USES", "to": "IpAddress"}],
             **{"return": [{"expr": "Genre.name", "as": "genre"}]})
    ir, errs, warns = validate(p, er_schema)
    assert errs == [], errs
    cypher, _ = compile_ir(ir)
    assert "(m)-[:HAS]->(genre:Genre)" in cypher and "(u)-[:USES]->(ipAddress:IpAddress)" in cypher
    assert "RETURN genre.name AS genre" in cypher


@pytest.mark.parametrize("item,expected", [
    ({"agg": "average", "expr": "m.year", "as": "x"}, "avg(m.year) AS x"),
    ({"agg": "count_distinct", "expr": "u", "as": "x"}, "count(DISTINCT u) AS x"),
    ({"expr": "count(DISTINCT u)", "as": "x"}, "count(DISTINCT u) AS x"),
    ({"expr": "SUM(w.watchCount)", "as": "x"}, "sum(w.watchCount) AS x"),
    ({"expr": "count(*)", "as": "x"}, "count(*) AS x"),
])
def test_aggregate_slips_are_fixed(er_schema, item, expected):
    ir, errs, _ = validate(plan(**{"return": [item]}), er_schema)
    assert errs == [], errs
    assert compile_ir(ir)[0].endswith("RETURN " + expected)
