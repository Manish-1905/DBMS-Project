import json

import main
from final_report import inline_params
from graph_state import rename_cypher, rename_schema


def test_inline_params():
    assert inline_params("MATCH (u:User) WHERE u.gender = $p0 AND u.state IN $p1 RETURN u",
                         {"p0": "Female", "p1": ["Ohio", "O'Hara"]}) == \
        "MATCH (u:User) WHERE u.gender = 'Female' AND u.state IN ['Ohio', 'O\\'Hara'] RETURN u"
    assert inline_params("RETURN $x", {}) == "RETURN $x"


def test_rename_cypher_only_touches_schema_names():
    q = ("MATCH (m:Movie {title: 'Up'})<-[w:WATCHED|RATED]-(u:User), (m)-[:HAS]->(g:Genre) "
         "WHERE m.title = 'Genre' AND g.name = 'Drama' RETURN m.title AS title, w.watchCount")
    assert rename_cypher(q, "R") == (
        "MATCH (m:Movie {name: 'Up'})<-[w:VIEWED|RATED]-(u:User), (m)-[:HAS]->(g:Category) "
        "WHERE m.name = 'Genre' AND g.name = 'Drama' RETURN m.name AS title, w.watchCount")


def test_rename_schema(er_schema):
    s = json.loads(rename_schema(json.dumps(er_schema), "R"))
    assert "Category" in s["node_props"] and "Genre" not in s["node_props"]
    assert [p["property"] for p in s["node_props"]["Movie"]] == ["movieId", "name", "year"]
    assert {"start": "User", "type": "VIEWED", "end": "Movie"} in s["relationships"]
    assert "VIEWED" in s["rel_props"]


def test_token_counter():
    main.TOKENS["prompt"] = main.TOKENS["completion"] = 0
    main._count_tokens({"usage": {"prompt_tokens": 120, "completion_tokens": 30}})
    main._count_tokens({"usage": {"prompt_tokens": 80, "completion_tokens": 10}})
    main._count_tokens({})
    assert main.TOKENS == {"prompt": 200, "completion": 40}
