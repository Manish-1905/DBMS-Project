from schema_retriever import retrieve


def test_keeps_what_the_question_needs(er_schema):
    pruned, info = retrieve("How many movies did each user watch?", er_schema)
    assert {"User", "Movie"} <= set(info["labels"])
    assert "WATCHED" in info["rels"]
    assert "IpAddress" not in pruned["node_props"]


def test_connects_distant_labels(er_schema):
    _, info = retrieve("Which genres does each user like?", er_schema)
    assert {"User", "Movie", "Genre"} <= set(info["labels"])
    assert {"WATCHED", "HAS"} <= set(info["rels"])


def test_fallback_to_full_schema(er_schema):
    pruned, info = retrieve("zzz qqq", er_schema)
    assert info["fallback"] and pruned == er_schema


def test_aliases_find_label(er_schema):
    question = "List the films released after 2000"
    _, without = retrieve(question, er_schema)
    semantics = {"labels": {"Movie": {"aliases": ["film", "films"], "description": ""}}}
    _, with_sem = retrieve(question, er_schema, semantics=semantics)
    assert "Movie" not in without["labels"] or without["fallback"]
    assert "Movie" in with_sem["labels"] and with_sem["semantics"]
