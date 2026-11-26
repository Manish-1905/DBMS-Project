"""Part 5: gold values, grounding, value variants, graph states, grounded prompt."""
import pytest

from gold_schema import gold_literals, literal_spans
from graph_state import rewrite_cypher, state_mapping, transform_value
from grounding import Grounder, prompt_text
from main import build_user_prompt
from make_value_variants import partial, typo, variants_for
from value_index import state_version

CATALOG = {"properties": {
    "User.gender": {"values": {"Female": 25, "Male": 25}},
    "User.country": {"values": {"US": 40, "Canada": 10}},
    "User.state": {"values": {"Ohio": 3, "New York": 4, "Washington": 2}},
    "User.firstName": {"values": {**{f"Name{i}": 1 for i in range(300)}, "Jacquelyn": 1}},
    "Genre.name": {"values": {"Comedy": 5, "Drama": 5, "Fantasy": 3}},
    "Movie.title": {"values": {**{f"Film {i}": 1 for i in range(300)},
                               "Four Christmases": 1, "Up": 1, "Bill Traylor: Chasing Ghosts": 1,
                               "Matrix, The": 1}},
}}


@pytest.fixture(scope="module")
def g():
    return Grounder(CATALOG)


def found(g, q, **kw):
    return {(x["property"], x["value"]) for x in g.ground(q, **kw)}


# ---------- gold values ----------

def test_gold_literals():
    q = ("MATCH (u:User {gender: 'Female'})-[:WATCHED]->(m:Movie) WHERE u.state = 'Ohio' "
         "AND m.title IN ['A', 'B'] AND u.email ENDS WITH 'x.com' RETURN m")
    assert gold_literals(q) == {("User.gender", "Female"), ("User.state", "Ohio"),
                                ("Movie.title", "A"), ("Movie.title", "B")}
    assert gold_literals("MATCH (g:Genre) WHERE g.name STARTS WITH 'Co' RETURN g") == set()


def test_literal_spans_point_at_quotes():
    q = "MATCH (u:User) WHERE u.gender = 'Female' RETURN u"
    (start, end, prop, value), = literal_spans(q)
    assert q[start:end] == "'Female'" and prop == "User.gender"


# ---------- grounding ----------

@pytest.mark.parametrize("question,expected,method", [
    ("movies watched by female users", ("User.gender", "Female"), "case"),
    ("users from ohio", ("User.state", "Ohio"), "case"),
    ("users from 'US'", ("User.country", "US"), "exact"),
    ("users in the USA", ("User.country", "US"), "alias"),
    ("movies watched by women", ("User.gender", "Female"), "alias"),
    ("users from NY", ("User.state", "New York"), "alias"),
    ("users who like comedies", ("Genre.name", "Comedy"), "stem"),
    ("the genre 'Fanatsy'", ("Genre.name", "Fantasy"), "typo"),
    ("movies watched by femlae users", ("User.gender", "Female"), "typo"),
    ("who watched 'Four Chrismases'", ("Movie.title", "Four Christmases"), "typo"),
    ("who watched 'Bill Traylor'", ("Movie.title", "Bill Traylor: Chasing Ghosts"), "fuzzy"),
    ("users whose first name is jacquelyn", ("User.firstName", "Jacquelyn"), "case"),
])
def test_grounding_methods(g, question, expected, method):
    res = g.ground(question)
    hit = [x for x in res if (x["property"], x["value"]) == expected]
    assert hit, res
    assert hit[0]["method"] == method


def test_common_words_are_not_titles(g):
    # "up" is a movie title, but a lowercase common word must not match it
    assert ("Movie.title", "Up") not in found(g, "what is it about up?")
    assert ("Movie.title", "Up") in found(g, "who watched 'Up'?")


def test_retrieved_schema_context_breaks_ties(g):
    res = g.ground("users from Washington", schema_labels={"User"})
    assert res[0]["value"] == "Washington" and res[0]["property"] == "User.state"


def test_prompt_text(g):
    txt = prompt_text(g.ground("female users from ohio"))
    assert "User.gender = 'Female'" in txt and "User.state = 'Ohio'" in txt
    assert prompt_text([]) == ""


def test_grounded_prompt_only_in_grounded_mode():
    block = "Values that exist in the graph ...\n- \"female\" -> User.gender = 'Female'"
    assert block in build_user_prompt("q?", "S", "grounded", grounding=block)
    assert block not in build_user_prompt("q?", "S", "retrieved_schema", grounding=block)


def test_state_version_tracks_values():
    a = {"User.gender": {"values": {"Female": 1, "Male": 1}}}
    b = {"User.gender": {"values": {"F": 1, "M": 1}}}
    assert state_version(a) == state_version({"User.gender": {"values": {"Male": 9, "Female": 2}}})
    assert state_version(a) != state_version(b)


# ---------- value variants ----------

def test_variant_helpers():
    assert typo("Fantasy") == "Fanatsy"
    assert partial("Bill Traylor: Chasing Ghosts") == "Bill Traylor"
    assert partial("A Glitch in the Matrix") == "Glitch in the Matrix"
    v = dict((k, q) for k, q, _ in variants_for(
        "Which users from 'Ohio' watched movies?", "User.state", "Ohio"))
    assert v["lowercase"] == "Which users from ohio watched movies?"
    assert v["unquoted"] == "Which users from Ohio watched movies?"
    assert v["alias"] == "Which users from 'OH' watched movies?"


# ---------- graph states ----------

def test_transforms():
    assert transform_value("Ohio", "postal") == "OH"
    assert transform_value("The Matrix", "article_last") == "Matrix, The"
    assert transform_value("Drama", "upper") == "DRAMA"
    assert transform_value("Female", {"Female": "F"}) == "F"


def test_rewrite_gold_is_property_aware():
    m = state_mapping("B")
    q = ("MATCH (u:User {gender: 'Female'})-[:WATCHED]->(m:Movie)-[:HAS]->(g:Genre) "
         "WHERE u.country = 'US' AND g.name IN ['Drama', 'Comedy'] "
         "AND u.firstName = 'Female' RETURN m.title")
    out, changed = rewrite_cypher(q, m)
    assert changed
    assert "{gender: 'F'}" in out and "u.country = 'United States'" in out
    assert "['DRAMA', 'COMEDY']" in out
    assert "u.firstName = 'Female'" in out          # other property untouched


def test_rewrite_gold_unaffected_query():
    q = "MATCH (m:Movie) WHERE m.year > 2000 RETURN m.title"
    assert rewrite_cypher(q, state_mapping("B")) == (q, False)
