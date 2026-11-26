"""Part 3 exit criterion: deterministic schema, version changes with the schema."""
import random

import pytest

from schema_inspect import _type, compare_with_mtq, schema_version


def test_version_is_deterministic(er_schema):
    assert schema_version(er_schema) == schema_version(er_schema)


def test_version_ignores_ordering(er_schema):
    shuffled = dict(er_schema)
    shuffled["relationships"] = list(reversed(er_schema["relationships"]))
    shuffled["node_props"] = {k: list(reversed(v))
                              for k, v in reversed(list(er_schema["node_props"].items()))}
    assert schema_version(shuffled) == schema_version(er_schema)


def test_version_ignores_values_and_indexes(er_schema):
    with_extras = dict(er_schema, value_stats={"User.state": {"distinct": 50}},
                       indexes=[{"name": "x"}])
    assert schema_version(with_extras) == schema_version(er_schema)


@pytest.mark.parametrize("change", [
    "new_label", "new_property", "new_relationship", "flipped_direction", "type_change"])
def test_version_changes_when_schema_changes(er_schema, change):
    before = schema_version(er_schema)
    if change == "new_label":
        er_schema["node_props"]["Order"] = []
    elif change == "new_property":
        er_schema["node_props"]["Movie"].append({"property": "rating", "datatype": "FLOAT"})
    elif change == "new_relationship":
        er_schema["relationships"].append({"start": "User", "type": "RATED", "end": "Movie"})
    elif change == "flipped_direction":
        er_schema["relationships"][0] = {"start": "Movie", "type": "WATCHED", "end": "User"}
    elif change == "type_change":
        er_schema["node_props"]["Movie"][2]["datatype"] = "STRING"
    assert schema_version(er_schema) != before


@pytest.mark.parametrize("raw,expected", [
    ("String", "STRING"), ("Long", "INTEGER"), ("Double", "FLOAT"),
    ("StringArray", "LIST"), ("STRING NOT NULL", "STRING"), ("INTEGER", "INTEGER"),
    ("LIST<STRING NOT NULL>", "LIST"), ("ZONED DATETIME", "DATE_TIME"),
    ("DateTime", "DATE_TIME"), ("Boolean", "BOOLEAN"),
])
def test_type_names_from_old_and_new_neo4j(raw, expected):
    assert _type([raw]) == expected


def test_compare_with_mtq_full_recall(er_schema):
    report = compare_with_mtq(er_schema, er_schema)
    assert all(v == 1.0 for k, v in report.items() if k.endswith("_recall"))


def test_compare_with_mtq_reports_missing(er_schema):
    partial = dict(er_schema, node_props={"User": er_schema["node_props"]["User"]})
    report = compare_with_mtq(partial, er_schema)
    assert report["labels_recall"] == pytest.approx(0.25)
    assert "Movie" in report["labels_missing"]


def test_random_permutations_same_version(er_schema):
    v = schema_version(er_schema)
    for seed in range(5):
        rnd = random.Random(seed)
        rels = er_schema["relationships"][:]
        rnd.shuffle(rels)
        assert schema_version(dict(er_schema, relationships=rels)) == v
