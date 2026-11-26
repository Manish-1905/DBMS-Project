import json

import semantic_layer as sl


def test_rule_aliases(er_schema):
    sem = sl.rule_semantics(er_schema)
    assert "ip address" in sem["labels"]["IpAddress"]["aliases"]
    assert "movies" in sem["labels"]["Movie"]["aliases"]
    assert sem["properties"]["User.firstName"]["aliases"] == ["first name"]
    assert sem["properties"]["WATCHED.watchCount"]["aliases"] == ["watch count"]


def test_merge_later_layer_wins_and_aliases_union():
    a = {"labels": {"Movie": {"description": "old", "aliases": ["movies"]}}}
    b = {"labels": {"Movie": {"description": "new", "aliases": ["film", "movies"]}}}
    m = sl.merge(a, b)
    assert m["labels"]["Movie"] == {"description": "new", "aliases": ["movies", "film"]}


def test_llm_layer_drops_invented_properties(er_schema, monkeypatch):
    def fake_post(url, json=None, timeout=None):
        prompt = json["messages"][1]["content"]
        name = prompt.splitlines()[0].split(": ")[1]
        body = {"description": f"About {name}", "aliases": ["Film", name, "x" * 60],
                "properties": {"title": {"description": "t", "aliases": ["name"]},
                               "madeUp": {"description": "?", "aliases": ["?"]}}}

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": __import__("json").dumps(body)}}]}
        return R()

    monkeypatch.setattr(sl.requests, "post", fake_post)
    sem = sl.llm_semantics(er_schema, "http://x", verbose=False)
    assert sem["labels"]["Movie"]["aliases"] == ["film"]        # lower-cased, own name + junk dropped
    assert "Movie.title" in sem["properties"]
    assert not any(k.endswith(".madeUp") for k in sem["properties"])


def test_build_writes_file(er_schema, tmp_path):
    schema_file = tmp_path / "er.json"
    schema_file.write_text(json.dumps(er_schema))
    (tmp_path / "sem").mkdir()
    (tmp_path / "sem" / "er.overrides.json").write_text(json.dumps(
        {"labels": {"User": {"description": "A viewer", "aliases": ["viewer"]}}}))
    sl.build("er", None, schema_file=str(schema_file), out_dir=str(tmp_path / "sem"))
    sem = sl.load_semantics("er", str(tmp_path / "sem"))
    assert sem["sources"] == ["rules", "overrides"]
    assert sem["labels"]["User"]["description"] == "A viewer"
    assert "viewer" in sem["labels"]["User"]["aliases"]
