from gold_schema import gold_schema_set


def test_basic_path():
    g = gold_schema_set("MATCH (u:User)-[w:WATCHED]->(m:Movie)-[:HAS]->(g:Genre) "
                        "WHERE u.state = 'CA' RETURN g.name, w.watchCount")
    assert g["labels"] == {"User", "Movie", "Genre"}
    assert g["rels"] == {"WATCHED", "HAS"}
    assert g["props"] == {"User.state", "Genre.name", "WATCHED.watchCount"}


def test_inline_map_and_strings_ignored():
    g = gold_schema_set("MATCH (u:User {gender: 'Female'}) WHERE u.firstName = 'a:b' "
                        "RETURN u.state")
    assert g["labels"] == {"User"}
    assert g["props"] == {"User.gender", "User.firstName", "User.state"}


def test_rel_alternatives_and_backticks():
    g = gold_schema_set("MATCH (a:`Ip Address`)<-[:USES|:OWNS]-(u:User) RETURN count(*)")
    assert g["labels"] == {"Ip Address", "User"}
    assert g["rels"] == {"USES", "OWNS"}


def test_empty():
    assert gold_schema_set(None) == {"labels": set(), "rels": set(), "props": set()}
