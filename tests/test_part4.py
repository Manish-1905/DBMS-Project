"""Part 4: distractors, ranked metrics, embedding / hybrid retrieval."""
import json

import pytest

from distractors import add_distractors, n_distractors
from embedder import HashEmbedder
from eval_retrieval import ranked_metrics
from schema_retriever import _clip, rank_elements, retrieve


@pytest.mark.parametrize("share,expected", [(0, 0), (0.4, 5), (0.8, 28), (0.95, 133)])
def test_distractor_counts_for_er(er_schema, share, expected):
    assert n_distractors(7, share) == expected
    noisy, info = add_distractors(er_schema, share)
    assert info["n_distractors"] == expected


def test_distractors_are_deterministic_and_leave_input_alone(er_schema):
    before = json.dumps(er_schema, sort_keys=True)
    a, _ = add_distractors(er_schema, 0.8, seed=1)
    b, _ = add_distractors(er_schema, 0.8, seed=1)
    c, _ = add_distractors(er_schema, 0.8, seed=2)
    assert a == b and a != c
    assert json.dumps(er_schema, sort_keys=True) == before


def test_real_schema_kept_and_never_rewired(er_schema):
    noisy, info = add_distractors(er_schema, 0.95)
    real = set(er_schema["node_props"])
    for L, props in er_schema["node_props"].items():
        assert noisy["node_props"][L] == props
    for r in er_schema["relationships"]:
        assert r in noisy["relationships"]
    for r in noisy["relationships"]:
        if r["type"] in info["rels"]:
            # a distractor relationship always touches a distractor label
            assert not ({r["start"], r["end"]} <= real)
    assert not (set(info["labels"]) & real)


def test_ranked_metrics():
    m = ranked_metrics({"User", "Movie"}, ["Movie", "Genre", "User"], "label")
    assert m["label_R@1"] == 0.5 and m["label_R@3"] == 1.0 and m["label_MRR"] == 1.0
    m = ranked_metrics({"User"}, ["Genre", "Movie"], "label")
    assert m["label_R@3"] == 0.0 and m["label_MRR"] == 0.0
    assert ranked_metrics(set(), [], "rel")["rel_MRR"] == 1.0


def test_clip_keeps_only_near_best():
    assert _clip({"a": 0.9, "b": 0.85, "c": 0.5}, 0.1) == pytest.approx(
        {"a": 0.1, "b": 0.05, "c": 0.0})


@pytest.mark.parametrize("method", ["embed", "hybrid"])
def test_embedding_methods_run_and_prune(er_schema, method):
    noisy, info = add_distractors(er_schema, 0.8)
    pruned, rinfo = retrieve("How many movies did each user watch?", noisy,
                             method=method, embedder=HashEmbedder())
    assert {"User", "Movie"} <= set(rinfo["labels"])
    assert len(pruned["node_props"]) < len(noisy["node_props"])
    rl, _ = rank_elements("How many movies did each user watch?", noisy,
                          method=method, embedder=HashEmbedder())
    assert len(rl) == len(noisy["node_props"])   # full ranking, no cutoff


def test_embed_needs_embedder(er_schema):
    with pytest.raises(ValueError):
        retrieve("movies", er_schema, method="embed")
