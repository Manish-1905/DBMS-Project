import pandas as pd
import pytest

from evaluate import compare_runs, mcnemar_p
from main import stratified_sample


def _df():
    types = ["complex"] * 60 + ["simple"] * 30 + ["eval"] * 10
    return pd.DataFrame({"question": [f"q{i}" for i in range(100)], "query_type": types})


def test_stratified_sample_keeps_type_mix_and_is_deterministic():
    s = stratified_sample(_df(), 20, seed=0)
    assert len(s) == 20
    assert s["query_type"].value_counts().to_dict() == {"complex": 12, "simple": 6, "eval": 2}
    assert s.equals(stratified_sample(_df(), 20, seed=0))
    assert not s.equals(stratified_sample(_df(), 20, seed=1))


@pytest.mark.parametrize("n", [1, 7, 33, 99])
def test_stratified_sample_exact_size(n):
    assert len(stratified_sample(_df(), n)) == n


def test_mcnemar():
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(5, 5) == 1.0
    assert mcnemar_p(10, 0) == pytest.approx(2 / 1024)
    assert mcnemar_p(12, 8) > 0.05


def test_compare_runs(capsys):
    rows = []
    for i in range(10):
        for run, ok in (("full_x", i < 3), ("retrieved_x", i < 6)):
            rows.append({"run_id": run, "question": f"q{i}", "gold_cypher": "g",
                         "result_correct": ok})
    compare_runs(rows, "full", "retrieved")
    out = capsys.readouterr().out
    assert "only A 0 | only B 3" in out and "30.0%" in out and "60.0%" in out
