import json

import numpy as np
import pytest

import stambo


def test_to_dict_matches_source_array_field_by_field():
    rng = np.random.default_rng(0)
    s1 = rng.normal(0.0, 1.0, 60)
    s2 = rng.normal(0.3, 1.0, 60)

    report = stambo.two_sample_test(s1, s2, statistics={"mean": np.mean}, n_bootstrap=200, seed=0, silent=True)
    as_dict = stambo.to_dict(report)

    arr = report["mean"]
    entry = as_dict["mean"]
    assert entry["p_value"] == pytest.approx(arr[0])
    assert entry["diff"] == pytest.approx(arr[1])
    assert entry["ci_es"] == pytest.approx((arr[2], arr[3]))
    assert entry["emp_s1"] == pytest.approx(arr[4])
    assert entry["ci_s1"] == pytest.approx((arr[5], arr[6]))
    assert entry["emp_s2"] == pytest.approx(arr[7])
    assert entry["ci_s2"] == pytest.approx((arr[8], arr[9]))


def test_to_dict_is_json_serializable_with_plain_python_floats():
    rng = np.random.default_rng(1)
    y_test = rng.integers(0, 2, 100)
    preds_1 = rng.uniform(0.0, 1.0, 100)
    preds_2 = np.clip(y_test * 0.5 + rng.normal(0.0, 0.3, 100), 0.0, 1.0)

    report = stambo.compare_models(y_test, preds_1, preds_2, ("ROCAUC", "AP"), n_bootstrap=100, seed=1, silent=True)
    as_dict = stambo.to_dict(report)

    # The raw report itself is not JSON-serializable (numpy.ndarray values).
    with pytest.raises(TypeError):
        json.dumps(report)

    # But to_dict's output always is, with plain Python floats (not numpy.float64).
    dumped = json.dumps(as_dict)
    round_tripped = json.loads(dumped)
    assert round_tripped["ROCAUC"]["p_value"] == pytest.approx(as_dict["ROCAUC"]["p_value"])
    for stat in as_dict:
        for key in ("p_value", "diff", "emp_s1", "emp_s2"):
            assert type(as_dict[stat][key]) is float
