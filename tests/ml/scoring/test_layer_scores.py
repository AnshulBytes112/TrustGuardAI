import pytest

from ml.scoring.layer_scores import LayerDecomposer


def test_layer_decomposer_single_sample():
    decomposer = LayerDecomposer()
    scores = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.8, 6: 0.9}

    profile = decomposer.decompose_sample("sample_1", scores)

    assert profile.sample_id == "sample_1"
    assert profile.dominant_layer == 6
    assert profile.trajectory == "late"
    assert abs(sum(profile.layer_attributions.values()) - 1.0) < 1e-4


def test_layer_decomposer_early_trajectory():
    decomposer = LayerDecomposer()
    scores = {1: 0.9, 2: 0.8, 3: 0.2, 4: 0.1, 5: 0.1, 6: 0.05}

    profile = decomposer.decompose_sample("sample_2", scores)
    assert profile.dominant_layer == 1
    assert profile.trajectory == "early"


def test_layer_decomposer_empty_fails():
    decomposer = LayerDecomposer()
    with pytest.raises(ValueError, match="cannot be empty"):
        decomposer.decompose_sample("s1", {})
