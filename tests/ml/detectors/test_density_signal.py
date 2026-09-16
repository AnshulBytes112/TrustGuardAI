import numpy as np
import pytest

from ml.detectors.trustguard.density import DefaultDensitySignalExtractor
from ml.detectors.trustguard.schemas import TrustGuardConfig
from ml.features.schemas import RepresentationResult


def _make_reps(sample_ids: list[str], matrix: np.ndarray) -> RepresentationResult:
    return RepresentationResult(
        sample_ids=sample_ids,
        representations=matrix,
        layer_representations={1: matrix},
        model_name="test-model",
        max_length=64,
    )


def test_knn_density_extractor_deterministic_and_leak_free():
    extractor = DefaultDensitySignalExtractor()
    config = TrustGuardConfig(neighborhood_k=3, density_method="knn_distance", layers=(1,))

    # Cluster at (1, 0)
    train_matrix = np.array([
        [1.0, 0.0],
        [0.99, 0.01],
        [0.98, 0.02],
        [0.97, 0.03],
    ])
    train_reps = _make_reps(["tr0", "tr1", "tr2", "tr3"], train_matrix)
    extractor.fit(train_reps, config)

    # Inlier near (1, 0), outlier far at (0, 1)
    test_matrix = np.array([
        [0.99, 0.01],   # Inlier
        [0.0, 1.0],     # Outlier
    ])
    test_reps = _make_reps(["te_inlier", "te_outlier"], test_matrix)

    result = extractor.extract(test_reps, config)
    assert len(result.items) == 2

    inlier = result.items[0]
    outlier = result.items[1]

    assert inlier.status == "SUCCESS"
    assert outlier.status == "SUCCESS"
    assert inlier.provenance == "train_manifold"
    assert outlier.provenance == "train_manifold"

    # Outlier must have significantly higher anomaly score
    assert outlier.normalized_value > inlier.normalized_value
    assert outlier.raw_value > inlier.raw_value
    assert "train_min_dist" in outlier.details


def test_local_outlier_factor_density_extractor():
    extractor = DefaultDensitySignalExtractor()
    config = TrustGuardConfig(neighborhood_k=3, density_method="local_outlier_factor", layers=(1,))

    train_matrix = np.array([
        [1.0, 0.0],
        [0.99, 0.01],
        [0.98, 0.02],
        [0.97, 0.03],
        [0.96, 0.04],
    ])
    train_reps = _make_reps(["tr0", "tr1", "tr2", "tr3", "tr4"], train_matrix)
    extractor.fit(train_reps, config)

    test_matrix = np.array([
        [0.98, 0.02],   # Inlier
        [0.0, 1.0],     # Outlier
    ])
    test_reps = _make_reps(["te_inlier", "te_outlier"], test_matrix)
    result = extractor.extract(test_reps, config)

    inlier = result.items[0]
    outlier = result.items[1]

    assert outlier.normalized_value >= inlier.normalized_value
    assert outlier.details["density_method"] == "local_outlier_factor"
    assert "raw_lof_score" in outlier.details
