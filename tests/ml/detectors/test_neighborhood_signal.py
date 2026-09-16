import numpy as np
import pytest

from ml.detectors.trustguard.neighborhood import DefaultNeighborhoodSignalExtractor
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


def test_neighborhood_agreement_and_purity():
    extractor = DefaultNeighborhoodSignalExtractor()
    config = TrustGuardConfig(neighborhood_k=3, layers=(1,))

    # 4 train samples: 3 class A at (1, 0), 1 class B at (0, 1)
    train_matrix = np.array([
        [1.0, 0.0],
        [0.99, 0.01],
        [0.98, 0.02],
        [0.0, 1.0],
    ])
    train_labels = ["A", "A", "A", "B"]
    train_reps = _make_reps(["tr0", "tr1", "tr2", "tr3"], train_matrix)

    extractor.fit(train_reps, config, labels=train_labels)

    # Test sample near cluster A with label A (high agreement -> low anomaly)
    # Test sample near cluster A with label B (mislabeled / backdoor -> 0 agreement -> high anomaly)
    # Unlabelled test sample near cluster A (purity 1.0 -> low anomaly)
    test_matrix = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    test_labels = ["A", "B", None]
    test_reps = _make_reps(["te_clean", "te_anom", "te_unlab"], test_matrix)

    result = extractor.extract(test_reps, config, labels=test_labels)
    assert len(result.items) == 3

    # 1. Labelled matching
    clean_item = result.items[0]
    assert clean_item.status == "SUCCESS"
    assert clean_item.provenance == "train_ground_truth"
    assert clean_item.raw_value == 1.0  # 100% neighbor agreement
    assert clean_item.normalized_value == 0.0  # 0 anomaly

    # 2. Labelled mismatch (poisoned / flipped)
    anom_item = result.items[1]
    assert anom_item.status == "SUCCESS"
    assert anom_item.raw_value == 0.0  # 0% agreement
    assert anom_item.normalized_value == 1.0  # maximal anomaly

    # 3. Unlabelled: uses local purity
    unlab_item = result.items[2]
    assert unlab_item.status == "UNLABELLED_PURITY"
    assert unlab_item.provenance == "unsupervised_neighborhood"
    assert unlab_item.raw_value == 1.0  # 100% purity
    assert unlab_item.normalized_value == 0.0


def test_neighborhood_insufficient_samples_handling():
    extractor = DefaultNeighborhoodSignalExtractor()
    # Request k=10 but train set only has 2 samples
    config = TrustGuardConfig(neighborhood_k=10, layers=(1,))

    train_reps = _make_reps(["tr0", "tr1"], np.array([[1.0, 0.0], [0.0, 1.0]]))
    extractor.fit(train_reps, config, labels=["A", "B"])

    test_reps = _make_reps(["te0"], np.array([[1.0, 0.0]]))
    result = extractor.extract(test_reps, config, labels=["A"])

    assert result.metadata["effective_k"] == 2
    assert result.items[0].details["effective_k"] == 2
    assert len(result.items[0].details["neighbor_labels"]) == 2


def test_neighborhood_no_test_leakage():
    extractor = DefaultNeighborhoodSignalExtractor()
    config = TrustGuardConfig(neighborhood_k=2, layers=(1,))

    train_reps = _make_reps(["tr0", "tr1"], np.array([[1.0, 0.0], [0.8, 0.2]]))
    extractor.fit(train_reps, config, labels=["TRAIN_A", "TRAIN_A"])

    # Test samples with totally different label
    test_reps = _make_reps(["te0"], np.array([[1.0, 0.0]]))
    res = extractor.extract(test_reps, config, labels=["TEST_LEAK_LABEL"])

    # Neighbors returned must strictly come from TRAIN labels
    assert res.items[0].details["neighbor_labels"] == ["TRAIN_A", "TRAIN_A"]
