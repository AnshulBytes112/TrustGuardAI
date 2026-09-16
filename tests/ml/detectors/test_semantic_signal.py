import numpy as np
import pytest

from ml.data.schemas import LabelStatus, Sample, Split
from ml.detectors.trustguard.schemas import TrustGuardConfig
from ml.detectors.trustguard.semantic import DefaultSemanticSignalExtractor
from ml.features.schemas import RepresentationResult


def _make_reps(sample_ids: list[str], matrix: np.ndarray) -> RepresentationResult:
    return RepresentationResult(
        sample_ids=sample_ids,
        representations=matrix,
        layer_representations={1: matrix},
        model_name="test-model",
        max_length=64,
    )


def test_semantic_multiclass_prototypes_and_margins():
    extractor = DefaultSemanticSignalExtractor()
    config = TrustGuardConfig(layers=(1,))

    # 3 classes: c0: [1, 0, 0], c1: [0, 1, 0], c2: [0, 0, 1]
    train_matrix = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.1],
        [0.0, 0.0, 1.0],
    ])
    train_labels = ["c0", "c0", "c1", "c1", "c2"]
    train_reps = _make_reps(["tr0", "tr1", "tr2", "tr3", "tr4"], train_matrix)

    extractor.fit(train_reps, config, labels=train_labels)
    assert len(extractor.prototypes) == 3
    assert "c0" in extractor.prototypes
    assert "c1" in extractor.prototypes
    assert "c2" in extractor.prototypes

    # Evaluate test samples
    test_matrix = np.array([
        [1.0, 0.0, 0.0],   # highly consistent with c0
        [0.0, 1.0, 0.0],   # labelled as c0, but closer to c1 (inconsistent / anomalous)
    ])
    test_labels = ["c0", "c0"]
    test_reps = _make_reps(["te0", "te1"], test_matrix)

    result = extractor.extract(test_reps, config, labels=test_labels)

    assert result.signal_type == "semantic"
    assert len(result.scores) == 2
    assert len(result.items) == 2

    clean_item = result.items[0]
    assert clean_item.sample_id == "te0"
    assert clean_item.status == "SUCCESS"
    assert clean_item.provenance == "train_ground_truth"
    assert clean_item.raw_value is not None and clean_item.raw_value > 0.5  # High positive margin
    assert clean_item.normalized_value is not None and clean_item.normalized_value < 0.3  # Low anomaly

    anom_item = result.items[1]
    assert anom_item.sample_id == "te1"
    assert anom_item.status == "SUCCESS"
    assert anom_item.raw_value is not None and anom_item.raw_value < 0.0  # Negative margin (closer to c1)
    assert anom_item.normalized_value is not None and anom_item.normalized_value > 0.5  # High anomaly


def test_semantic_unlabelled_samples_with_and_without_predictions():
    extractor = DefaultSemanticSignalExtractor()
    config = TrustGuardConfig(layers=(1,))

    train_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    train_labels = ["clean", "poison"]
    train_reps = _make_reps(["tr0", "tr1"], train_matrix)
    extractor.fit(train_reps, config, labels=train_labels)

    test_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
    test_reps = _make_reps(["u0", "u1"], test_matrix)

    # 1. Without model predictions: marked SKIPPED_UNLABELLED
    res_no_pred = extractor.extract(test_reps, config, labels=[None, None])
    assert res_no_pred.items[0].status == "SKIPPED_UNLABELLED"
    assert res_no_pred.items[0].provenance == "unlabelled_no_ground_truth"
    assert res_no_pred.items[0].raw_value is None
    assert res_no_pred.items[0].normalized_value == 0.5

    # 2. With model predictions: marked PREDICTED_LABEL
    res_pred = extractor.extract(
        test_reps,
        config,
        labels=[None, None],
        predicted_labels=["clean", "poison"],
    )
    assert res_pred.items[0].status == "PREDICTED_LABEL"
    assert res_pred.items[0].provenance == "model_prediction"
    assert res_pred.items[0].raw_value is not None
    assert res_pred.items[0].normalized_value is not None


def test_semantic_single_class_edge_case():
    extractor = DefaultSemanticSignalExtractor()
    config = TrustGuardConfig(layers=(1,))

    train_matrix = np.array([[1.0, 0.0], [1.0, 0.1]])
    train_labels = ["only_class", "only_class"]
    train_reps = _make_reps(["tr0", "tr1"], train_matrix)

    extractor.fit(train_reps, config, labels=train_labels)
    assert len(extractor.prototypes) == 1

    test_reps = _make_reps(["te0"], np.array([[1.0, 0.0]]))
    res = extractor.extract(test_reps, config, labels=["only_class"])
    assert res.items[0].status == "SUCCESS"
    assert res.items[0].details["best_alternative_class"] == "NONE"


def test_semantic_empty_or_unfitted_errors():
    extractor = DefaultSemanticSignalExtractor()
    config = TrustGuardConfig(layers=(1,))
    empty_reps = _make_reps([], np.empty((0, 2)))

    with pytest.raises(ValueError, match="at least one sample"):
        extractor.fit(empty_reps, config)

    with pytest.raises(RuntimeError, match="must be fitted"):
        extractor.extract(_make_reps(["s1"], np.array([[1.0, 0.0]])), config)
