import copy
from typing import Any
import pytest

from ml.data.schemas import LabelStatus, Sample, Split
from ml.detectors.schemas import DetectionResult
from ml.purification.purifier import DatasetPurifier
from ml.purification.schemas import (
    DatasetPurificationResult,
    PurificationConfig,
)


def _make_samples(n: int = 6) -> list[Sample]:
    return [
        Sample(
            sample_id=f"s_{i}",
            text=f"Clean text content for sample {i}" if i < 4 else f"Poisoned text content for sample {i} with trigger",
            label="POS" if i % 2 == 0 else "NEG",
            label_status=LabelStatus.KNOWN,
            split=Split.TRAIN,
            dataset_id="benchmark_ds",
            dataset_version="v1.0",
            poison_ground_truth=True if i >= 4 else False,
            original_label="POS" if i % 2 == 0 else "NEG",
            original_label_status=LabelStatus.KNOWN,
        )
        for i in range(n)
    ]


def test_purification_remove_policy_invariance_and_correctness():
    purifier = DatasetPurifier()
    original_samples = _make_samples(6)
    samples_backup = copy.deepcopy(original_samples)

    # Simulated TrustGuard detection scores:
    # Samples 0, 1, 2, 3: low suspicion (0.1, 0.15, 0.2, 0.25)
    # Samples 4, 5: high suspicion (0.85, 0.90)
    detection = DetectionResult(
        sample_ids=[s.sample_id for s in original_samples],
        scores=[0.10, 0.15, 0.20, 0.25, 0.85, 0.90],
        is_anomalous=[False, False, False, False, True, True],
        layer_scores={},
        detector_name="trustguard-multi-signal",
        signal_results={
            "semantic": {"items": [{"sample_id": s.sample_id, "normalized_value": sc} for s, sc in zip(original_samples, [0.1, 0.15, 0.2, 0.25, 0.85, 0.90], strict=True)]}
        },
    )

    config = PurificationConfig(
        policy="REMOVE",
        risk_threshold=0.50,
        purified_version_suffix="purified_v1",
        experiment_fingerprint="exp_fp_123",
    )

    result = purifier.purify(original_samples, detection, config=config, threshold=0.50)

    # 1. Verify original dataset is completely unmodified
    assert len(original_samples) == len(samples_backup)
    for orig, backup in zip(original_samples, samples_backup, strict=True):
        assert orig.model_dump() == backup.model_dump()

    # 2. Verify partition correctness: retained + isolated = original
    assert result.metrics.total_samples == 6
    assert result.metrics.retained_samples == 4
    assert result.metrics.isolated_samples == 2
    assert len(result.retained_dataset) == 4
    assert len(result.isolated_dataset) == 2
    assert result.metrics.retained_samples + result.metrics.isolated_samples == result.metrics.total_samples

    # Retained dataset version mapped to purified version
    for ret_s in result.retained_dataset:
        assert ret_s.dataset_version == "v1.0_purified_v1"
        assert ret_s.purification_action == "RETAINED"
        assert ret_s.sample_weight == 1.0
        assert ret_s.suspicion_score < 0.50
        assert ret_s.trust_score > 0.50

    # Isolated dataset preserved original version and marked ISOLATED
    for iso_s in result.isolated_dataset:
        assert iso_s.dataset_version == "v1.0"
        assert iso_s.purification_action == "ISOLATED"
        assert iso_s.sample_weight == 0.0
        assert iso_s.suspicion_score >= 0.50

    # 3. Verify metrics with ground truth
    # Ground truth: s_0..s_3 are False (clean), s_4..s_5 are True (poison)
    # Decision flagged s_4 and s_5: Perfect detection (TP=2, FP=0, TN=4, FN=0)
    assert result.metrics.true_positives == 2
    assert result.metrics.false_positives == 0
    assert result.metrics.true_negatives == 4
    assert result.metrics.false_negatives == 0
    assert result.metrics.precision == 1.0
    assert result.metrics.recall == 1.0
    assert result.metrics.f1_score == 1.0


def test_purification_reweight_policy():
    purifier = DatasetPurifier()
    original_samples = _make_samples(4)

    detection = DetectionResult(
        sample_ids=[s.sample_id for s in original_samples],
        scores=[0.1, 0.4, 0.7, 0.9],
        is_anomalous=[False, False, True, True],
        layer_scores={},
        detector_name="trustguard-multi-signal",
    )

    config = PurificationConfig(
        policy="REWEIGHT",
        risk_threshold=0.5,
    )

    result = purifier.purify(original_samples, detection, config=config, threshold=0.5)

    # In REWEIGHT policy, all samples are kept in retained dataset with continuous weights
    assert len(result.retained_dataset) == 4
    assert len(result.isolated_dataset) == 2  # Samples >= 0.5 flagged

    # Sample weights match trust_scores
    assert abs(result.retained_dataset[0].sample_weight - 0.9) < 1e-4  # 1.0 - 0.1
    assert abs(result.retained_dataset[1].sample_weight - 0.6) < 1e-4  # 1.0 - 0.4
    assert abs(result.retained_dataset[2].sample_weight - 0.3) < 1e-4  # 1.0 - 0.7
    assert abs(result.retained_dataset[3].sample_weight - 0.1) < 1e-4  # 1.0 - 0.9


def test_purification_without_ground_truth():
    purifier = DatasetPurifier()
    # Unlabelled samples without poison_ground_truth
    samples = [
        Sample(
            sample_id="u_0",
            text="Unlabelled text 0",
            label=None,
            label_status=LabelStatus.UNKNOWN,
            split=Split.TRAIN,
            dataset_id="unlab_ds",
            dataset_version="v1",
        ),
        Sample(
            sample_id="u_1",
            text="Unlabelled text 1",
            label=None,
            label_status=LabelStatus.UNKNOWN,
            split=Split.TRAIN,
            dataset_id="unlab_ds",
            dataset_version="v1",
        ),
    ]

    detection = DetectionResult(
        sample_ids=["u_0", "u_1"],
        scores=[0.2, 0.8],
        is_anomalous=[False, True],
        layer_scores={},
        detector_name="trustguard-multi-signal",
    )

    result = purifier.purify(samples, detection, threshold=0.5)

    assert result.metrics.total_samples == 2
    assert result.metrics.retained_samples == 1
    assert result.metrics.isolated_samples == 1
    assert result.metrics.true_positives is None  # Ground truth absent


def test_purification_id_mismatch_error():
    purifier = DatasetPurifier()
    samples = _make_samples(2)

    detection = DetectionResult(
        sample_ids=["mismatched_id_1", "mismatched_id_2"],
        scores=[0.1, 0.9],
        is_anomalous=[False, True],
        layer_scores={},
        detector_name="trustguard-multi-signal",
    )

    with pytest.raises(ValueError, match="Input samples contain IDs missing from detection result"):
        purifier.purify(samples, detection)
