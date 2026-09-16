from collections.abc import Sequence
from typing import Any
import numpy as np
import pytest

from ml.data.schemas import LabelStatus, Sample, Split
from ml.detectors.trustguard import (
    DefaultSignalScorer,
    DefaultValidationCalibrator,
    SampleSignalResult,
    SampleTrustAssessment,
    SignalResult,
    TrustGuardConfig,
    TrustGuardDetector,
    TrustGuardScoreResult,
    optimize_validation_weights,
)
from ml.evaluation.calibration import ThresholdCalibrationConfig, ThresholdCalibrator
from ml.features.schemas import RepresentationResult


def _make_sample(sample_id: str, label: str = "POS", poison: bool = False, split: Split = Split.TRAIN) -> Sample:
    return Sample(
        sample_id=sample_id,
        text=f"Sample text for {sample_id}",
        label=label,
        label_status=LabelStatus.KNOWN,
        split=split,
        dataset_id="test_ds",
        dataset_version="1.0",
        poison_ground_truth=poison,
    )


def _make_signal_result(signal_name: str, scores: list[float], sample_ids: list[str]) -> SignalResult:
    items = [
        SampleSignalResult(
            sample_id=sid,
            signal_name=signal_name,  # type: ignore
            raw_value=score,
            normalized_value=score,
            status="SUCCESS",
            provenance="test",
            config_fingerprint="fp123",
            details={},
        )
        for sid, score in zip(sample_ids, scores, strict=True)
    ]
    return SignalResult(
        signal_type=signal_name,  # type: ignore
        scores=scores,
        sample_ids=sample_ids,
        items=items,
        metadata={},
    )


def test_trust_and_suspicion_score_mathematical_equations():
    scorer = DefaultSignalScorer()
    config = TrustGuardConfig(
        enabled_signals=["semantic", "neighborhood", "stability", "density"],
        scoring_strategy="weighted_fusion",
        weighting_strategy="equal",
    )

    sample_ids = ["s1", "s2"]
    # Suspicion scores (a_j)
    sr_sem = _make_signal_result("semantic", [0.2, 0.8], sample_ids)
    sr_nei = _make_signal_result("neighborhood", [0.1, 0.9], sample_ids)
    sr_sta = _make_signal_result("stability", [0.3, 0.7], sample_ids)
    sr_den = _make_signal_result("density", [0.4, 0.6], sample_ids)

    result = scorer.assess([sr_sem, sr_nei, sr_sta, sr_den], config, threshold=0.5)

    assert isinstance(result, TrustGuardScoreResult)
    assert len(result.assessments) == 2

    # Weights for 4 equal signals: 0.25 each
    for w in result.weights.values():
        assert abs(w - 0.25) < 1e-4

    # Sample 1:
    # Expected Suspicion = 0.25*(0.2 + 0.1 + 0.3 + 0.4) = 0.25 * 1.0 = 0.25
    # Expected Trust = 1.0 - 0.25 = 0.75
    s1_eval = result.assessments[0]
    assert s1_eval.sample_id == "s1"
    assert abs(s1_eval.suspicion_score - 0.25) < 1e-4
    assert abs(s1_eval.trust_score - 0.75) < 1e-4
    assert s1_eval.prediction is False  # 0.25 < threshold 0.5
    assert abs(s1_eval.trust_score + s1_eval.suspicion_score - 1.0) < 1e-6

    # Verify contributions
    assert abs(s1_eval.contributions["semantic"] - (0.25 * 0.2)) < 1e-4
    assert abs(s1_eval.contributions["density"] - (0.25 * 0.4)) < 1e-4
    assert s1_eval.dominant_signal == "density"  # highest contribution (0.10)

    # Sample 2:
    # Expected Suspicion = 0.25*(0.8 + 0.9 + 0.7 + 0.6) = 0.25 * 3.0 = 0.75
    # Expected Trust = 1.0 - 0.75 = 0.25
    s2_eval = result.assessments[1]
    assert s2_eval.sample_id == "s2"
    assert abs(s2_eval.suspicion_score - 0.75) < 1e-4
    assert abs(s2_eval.trust_score - 0.25) < 1e-4
    assert s2_eval.prediction is True  # 0.75 >= threshold 0.5
    assert abs(s2_eval.trust_score + s2_eval.suspicion_score - 1.0) < 1e-6
    assert s2_eval.dominant_signal == "neighborhood"  # 0.25 * 0.9 = 0.225


def test_manual_weights_and_contributions():
    scorer = DefaultSignalScorer()
    config = TrustGuardConfig(
        enabled_signals=["semantic", "density"],
        scoring_strategy="weighted_fusion",
        weighting_strategy="manual",
        weights={"semantic": 0.8, "density": 0.2},
    )

    sample_ids = ["s1"]
    sr_sem = _make_signal_result("semantic", [0.5], sample_ids)
    sr_den = _make_signal_result("density", [0.1], sample_ids)

    result = scorer.assess([sr_sem, sr_den], config, threshold=0.4)
    s1_eval = result.assessments[0]

    # Suspicion = 0.8*0.5 + 0.2*0.1 = 0.40 + 0.02 = 0.42
    assert abs(s1_eval.suspicion_score - 0.42) < 1e-4
    assert abs(s1_eval.trust_score - 0.58) < 1e-4
    assert s1_eval.prediction is True  # 0.42 >= 0.4
    assert abs(s1_eval.contributions["semantic"] - 0.40) < 1e-4
    assert abs(s1_eval.contributions["density"] - 0.02) < 1e-4
    assert s1_eval.dominant_signal == "semantic"


def test_validation_weight_optimization_zero_test_leakage():
    # Construct 4 validation samples: 2 clean, 2 poisoned
    val_samples = [
        _make_sample("v1", "POS", poison=False, split=Split.VALIDATION),
        _make_sample("v2", "POS", poison=False, split=Split.VALIDATION),
        _make_sample("v3", "POS", poison=True, split=Split.VALIDATION),
        _make_sample("v4", "POS", poison=True, split=Split.VALIDATION),
    ]
    val_ids = [s.sample_id for s in val_samples]

    # Signal 1 (semantic): perfectly separates clean vs poison [0.1, 0.1, 0.9, 0.9]
    # Signal 2 (density): noisy/inverted [0.8, 0.7, 0.2, 0.3]
    sr_good = _make_signal_result("semantic", [0.1, 0.1, 0.9, 0.9], val_ids)
    sr_noisy = _make_signal_result("density", [0.8, 0.7, 0.2, 0.3], val_ids)

    learned_weights = optimize_validation_weights(
        val_signal_results=[sr_good, sr_noisy],
        val_samples=val_samples,
        objective="f1",
    )

    assert "semantic" in learned_weights
    assert "density" in learned_weights
    assert abs(sum(learned_weights.values()) - 1.0) < 1e-3

    # Semantic signal should receive dominant weight because it separates validation ground truth
    assert learned_weights["semantic"] > learned_weights["density"]


def test_threshold_calibration_f1_optimal():
    calibrator = DefaultValidationCalibrator(method="f1_optimal")
    config = TrustGuardConfig(threshold_calibration_method="f1_optimal")

    # 4 clean (0.1, 0.2, 0.3, 0.4), 2 poisoned (0.7, 0.8)
    val_samples = [
        _make_sample("v1", poison=False, split=Split.VALIDATION),
        _make_sample("v2", poison=False, split=Split.VALIDATION),
        _make_sample("v3", poison=False, split=Split.VALIDATION),
        _make_sample("v4", poison=False, split=Split.VALIDATION),
        _make_sample("v5", poison=True, split=Split.VALIDATION),
        _make_sample("v6", poison=True, split=Split.VALIDATION),
    ]
    val_scores = [0.1, 0.2, 0.3, 0.4, 0.7, 0.8]

    threshold = calibrator.calibrate(val_samples, val_scores, config)
    # Optimal threshold separating clean and poisoned should be 0.7
    assert threshold == 0.7


def test_end_to_end_detector_validation_calibration_and_attribution():
    detector = TrustGuardDetector()
    config = TrustGuardConfig(
        enabled_signals=["semantic", "neighborhood", "density"],
        layers=(1,),
        neighborhood_k=2,
        weighting_strategy="learned_validation",
        threshold_calibration_method="f1_optimal",
    )

    # 1. Fit on TRAIN
    train_matrix = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [0.1, 0.9],
    ])
    train_labels = ["POS", "POS", "NEG", "NEG"]
    train_reps = RepresentationResult(
        sample_ids=["tr1", "tr2", "tr3", "tr4"],
        representations=train_matrix,
        layer_representations={1: train_matrix},
        model_name="test_model",
        max_length=64,
    )
    detector.fit(train_reps, config, labels=train_labels)

    # 2. Calibrate on VALIDATION
    val_matrix = np.array([
        [1.0, 0.0],   # clean POS
        [0.9, 0.1],   # clean POS
        [0.0, 1.0],   # poisoned (labelled POS, representation is NEG)
        [0.1, 0.9],   # poisoned (labelled POS, representation is NEG)
    ])
    val_samples = [
        _make_sample("v1", "POS", poison=False, split=Split.VALIDATION),
        _make_sample("v2", "POS", poison=False, split=Split.VALIDATION),
        _make_sample("v3", "POS", poison=True, split=Split.VALIDATION),
        _make_sample("v4", "POS", poison=True, split=Split.VALIDATION),
    ]
    val_reps = RepresentationResult(
        sample_ids=[s.sample_id for s in val_samples],
        representations=val_matrix,
        layer_representations={1: val_matrix},
        model_name="test_model",
        max_length=64,
    )

    learned_w, cal_thresh = detector.calibrate_validation(val_reps, val_samples, config)
    assert learned_w is not None
    assert cal_thresh is not None
    assert detector.learned_weights == learned_w
    assert detector.calibrated_threshold == cal_thresh

    # 3. Detect on TEST
    test_matrix = np.array([
        [1.0, 0.0],   # clean POS
        [0.0, 1.0],   # poisoned
    ])
    test_samples = [
        _make_sample("te1", "POS", poison=False, split=Split.TEST),
        _make_sample("te2", "POS", poison=True, split=Split.TEST),
    ]
    test_reps = RepresentationResult(
        sample_ids=[s.sample_id for s in test_samples],
        representations=test_matrix,
        layer_representations={1: test_matrix},
        model_name="test_model",
        max_length=64,
    )

    detection = detector.detect(test_reps, config, samples=test_samples)
    assert detection.signal_results is not None
    assert "trustguard_score_result" in detection.signal_results

    score_result_dict = detection.signal_results["trustguard_score_result"]
    assert len(score_result_dict["assessments"]) == 2

    clean_assessment = score_result_dict["assessments"][0]
    poison_assessment = score_result_dict["assessments"][1]

    assert clean_assessment["trust_score"] > poison_assessment["trust_score"]
    assert poison_assessment["suspicion_score"] > clean_assessment["suspicion_score"]
    assert clean_assessment["prediction"] is False
    assert poison_assessment["prediction"] is True
    assert "contributions" in clean_assessment
    assert "dominant_signal" in clean_assessment
