from abc import ABC, abstractmethod
from collections.abc import Sequence
import itertools
from typing import Any
import numpy as np

from ml.data.schemas import Sample
from ml.detectors.schemas import DetectionResult
from ml.detectors.trustguard.schemas import (
    SignalResult,
    TrustGuardConfig,
)
from ml.detectors.trustguard.utils import validate_sample_ids_alignment


def optimize_validation_weights(
    val_signal_results: Sequence[SignalResult],
    val_samples: Sequence[Sample],
    objective: str = "f1",
    resolution: int = 10,
) -> dict[str, float]:
    """
    Optimizes signal weights strictly using VALIDATION data to maximize validation performance.
    Subject to constraints: sum(weights) == 1.0 and weights >= 0.
    TEST data is never used.
    """
    if not val_signal_results:
        raise ValueError("Cannot optimize weights with empty signal results.")

    if not val_samples:
        raise ValueError("Cannot optimize weights with empty validation samples.")

    ref_sample_ids = [s.sample_id for s in val_samples]
    items_lists = [sr.items for sr in val_signal_results if sr.items]
    if items_lists:
        validate_sample_ids_alignment(ref_sample_ids, items_lists)

    # Map sample_id to poison_ground_truth
    gt_map = {s.sample_id: s.poison_ground_truth for s in val_samples}
    valid_indices = [
        i for i, sid in enumerate(ref_sample_ids) if gt_map.get(sid) is not None
    ]

    if not valid_indices:
        # Fall back to equal weights if ground truth is missing
        n_signals = len(val_signal_results)
        return {sr.signal_type: round(1.0 / n_signals, 4) for sr in val_signal_results}

    y_true = np.array([bool(gt_map[ref_sample_ids[i]]) for i in valid_indices], dtype=bool)
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        # Validation must contain both clean and poisoned samples
        n_signals = len(val_signal_results)
        return {sr.signal_type: round(1.0 / n_signals, 4) for sr in val_signal_results}

    signal_names = [sr.signal_type for sr in val_signal_results]
    n_signals = len(signal_names)

    if n_signals == 1:
        return {signal_names[0]: 1.0}

    # Extract score matrix: shape (n_signals, n_valid)
    raw_matrix = np.array([
        [sr.scores[i] for i in valid_indices] for sr in val_signal_results
    ], dtype=np.float64)

    # Generate simplex weight grid (sum == resolution, partition into n_signals non-negative parts)
    candidate_weights: list[np.ndarray] = []
    for partition in itertools.product(range(resolution + 1), repeat=n_signals):
        if sum(partition) == resolution:
            w = np.array(partition, dtype=np.float64) / resolution
            candidate_weights.append(w)

    from ml.evaluation.calibration import (
        ThresholdCalibrationConfig,
        ThresholdCalibrator,
    )

    calibrator = ThresholdCalibrator()
    calib_method = "f1_optimal" if objective in ("f1", "f1_optimal") else "youden_j"
    calib_config = ThresholdCalibrationConfig(method=calib_method)

    valid_samples_subset = [val_samples[i] for i in valid_indices]
    valid_sample_ids = [ref_sample_ids[i] for i in valid_indices]

    best_weights: np.ndarray | None = None
    best_objective_val = -float("inf")
    best_margin = -float("inf")

    # Evaluate each candidate weight vector on validation set
    for w in candidate_weights:
        # Fused suspicion score: (n_valid,)
        composite_scores = np.dot(w, raw_matrix)
        det_result = DetectionResult(
            sample_ids=valid_sample_ids,
            scores=composite_scores.tolist(),
            is_anomalous=[False] * len(valid_sample_ids),
            layer_scores={},
            detector_name="trustguard-weight-optimizer",
        )

        res = calibrator.calibrate(valid_samples_subset, det_result, calib_config)

        # Compute separation margin on validation split: mean(poison) - mean(clean)
        pos_scores = composite_scores[y_true]
        neg_scores = composite_scores[~y_true]
        margin = float(np.mean(pos_scores) - np.mean(neg_scores))

        # Tie-breaking logic:
        # 1. Higher primary objective (F1 / Youden's J)
        # 2. Higher separation margin between poisoned and clean samples
        # 3. Preference for simpler / uniform distribution
        is_better = False
        if best_weights is None or res.objective_value > best_objective_val + 1e-6:
            is_better = True
        elif abs(res.objective_value - best_objective_val) <= 1e-6 and best_weights is not None:
            if margin > best_margin + 1e-6:
                is_better = True
            elif abs(margin - best_margin) <= 1e-6:
                uniform_w = np.full(n_signals, 1.0 / n_signals)
                curr_dist = np.linalg.norm(w - uniform_w)
                best_dist = np.linalg.norm(best_weights - uniform_w)
                if curr_dist < best_dist - 1e-6:
                    is_better = True

        if is_better:
            best_weights = w
            best_objective_val = res.objective_value
            best_margin = margin

    if best_weights is None:
        best_weights = np.full(n_signals, 1.0 / n_signals)

    return {
        name: round(float(weight), 4)
        for name, weight in zip(signal_names, best_weights, strict=True)
    }


class ValidationCalibrator(ABC):
    """
    Interface for calibrating TrustGuard decision threshold using validation split data.
    Ensures strict separation between validation calibration and test evaluation.
    """

    @abstractmethod
    def calibrate(
        self,
        validation_samples: Sequence[Sample],
        validation_scores: list[float],
        config: TrustGuardConfig,
    ) -> float:
        """
        Derive optimal decision threshold on validation data.
        """


class DefaultValidationCalibrator(ValidationCalibrator):
    """
    Standard calibrator deriving optimal threshold (Youden's J or F1) on validation split.
    """

    def __init__(self, method: str = "youden_j") -> None:
        self.method = method
        self._calibrator = None

    def calibrate(
        self,
        validation_samples: Sequence[Sample],
        validation_scores: list[float],
        config: TrustGuardConfig,
    ) -> float:
        if not validation_samples:
            raise ValueError("Validation samples cannot be empty.")
        if len(validation_samples) != len(validation_scores):
            raise ValueError("Length mismatch between validation samples and scores.")

        from ml.evaluation.calibration import (
            ThresholdCalibrationConfig,
            ThresholdCalibrator,
        )

        if self._calibrator is None:
            self._calibrator = ThresholdCalibrator()

        method = config.threshold_calibration_method or self.method
        calib_config = ThresholdCalibrationConfig(method=method)

        det_result = DetectionResult(
            sample_ids=[s.sample_id for s in validation_samples],
            scores=validation_scores,
            is_anomalous=[False] * len(validation_samples),
            layer_scores={},
            detector_name="trustguard-calibrator",
        )

        res = self._calibrator.calibrate(validation_samples, det_result, calib_config)
        return res.threshold
