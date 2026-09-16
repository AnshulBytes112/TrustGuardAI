import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from ml.data.schemas import Sample
from ml.detectors.schemas import DetectionResult


class ThresholdCalibrationConfig(BaseModel):
    """
    Configuration for threshold calibration.
    """
    method: str = Field(default="youden_j", min_length=1)

    model_config = ConfigDict(frozen=True)


CalibrationConfig = ThresholdCalibrationConfig


class ThresholdCalibrationResult(BaseModel):
    """
    Immutable structured result for threshold calibration.
    """
    threshold: float
    method: str
    objective: str
    objective_value: float
    
    calibration_samples: int = Field(..., ge=0)
    poisoned_samples: int = Field(..., ge=0)
    clean_samples: int = Field(..., ge=0)
    excluded_unknown_samples: int = Field(..., ge=0)
    
    detector_name: str = Field(..., min_length=1)

    model_config = ConfigDict(frozen=True)


def apply_threshold(detection: DetectionResult, threshold: float) -> DetectionResult:
    """
    Applies a binary threshold to a DetectionResult's continuous scores.
    Returns a new DetectionResult instance with updated is_anomalous values.
    """
    new_is_anomalous = [score >= threshold for score in detection.scores]
    return DetectionResult(
        sample_ids=detection.sample_ids,
        scores=detection.scores,
        is_anomalous=new_is_anomalous,
        layer_scores=detection.layer_scores,
        detector_name=detection.detector_name,
        signal_results=detection.signal_results,
    )


class ThresholdCalibrator:
    """
    Offline supervised component that selects a threshold from continuous anomaly scores.
    """

    def calibrate(
        self,
        samples: Sequence[Sample],
        detection: DetectionResult,
        config: ThresholdCalibrationConfig
    ) -> ThresholdCalibrationResult:
        
        # 1. Validation & ID Matching
        sample_map = {}
        for s in samples:
            if s.sample_id in sample_map:
                raise ValueError(f"Duplicate sample ID in ground truth: {s.sample_id}")
            sample_map[s.sample_id] = s

        det_ids = set(detection.sample_ids)
        if len(det_ids) != len(detection.sample_ids):
            raise ValueError("Duplicate sample ID in detection result.")

        gt_ids = set(sample_map.keys())
        missing_ids = gt_ids - det_ids
        if missing_ids:
            raise ValueError(f"Ground truth contains IDs missing from detection result: {missing_ids}")
            
        extra_ids = det_ids - gt_ids
        if extra_ids:
            raise ValueError(f"Detection result contains IDs missing from ground truth: {extra_ids}")

        # 2. Extract Data
        y_true = []
        y_score = []
        excluded_count = 0
        
        for s_id, score in zip(detection.sample_ids, detection.scores):
            gt_status = sample_map[s_id].poison_ground_truth
            if gt_status is None:
                excluded_count += 1
                continue
                
            if math.isnan(score) or math.isinf(score):
                raise ValueError(f"Invalid score encountered: {score} for sample {s_id}")
                
            y_true.append(gt_status)
            y_score.append(score)

        if not y_true:
            raise ValueError("No valid calibration samples found.")

        actual_poisoned = sum(1 for yt in y_true if yt)
        actual_clean = len(y_true) - actual_poisoned

        if actual_poisoned == 0 or actual_clean == 0:
            raise ValueError("Calibration requires both clean and poisoned samples.")

        # 3. Generate deterministic threshold candidates
        unique_scores = sorted(set(y_score))
        candidates = unique_scores

        best_threshold = None
        best_objective_value = -float("inf")
        best_tpr = -float("inf")
        best_fpr = float("inf")
        best_precision = -float("inf")

        is_f1_mode = config.method in ("f1", "f1_optimal")

        # 4. Search via selected calibration method (Youden's J or F1)
        for t in candidates:
            # Binary rule: score >= t
            tp = sum(1 for yt, yp in zip(y_true, y_score) if yt and yp >= t)
            fp = sum(1 for yt, yp in zip(y_true, y_score) if not yt and yp >= t)
            fn = actual_poisoned - tp

            tpr = tp / actual_poisoned
            fpr = fp / actual_clean
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            if is_f1_mode:
                obj_val = (2 * precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0
            else:
                obj_val = tpr - fpr  # Youden's J

            # Tie-breaking logic:
            # 1. Maximize primary objective
            # 2. Prefer higher Recall (TPR)
            # 3. Prefer lower FPR (or higher precision)
            # 4. Numerically lower threshold
            
            is_better = False
            tolerance = 1e-9
            
            if best_threshold is None or obj_val > best_objective_value + tolerance:
                is_better = True
            elif abs(obj_val - best_objective_value) <= tolerance:
                # Tied on primary objective
                if tpr > best_tpr + tolerance:
                    is_better = True
                elif abs(tpr - best_tpr) <= tolerance:
                    if is_f1_mode:
                        if precision > best_precision + tolerance:
                            is_better = True
                        elif abs(precision - best_precision) <= tolerance and t < best_threshold:
                            is_better = True
                    else:
                        if fpr < best_fpr - tolerance:
                            is_better = True
                        elif abs(fpr - best_fpr) <= tolerance and t < best_threshold:
                            is_better = True

            if is_better:
                best_threshold = t
                best_objective_value = obj_val
                best_tpr = tpr
                best_fpr = fpr
                best_precision = precision

        objective_name = (
            "maximize_f1_then_recall_then_precision"
            if is_f1_mode
            else "maximize_youden_j_then_tpr_then_inv_fpr"
        )

        return ThresholdCalibrationResult(
            threshold=best_threshold,
            method=config.method,
            objective=objective_name,
            objective_value=best_objective_value,
            calibration_samples=len(y_true),
            poisoned_samples=actual_poisoned,
            clean_samples=actual_clean,
            excluded_unknown_samples=excluded_count,
            detector_name=detection.detector_name
        )
