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
        detector_name=detection.detector_name
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

        # 4. Search via Youden's J
        for t in candidates:
            # Binary rule: score >= t
            tp = sum(1 for yt, yp in zip(y_true, y_score) if yt and yp >= t)
            fp = sum(1 for yt, yp in zip(y_true, y_score) if not yt and yp >= t)

            tpr = tp / actual_poisoned
            fpr = fp / actual_clean
            youden_j = tpr - fpr

            # Tie-breaking logic:
            # 1. Maximize Youden's J
            # 2. Prefer higher Recall (TPR)
            # 3. Prefer lower FPR
            # 4. Numerically lower threshold
            
            # Using a very small tolerance to prevent float instability on exact equality
            is_better = False
            tolerance = 1e-9
            
            if best_threshold is None or youden_j > best_objective_value + tolerance:
                is_better = True
            elif abs(youden_j - best_objective_value) <= tolerance:
                # Tied on Youden J
                if tpr > best_tpr + tolerance:
                    is_better = True
                elif abs(tpr - best_tpr) <= tolerance:
                    # Tied on TPR
                    if fpr < best_fpr - tolerance:
                        is_better = True
                    elif abs(fpr - best_fpr) <= tolerance and t < best_threshold:
                        # Tied on FPR, prefer numerically lower threshold
                        is_better = True

            if is_better:
                best_threshold = t
                best_objective_value = youden_j
                best_tpr = tpr
                best_fpr = fpr

        return ThresholdCalibrationResult(
            threshold=best_threshold,
            method=config.method,
            objective="maximize_youden_j_then_tpr_then_inv_fpr",
            objective_value=best_objective_value,
            calibration_samples=len(y_true),
            poisoned_samples=actual_poisoned,
            clean_samples=actual_clean,
            excluded_unknown_samples=excluded_count,
            detector_name=detection.detector_name
        )
