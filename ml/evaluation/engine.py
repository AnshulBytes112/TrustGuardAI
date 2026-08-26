from collections.abc import Sequence

from sklearn.metrics import roc_auc_score

from ml.data.schemas import Sample
from ml.detectors.schemas import DetectionResult
from ml.evaluation.schemas import EvaluationReport
from ml.interfaces import EvaluationEngine


class DetectionEvaluationEngine(EvaluationEngine):
    """
    Evaluates detector output against ground truth poisoning labels.
    """

    def evaluate(
        self, samples: Sequence[Sample], detection: DetectionResult
    ) -> EvaluationReport:
        # 1. Validation & ID Matching
        sample_map = {}
        for s in samples:
            if s.sample_id in sample_map:
                raise ValueError(f"Duplicate sample ID in ground truth: {s.sample_id}")
            sample_map[s.sample_id] = s

        det_ids = set(detection.sample_ids)
        if len(det_ids) != len(detection.sample_ids):
            raise ValueError("Duplicate sample ID in detection result.")

        # Check for missing/extra IDs in detection relative to valid ground truth
        # Note: We only require alignment for samples that have known ground truth
        # Wait, the prompt says "If a ground-truth sample has no corresponding detection result: raise error"
        # and "If the detector returns a sample ID that does not exist in the evaluation ground truth: handle this explicitly (validation error)"
        
        gt_ids = set(sample_map.keys())
        missing_ids = gt_ids - det_ids
        if missing_ids:
            raise ValueError(f"Ground truth contains IDs missing from detection result: {missing_ids}")
            
        extra_ids = det_ids - gt_ids
        if extra_ids:
            raise ValueError(f"Detection result contains IDs missing from ground truth: {extra_ids}")

        # 2. Filter valid ground truth & Align arrays
        y_true = []
        y_score = []
        y_pred = []
        
        # We must align explicitly by sample_id, not just zip()
        for s_id, score, is_anom in zip(
            detection.sample_ids, detection.scores, detection.is_anomalous
        ):
            gt_status = sample_map[s_id].poison_ground_truth
            if gt_status is None:
                # Exclude unknown ground truth from metrics
                continue
                
            y_true.append(gt_status)
            y_score.append(score)
            y_pred.append(is_anom)
            
        total_evaluated = len(y_true)
        if total_evaluated == 0:
            raise ValueError("No samples with known poison ground truth to evaluate.")

        # 3. Compute Confusion Matrix
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and not yp)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt and not yp)

        actual_poisoned = tp + fn
        actual_clean = tn + fp
        
        # 4. Threshold-dependent Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall: float | None = tp / (tp + fn) if (tp + fn) > 0 else None
        
        f1 = 0.0
        if precision > 0 and recall is not None and recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            
        accuracy = (tp + tn) / total_evaluated
        
        fpr: float | None = fp / (fp + tn) if (fp + tn) > 0 else None
        fnr: float | None = fn / (fn + tp) if (fn + tp) > 0 else None
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else None
        tnr = tn / (tn + fp) if (tn + fp) > 0 else None
        
        balanced_accuracy: float | None = None
        if tpr is not None and tnr is not None:
            balanced_accuracy = (tpr + tnr) / 2

        # 5. Score-based Metrics (AUROC)
        auroc: float | None = None
        if actual_poisoned > 0 and actual_clean > 0:
            auroc = roc_auc_score(y_true, y_score)
            
        return EvaluationReport(
            total_evaluated=total_evaluated,
            poisoned_samples=actual_poisoned,
            clean_samples=actual_clean,
            true_positive=tp,
            false_positive=fp,
            true_negative=tn,
            false_negative=fn,
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
            fpr=fpr,
            fnr=fnr,
            balanced_accuracy=balanced_accuracy,
            auroc=auroc,
            auprc=None,  # Not explicitly required by spec, skipped for simplicity
            detector_name=detection.detector_name
        )
