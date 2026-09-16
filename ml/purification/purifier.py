from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
import numpy as np

from ml.data.schemas import Sample
from ml.detectors.schemas import DetectionResult
from ml.evaluation.engine import DetectionEvaluationEngine
from ml.purification.schemas import (
    DatasetPurificationResult,
    PurificationConfig,
    PurificationMetrics,
    PurificationPolicy,
    PurifiedSample,
    QuarantineAction,
    QuarantineEvent,
    SampleState,
)


class DatasetPurifier:
    """
    Executes real dataset purification based strictly on TrustGuard detection decisions.
    Separates samples into retained and isolated datasets based on TrustGuard scores and
    calibrated thresholds without inspecting trigger strings or ground-truth labels.
    """

    def __init__(self, evaluation_engine: DetectionEvaluationEngine | None = None) -> None:
        self.evaluation_engine = evaluation_engine or DetectionEvaluationEngine()

    def purify(
        self,
        samples: Sequence[Sample],
        detection: DetectionResult,
        config: PurificationConfig | None = None,
        threshold: float | None = None,
        policy: PurificationPolicy | None = None,
    ) -> DatasetPurificationResult:
        if not samples:
            raise ValueError("Input samples list cannot be empty.")

        cfg = config or PurificationConfig()
        active_policy: PurificationPolicy = policy or cfg.policy
        active_threshold: float = (
            threshold if threshold is not None else cfg.risk_threshold
        )
        experiment_fp: str = cfg.experiment_fingerprint

        # 1. Validation & ID Alignment
        sample_map = {s.sample_id: s for s in samples}
        if len(sample_map) != len(samples):
            raise ValueError("Duplicate sample ID found in input samples.")

        det_ids = set(detection.sample_ids)
        if len(det_ids) != len(detection.sample_ids):
            raise ValueError("Duplicate sample ID in detection result.")

        gt_ids = set(sample_map.keys())
        missing_ids = gt_ids - det_ids
        if missing_ids:
            raise ValueError(f"Input samples contain IDs missing from detection result: {missing_ids}")
        extra_ids = det_ids - gt_ids
        if extra_ids:
            raise ValueError(f"Detection result contains IDs missing from input samples: {extra_ids}")

        original_id = samples[0].dataset_id
        original_ver = samples[0].dataset_version
        purified_ver = f"{original_ver}_{cfg.purified_version_suffix}"

        # Extract per-sample individual signal scores if available
        individual_signals_map: dict[str, dict[str, float]] = {}
        if detection.signal_results and "trustguard_score_result" in detection.signal_results:
            assessments = detection.signal_results["trustguard_score_result"].get("assessments", [])
            for a in assessments:
                individual_signals_map[a["sample_id"]] = a.get("individual_signal_scores", {})
        elif detection.signal_results:
            for sig_name, sig_data in detection.signal_results.items():
                if isinstance(sig_data, dict) and "items" in sig_data:
                    for item in sig_data["items"]:
                        sid = item["sample_id"]
                        val = item.get("normalized_value")
                        if val is not None:
                            individual_signals_map.setdefault(sid, {})[sig_name] = float(val)

        retained_dataset: list[PurifiedSample] = []
        isolated_dataset: list[PurifiedSample] = []
        events: list[QuarantineEvent] = []

        # 2. Purification Decision (strictly based on score and threshold)
        for s_id, score, is_anom in zip(
            detection.sample_ids, detection.scores, detection.is_anomalous, strict=True
        ):
            s = sample_map[s_id]
            suspicion_score = float(max(0.0, min(1.0, score)))
            trust_score = float(round(max(0.0, min(1.0, 1.0 - suspicion_score)), 6))
            is_suspicious = suspicion_score >= active_threshold
            ind_signals = individual_signals_map.get(s_id, {})

            if active_policy == "REMOVE":
                if is_suspicious:
                    # Isolate suspicious sample
                    purified_item = PurifiedSample(
                        sample_id=s.sample_id,
                        text=s.text,
                        label=s.label,
                        label_status=s.label_status,
                        split=s.split,
                        dataset_id=original_id,
                        dataset_version=original_ver,
                        poison_ground_truth=s.poison_ground_truth,
                        original_label=s.original_label,
                        original_label_status=s.original_label_status,
                        suspicion_score=suspicion_score,
                        trust_score=trust_score,
                        individual_signal_scores=ind_signals,
                        prediction=is_suspicious,
                        sample_weight=0.0,
                        purification_action="ISOLATED",
                        experiment_fingerprint=experiment_fp,
                    )
                    isolated_dataset.append(purified_item)

                    event = QuarantineEvent(
                        sample_id=s.sample_id,
                        action=QuarantineAction.QUARANTINE_AUTO_THRESHOLD,
                        previous_state=SampleState.ACTIVE,
                        new_state=SampleState.QUARANTINED,
                        reason=f"TrustGuard suspicion score {suspicion_score:.4f} >= threshold {active_threshold:.4f}",
                        threshold=active_threshold,
                        timestamp=datetime.now(UTC),
                        metadata={
                            "suspicion_score": suspicion_score,
                            "trust_score": trust_score,
                            "individual_signals": ind_signals,
                        },
                    )
                    events.append(event)
                else:
                    # Retain clean sample
                    purified_item = PurifiedSample(
                        sample_id=s.sample_id,
                        text=s.text,
                        label=s.label,
                        label_status=s.label_status,
                        split=s.split,
                        dataset_id=original_id,
                        dataset_version=purified_ver,
                        poison_ground_truth=s.poison_ground_truth,
                        original_label=s.original_label,
                        original_label_status=s.original_label_status,
                        suspicion_score=suspicion_score,
                        trust_score=trust_score,
                        individual_signal_scores=ind_signals,
                        prediction=is_suspicious,
                        sample_weight=1.0,
                        purification_action="RETAINED",
                        experiment_fingerprint=experiment_fp,
                    )
                    retained_dataset.append(purified_item)

            elif active_policy == "REWEIGHT":
                # Continuous weighting policy: all samples kept in retained dataset with sample_weight = trust_score
                weight = trust_score
                action = "REWEIGHTED" if is_suspicious else "RETAINED"

                retained_item = PurifiedSample(
                    sample_id=s.sample_id,
                    text=s.text,
                    label=s.label,
                    label_status=s.label_status,
                    split=s.split,
                    dataset_id=original_id,
                    dataset_version=purified_ver,
                    poison_ground_truth=s.poison_ground_truth,
                    original_label=s.original_label,
                    original_label_status=s.original_label_status,
                    suspicion_score=suspicion_score,
                    trust_score=trust_score,
                    individual_signal_scores=ind_signals,
                    prediction=is_suspicious,
                    sample_weight=weight,
                    purification_action=action,
                    experiment_fingerprint=experiment_fp,
                )
                retained_dataset.append(retained_item)

                if is_suspicious:
                    isolated_item = PurifiedSample(
                        sample_id=s.sample_id,
                        text=s.text,
                        label=s.label,
                        label_status=s.label_status,
                        split=s.split,
                        dataset_id=original_id,
                        dataset_version=original_ver,
                        poison_ground_truth=s.poison_ground_truth,
                        original_label=s.original_label,
                        original_label_status=s.original_label_status,
                        suspicion_score=suspicion_score,
                        trust_score=trust_score,
                        individual_signal_scores=ind_signals,
                        prediction=is_suspicious,
                        sample_weight=weight,
                        purification_action="ISOLATED",
                        experiment_fingerprint=experiment_fp,
                    )
                    isolated_dataset.append(isolated_item)
            else:
                raise ValueError(f"Unsupported purification policy: {active_policy}")

        total_samples = len(samples)
        retained_count = len(retained_dataset)
        isolated_count = len(isolated_dataset)
        retention_rate = round(retained_count / total_samples, 4)
        isolation_rate = round(isolated_count / total_samples, 4)

        # 3. Supervised evaluation against ground truth (strictly post-decision for reporting)
        has_gt = any(s.poison_ground_truth is not None for s in samples)
        if has_gt:
            # Build thresholded detection result matching active threshold
            thresholded_detection = DetectionResult(
                sample_ids=detection.sample_ids,
                scores=detection.scores,
                is_anomalous=[sc >= active_threshold for sc in detection.scores],
                layer_scores=detection.layer_scores,
                detector_name=detection.detector_name,
            )
            report = self.evaluation_engine.evaluate(samples, thresholded_detection)
            metrics = PurificationMetrics(
                total_samples=total_samples,
                retained_samples=retained_count,
                isolated_samples=isolated_count,
                retention_rate=retention_rate,
                isolation_rate=isolation_rate,
                true_positives=report.true_positive,
                false_positives=report.false_positive,
                true_negatives=report.true_negative,
                false_negatives=report.false_negative,
                precision=report.precision,
                recall=report.recall,
                f1_score=report.f1,
                fpr=report.fpr,
                fnr=report.fnr,
                auroc=report.auroc,
                balanced_accuracy=report.balanced_accuracy,
            )
        else:
            metrics = PurificationMetrics(
                total_samples=total_samples,
                retained_samples=retained_count,
                isolated_samples=isolated_count,
                retention_rate=retention_rate,
                isolation_rate=isolation_rate,
            )

        return DatasetPurificationResult(
            original_dataset_id=original_id,
            original_dataset_version=original_ver,
            purified_dataset_version=purified_ver,
            policy=active_policy,
            threshold=active_threshold,
            experiment_fingerprint=experiment_fp,
            retained_dataset=retained_dataset,
            isolated_dataset=isolated_dataset,
            metrics=metrics,
            events=events,
        )
