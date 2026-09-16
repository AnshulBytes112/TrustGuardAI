import json
import uuid
from datetime import UTC, datetime

import numpy as np
from sqlalchemy.orm import Session

from backend.models.dataset import DatasetModel, SampleModel
from backend.models.experiment import ExperimentModel, MetricModel, SampleScoreModel
from ml.data.schemas import LabelStatus, Sample, Split
from ml.detectors.flare import FlareDetector
from ml.detectors.isolation_forest import IsolationForestDetector
from ml.detectors.kmeans import KMeansDetector
from ml.detectors.schemas import DetectorConfig
from ml.evaluation.calibration import (
    ThresholdCalibrationConfig,
    ThresholdCalibrator,
    apply_threshold,
)
from ml.evaluation.engine import DetectionEvaluationEngine
from ml.explainability.generator import ExplanationGenerator
from ml.features.config import RepresentationConfig
from ml.features.representations import DistilBERTRepresentationProvider
from ml.features.service import RepresentationService
from ml.scoring.layer_scores import LayerDecomposer
from ml.scoring.risk_fusion import RiskFusionEngine


class ScanService:
    @staticmethod
    def execute_scan(
        db: Session,
        experiment_id: str,
        dataset_id: str,
        detector_type: str = "FLARE",
        layers: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
        seed: int = 42,
    ) -> ExperimentModel:
        exp = db.query(ExperimentModel).filter(ExperimentModel.id == experiment_id).first()
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found.")

        dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        if not dataset:
            exp.status = "FAILED"
            exp.error_message = f"Dataset {dataset_id} not found."
            db.commit()
            return exp

        sample_models = db.query(SampleModel).filter(SampleModel.dataset_id == dataset_id).all()
        if not sample_models:
            exp.status = "FAILED"
            exp.error_message = "Dataset has no samples."
            db.commit()
            return exp

        try:
            exp.status = "RUNNING"
            db.commit()

            # Convert DB models to ML canonical Samples
            samples: list[Sample] = []
            for sm in sample_models:
                p_gt = True if sm.poison_ground_truth == 1 else (False if sm.poison_ground_truth == 0 else None)
                l_status = LabelStatus.KNOWN if sm.label_status == "KNOWN" else LabelStatus.UNKNOWN
                s_split = Split(sm.split) if sm.split in ["TRAIN", "VALIDATION", "TEST"] else Split.TRAIN
                samples.append(
                    Sample(
                        sample_id=sm.id,
                        text=sm.text,
                        label=sm.label,
                        label_status=l_status,
                        split=s_split,
                        dataset_id=dataset.id,
                        dataset_version=dataset.version,
                        poison_ground_truth=p_gt,
                    )
                )

            # Feature representation extraction
            rep_cfg = RepresentationConfig(model_name="distilbert-base-uncased", layers=layers, max_length=64)
            provider = DistilBERTRepresentationProvider(rep_cfg)
            rep_service = RepresentationService(provider, rep_cfg)

            train_samples = [s for s in samples if s.split == Split.TRAIN]
            val_samples = [s for s in samples if s.split == Split.VALIDATION]
            test_samples = [s for s in samples if s.split == Split.TEST]

            if not train_samples:
                train_samples = samples[: max(1, int(len(samples) * 0.6))]
            if not val_samples:
                val_samples = samples[len(train_samples) : max(len(train_samples) + 1, int(len(samples) * 0.8))]
            if not test_samples:
                test_samples = samples[len(train_samples) + len(val_samples) :]

            # Select Detector
            detector_name = detector_type.upper()
            if detector_name == "ISOLATION_FOREST":
                detector = IsolationForestDetector(n_estimators=50, random_state=seed)
            elif detector_name == "KMEANS":
                detector = KMeansDetector(n_clusters=3, random_state=seed)
            else:
                detector = FlareDetector()

            det_cfg = DetectorConfig(layers=layers, threshold=0.5, aggregation="mean")

            # Extract and Fit
            train_reps = rep_service.extract(train_samples)
            detector.fit(train_reps, det_cfg)

            # Detect on all samples
            all_reps = rep_service.extract(samples)
            detection_raw = detector.detect(all_reps, det_cfg)

            # Calibrate threshold if validation split has clean and poisoned ground truth
            val_poisoned = sum(1 for s in val_samples if s.poison_ground_truth is True)
            val_clean = sum(1 for s in val_samples if s.poison_ground_truth is False)

            if val_poisoned > 0 and val_clean > 0:
                val_reps = rep_service.extract(val_samples)
                val_det = detector.detect(val_reps, det_cfg)
                calibrator = ThresholdCalibrator()
                cal_res = calibrator.calibrate(val_samples, val_det, ThresholdCalibrationConfig(method="youden_j"))
                calibrated_threshold = cal_res.threshold
            else:
                # Default threshold based on median/mean distribution
                calibrated_threshold = float(np.percentile(detection_raw.scores, 85))

            exp.threshold = calibrated_threshold

            # Multi-layer decomposition & Risk fusion & Explainability
            decomposer = LayerDecomposer()
            risk_engine = RiskFusionEngine()
            xai_gen = ExplanationGenerator()

            layer_profiles = decomposer.decompose_batch(detection_raw.sample_ids, detection_raw.layer_scores)

            # Compute clustering and layer scores
            kmeans_baseline = KMeansDetector(n_clusters=3, random_state=seed)
            kmeans_baseline.fit(train_reps, det_cfg)
            k_res = kmeans_baseline.detect(all_reps, det_cfg)

            # Risk fusion
            scores_norm = [float(np.clip(s / (max(detection_raw.scores) + 1e-9), 0.0, 1.0)) for s in detection_raw.scores]
            cluster_norm = [float(np.clip(s / (max(k_res.scores) + 1e-9), 0.0, 1.0)) for s in k_res.scores]
            layer_conc = [float(lp.layer_attributions.get(lp.dominant_layer, 0.0)) for lp in layer_profiles]

            risk_result = risk_engine.fuse_batch(
                sample_ids=detection_raw.sample_ids,
                detector_scores=scores_norm,
                cluster_scores=cluster_norm,
                layer_scores=layer_conc,
            )

            # Save sample scores
            score_models = []
            for sample, risk_item, lp, d_score in zip(samples, risk_result.items, layer_profiles, detection_raw.scores, strict=True):
                explanation = xai_gen.generate_explanation(sample, risk_item, lp)
                evidence_data = {
                    "layer_attributions": {str(k): round(v, 4) for k, v in lp.layer_attributions.items()},
                    "dominant_layer": lp.dominant_layer,
                    "trajectory": lp.trajectory,
                    "token_attributions": [t.model_dump() for t in explanation.token_attributions],
                    "evidence_summary": explanation.evidence_summary,
                }

                score_id = f"sc_{uuid.uuid4().hex[:12]}"
                sm = SampleScoreModel(
                    id=score_id,
                    experiment_id=experiment_id,
                    sample_id=sample.sample_id,
                    detector=detector_type,
                    raw_score=round(float(d_score), 4),
                    normalized_score=round(risk_item.detector_score, 4),
                    risk_score=round(risk_item.risk_score, 4),
                    risk_level=risk_item.risk_level,
                    dominant_evidence=risk_item.dominant_evidence,
                    evidence_json=json.dumps(evidence_data),
                )
                score_models.append(sm)

            db.bulk_save_objects(score_models)

            # Compute evaluation metrics on TEST split if available
            test_indices = [i for i, s in enumerate(samples) if s.split == Split.TEST and s.poison_ground_truth is not None]
            if test_indices:
                test_samples_eval = [samples[i] for i in test_indices]
                # Filter detection result for test only
                test_det_filtered = apply_threshold(detector.detect(rep_service.extract(test_samples_eval), det_cfg), calibrated_threshold)
                eval_engine = DetectionEvaluationEngine()
                eval_report = eval_engine.evaluate(test_samples_eval, test_det_filtered)

                metric_records = [
                    MetricModel(id=f"m_{uuid.uuid4().hex[:8]}", experiment_id=experiment_id, metric_name="precision", metric_value=eval_report.precision),
                    MetricModel(id=f"m_{uuid.uuid4().hex[:8]}", experiment_id=experiment_id, metric_name="recall", metric_value=eval_report.recall),
                    MetricModel(id=f"m_{uuid.uuid4().hex[:8]}", experiment_id=experiment_id, metric_name="f1", metric_value=eval_report.f1),
                    MetricModel(id=f"m_{uuid.uuid4().hex[:8]}", experiment_id=experiment_id, metric_name="accuracy", metric_value=eval_report.accuracy),
                ]
                if eval_report.auroc is not None:
                    metric_records.append(MetricModel(id=f"m_{uuid.uuid4().hex[:8]}", experiment_id=experiment_id, metric_name="auroc", metric_value=eval_report.auroc))
                db.bulk_save_objects(metric_records)

            exp.status = "COMPLETED"
            exp.completed_at = datetime.now(UTC)
            db.commit()
            db.refresh(exp)
            return exp

        except Exception as e:
            exp.status = "FAILED"
            exp.error_message = str(e)
            db.commit()
            raise
