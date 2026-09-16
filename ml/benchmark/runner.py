import datetime
import hashlib
import logging
import platform
import time
from typing import Any

import numpy as np

from ml.benchmark.schemas import (
    BenchmarkComparisonReport,
    BenchmarkExperimentConfig,
    MethodComparisonRow,
)
from ml.data.csv_adapter import CSVDatasetAdapter
from ml.data.jsonl_adapter import JSONLDatasetAdapter
from ml.data.schemas import Sample, Split
from ml.detectors.onion import OnionDetector
from ml.detectors.random_filtering import RandomFilteringDetector
from ml.detectors.registry import DetectorRegistry
from ml.detectors.trustguard.detector import TrustGuardDetector
from ml.evaluation.calibration import ThresholdCalibrator, apply_threshold
from ml.evaluation.engine import DetectionEvaluationEngine
from ml.experiments.schemas import CSVDatasetConfig, JSONLDatasetConfig
from ml.features.representations import DistilBERTRepresentationProvider
from ml.features.service import RepresentationService
from ml.models.classifier import TrainableDownstreamClassifier
from ml.poisoning.engine import TextPoisoningEngine

logger = logging.getLogger(__name__)


class BaselineBenchmarkRunner:
    """
    Unified, reproducible baseline benchmark execution framework.
    Evaluates Random Filtering, ONION, FLARE, and TrustGuard under identical data, splits, and protocols.
    """

    def __init__(
        self,
        representation_service: RepresentationService | None = None,
        threshold_calibrator: ThresholdCalibrator | None = None,
        evaluation_engine: DetectionEvaluationEngine | None = None,
        poisoning_engine: TextPoisoningEngine | None = None,
    ):
        self.representation_service = representation_service
        self.threshold_calibrator = threshold_calibrator or ThresholdCalibrator()
        self.evaluation_engine = evaluation_engine or DetectionEvaluationEngine()
        self.poisoning_engine = poisoning_engine or TextPoisoningEngine()

    def run(self, config: BenchmarkExperimentConfig) -> BenchmarkComparisonReport:
        logger.info(f"Starting Baseline Benchmark: {config.experiment_name}")

        # 1. Load Dataset once
        if isinstance(config.dataset, CSVDatasetConfig):
            adapter = CSVDatasetAdapter(config.dataset.configuration)
        elif isinstance(config.dataset, JSONLDatasetConfig):
            adapter = JSONLDatasetAdapter(config.dataset.configuration)
        else:
            raise TypeError(f"Unsupported dataset configuration type: {type(config.dataset)}")

        import_result = adapter.load(config.dataset.path)
        base_samples = import_result.samples
        if not base_samples:
            raise ValueError("Input dataset is empty.")

        dataset_id = base_samples[0].dataset_id
        dataset_version = base_samples[0].dataset_version

        # 2. Setup representation provider/service if not injected
        if self.representation_service is None:
            rep_provider = DistilBERTRepresentationProvider(config.representation_config)
            rep_service = RepresentationService(rep_provider, config.representation_config)
        else:
            rep_service = self.representation_service

        rows: list[MethodComparisonRow] = []

        # Iterate across configured seeds
        for current_seed in config.seeds:
            logger.info(f"Executing benchmark for seed {current_seed}...")

            # 3. Apply poisoning ONCE for this seed
            if config.poisoning_config:
                # Override seed in poisoning config if specified
                seed_poisoning_config = config.poisoning_config.model_copy(
                    update={"seed": current_seed}
                )
                poisoning_result = self.poisoning_engine.poison(
                    base_samples, seed_poisoning_config
                )
                current_samples = poisoning_result.samples
                poisoning_metadata = poisoning_result.metadata
                attack_type = poisoning_metadata.attack_type
                poison_rate = poisoning_metadata.poison_rate
            else:
                current_samples = list(base_samples)
                poisoning_metadata = None
                attack_type = "clean"
                poison_rate = 0.0

            # 4. Split ONCE into TRAIN, VALIDATION, TEST
            train_samples = [s for s in current_samples if s.split == Split.TRAIN]
            val_samples = [s for s in current_samples if s.split == Split.VALIDATION]
            test_samples = [s for s in current_samples if s.split == Split.TEST]

            if not train_samples:
                raise ValueError("Dataset must contain a TRAIN split.")
            if not val_samples:
                raise ValueError("Dataset must contain a VALIDATION split.")
            if not test_samples:
                raise ValueError("Dataset must contain a TEST split.")

            # Compute dataset fingerprint
            dataset_fingerprint = self._compute_dataset_fingerprint(current_samples)

            # 5. Extract representations ONCE per split
            train_reps = rep_service.extract(train_samples)
            val_reps = rep_service.extract(val_samples)
            test_reps = rep_service.extract(test_samples)

            train_poison_count = sum(1 for s in train_samples if s.poison_ground_truth is True)
            val_poison_count = sum(1 for s in val_samples if s.poison_ground_truth is True)
            test_poison_count = sum(1 for s in test_samples if s.poison_ground_truth is True)

            # 6. Run each method under identical conditions
            for method_name in config.methods:
                logger.info(f"Running method: {method_name} (seed={current_seed})")
                start_total_time = time.perf_counter()

                # Instantiate detector via registry
                detector = DetectorRegistry.create(method_name)
                detector_cfg = config.detector_configs.get(method_name)

                # Default fallback configs if not explicitly provided
                if detector_cfg is None:
                    if method_name == "flare":
                        from ml.detectors.schemas import DetectorConfig
                        detector_cfg = DetectorConfig()
                    elif method_name == "trustguard":
                        from ml.detectors.trustguard import TrustGuardConfig
                        detector_cfg = TrustGuardConfig(seed=current_seed)
                    elif method_name == "onion":
                        from ml.detectors.schemas import OnionDetectorConfig
                        detector_cfg = OnionDetectorConfig()
                    elif method_name == "random_filtering":
                        from ml.detectors.schemas import RandomFilteringConfig
                        detector_cfg = RandomFilteringConfig(
                            seed=current_seed,
                            filtering_budget=config.filtering_budget,
                        )

                # Set seed on detector config if applicable
                if detector_cfg is not None and hasattr(detector_cfg, "seed"):
                    detector_cfg = detector_cfg.model_copy(update={"seed": current_seed})

                # Step 1: Fit on TRAIN ONLY
                t0_det = time.perf_counter()
                if isinstance(detector, TrustGuardDetector):
                    detector.fit(
                        train_reps,
                        detector_cfg,
                        samples=train_samples,
                        representation_provider=rep_service.provider,
                    )
                elif isinstance(detector, OnionDetector):
                    detector.fit(train_reps, detector_cfg, samples=train_samples)
                elif isinstance(detector, RandomFilteringDetector):
                    detector.fit(train_reps, detector_cfg)
                else:
                    detector.fit(train_reps, detector_cfg)

                # Step 2: Detect & Calibrate on VALIDATION ONLY
                if isinstance(detector, TrustGuardDetector):
                    detector.calibrate_validation(val_reps, val_samples, detector_cfg)
                    val_det = detector.detect(val_reps, detector_cfg, samples=val_samples)
                elif isinstance(detector, OnionDetector):
                    val_det = detector.detect(val_reps, detector_cfg, samples=val_samples)
                elif isinstance(detector, RandomFilteringDetector):
                    val_det = detector.detect(val_reps, detector_cfg)
                else:
                    val_det = detector.detect(val_reps, detector_cfg)

                # Threshold Calibration using VALIDATION ground truth
                if val_poison_count > 0 and (len(val_samples) - val_poison_count) > 0:
                    calib_res = self.threshold_calibrator.calibrate(
                        val_samples, val_det, config.calibration_config
                    )
                    calibrated_threshold = calib_res.threshold
                else:
                    calibrated_threshold = 0.5

                # Step 3: Detect on held-out TEST split
                if isinstance(detector, TrustGuardDetector):
                    test_det = detector.detect(test_reps, detector_cfg, samples=test_samples)
                elif isinstance(detector, OnionDetector):
                    test_det = detector.detect(test_reps, detector_cfg, samples=test_samples)
                elif isinstance(detector, RandomFilteringDetector):
                    test_det = detector.detect(test_reps, detector_cfg)
                else:
                    test_det = detector.detect(test_reps, detector_cfg)

                det_runtime = time.perf_counter() - t0_det

                # Apply validation threshold to test scores
                test_det_binary = apply_threshold(test_det, calibrated_threshold)

                # Evaluate detection performance on TEST
                eval_report = self.evaluation_engine.evaluate(test_samples, test_det_binary)

                # Step 4: TRAIN Purification
                t0_pur = time.perf_counter()
                if isinstance(detector, TrustGuardDetector):
                    train_det = detector.detect(train_reps, detector_cfg, samples=train_samples)
                elif isinstance(detector, OnionDetector):
                    train_det = detector.detect(train_reps, detector_cfg, samples=train_samples)
                elif isinstance(detector, RandomFilteringDetector):
                    train_det = detector.detect(train_reps, detector_cfg)
                else:
                    train_det = detector.detect(train_reps, detector_cfg)

                # Determine which TRAIN samples to remove (Zero ground-truth leakage)
                train_scores = train_det.scores
                n_train = len(train_samples)

                if config.benchmark_mode == "matched_budget":
                    budget = config.filtering_budget
                    k_remove = int(round(budget * n_train))
                    # Sort train indices by score descending
                    sorted_indices = np.argsort(-np.array(train_scores))
                    removed_indices = set(sorted_indices[:k_remove])
                    retained_indices = [i for i in range(n_train) if i not in removed_indices]
                    applied_threshold = (
                        float(train_scores[sorted_indices[k_remove - 1]]) if k_remove > 0 else 1.0
                    )
                else:
                    # Natural threshold mode
                    applied_threshold = calibrated_threshold
                    retained_indices = [
                        i for i, s in enumerate(train_scores) if s < calibrated_threshold
                    ]
                    removed_indices = set(range(n_train)) - set(retained_indices)

                # Ensure at least 2 samples remain for retraining
                if len(retained_indices) < 2:
                    retained_indices = list(range(min(2, n_train)))
                    removed_indices = set(range(n_train)) - set(retained_indices)

                num_removed = len(removed_indices)
                num_flagged = sum(1 for is_anom in test_det_binary.is_anomalous if is_anom)
                retention_rate = len(retained_indices) / float(n_train)
                pur_runtime = time.perf_counter() - t0_pur

                # Step 5: Retrain downstream classifier on purified TRAIN representations
                t0_retrain = time.perf_counter()
                purified_train_reps_matrix = train_reps.representations[retained_indices]
                purified_train_labels = [train_samples[i].label for i in retained_indices]

                classifier = TrainableDownstreamClassifier(
                    config=config.retraining_config.model_copy(update={"seed": current_seed})
                )
                classifier.fit(purified_train_reps_matrix, purified_train_labels)
                retrain_runtime = time.perf_counter() - t0_retrain

                # Step 6: Evaluate Downstream Model on Held-out Clean and Poisoned TEST sets
                clean_test_indices = [
                    i for i, s in enumerate(test_samples) if s.poison_ground_truth is False
                ]
                poison_test_indices = [
                    i for i, s in enumerate(test_samples) if s.poison_ground_truth is True
                ]

                # 6.1 Clean Accuracy (CA)
                if clean_test_indices:
                    clean_reps = test_reps.representations[clean_test_indices]
                    clean_true_labels = [test_samples[i].label for i in clean_test_indices]
                    clean_preds = classifier.predict(clean_reps)
                    clean_acc = sum(
                        1 for yp, yt in zip(clean_preds, clean_true_labels) if yp == yt
                    ) / float(len(clean_true_labels))
                else:
                    clean_acc = 0.0

                # 6.2 Attack Success Rate (ASR)
                target_label = (
                    config.poisoning_config.target_label
                    if config.poisoning_config
                    else "backdoor_target"
                )
                if poison_test_indices:
                    poison_reps = test_reps.representations[poison_test_indices]
                    poison_preds = classifier.predict(poison_reps)
                    asr = sum(
                        1 for yp in poison_preds if yp == target_label
                    ) / float(len(poison_test_indices))
                else:
                    asr = 0.0

                total_runtime = time.perf_counter() - start_total_time

                row = MethodComparisonRow(
                    method=method_name,
                    dataset=dataset_id,
                    dataset_fingerprint=dataset_fingerprint,
                    attack=attack_type,
                    poison_rate=poison_rate,
                    seed=current_seed,
                    train_size=len(train_samples),
                    validation_size=len(val_samples),
                    test_size=len(test_samples),
                    poisoned_train_count=train_poison_count,
                    poisoned_validation_count=val_poison_count,
                    poisoned_test_count=test_poison_count,
                    threshold=round(applied_threshold, 4),
                    filtering_budget=config.filtering_budget if config.benchmark_mode == "matched_budget" else None,
                    num_flagged=num_flagged,
                    num_removed=num_removed,
                    retention_rate=round(retention_rate, 4),
                    precision=round(eval_report.precision, 4),
                    recall=round(eval_report.recall, 4) if eval_report.recall is not None else None,
                    f1=round(eval_report.f1, 4),
                    fpr=round(eval_report.fpr, 4) if eval_report.fpr is not None else None,
                    fnr=round(eval_report.fnr, 4) if eval_report.fnr is not None else None,
                    auroc=round(eval_report.auroc, 4) if eval_report.auroc is not None else None,
                    auprc=round(eval_report.auprc, 4) if eval_report.auprc is not None else None,
                    downstream_clean_accuracy=round(clean_acc, 4),
                    downstream_attack_success_rate=round(asr, 4),
                    runtime_seconds=round(total_runtime, 4),
                    detector_runtime_seconds=round(det_runtime, 4),
                    purification_runtime_seconds=round(pur_runtime, 4),
                    retraining_runtime_seconds=round(retrain_runtime, 4),
                    peak_memory_mb=None,
                )
                rows.append(row)

        # 7. Multi-seed Statistical Summary (if > 1 seed)
        stat_summaries: dict[str, dict[str, float]] | None = None
        if len(config.seeds) > 1:
            stat_summaries = self._compute_statistical_summaries(rows, config.methods)

        # 8. Construct Final Comparison Report
        experiment_fingerprint = config.compute_fingerprint()
        report = BenchmarkComparisonReport(
            experiment_name=config.experiment_name,
            experiment_fingerprint=experiment_fingerprint,
            dataset_fingerprint=dataset_fingerprint,
            configuration_fingerprint=experiment_fingerprint,
            seeds=config.seeds,
            benchmark_mode=config.benchmark_mode,
            software_configuration={
                "os": platform.system(),
                "python": platform.python_version(),
                "processor": platform.processor() or "generic",
            },
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            rows=rows,
            statistical_summaries=stat_summaries,
        )

        logger.info(f"Baseline Benchmark completed: {config.experiment_name}")
        return report

    def _compute_dataset_fingerprint(self, samples: list[Sample]) -> str:
        sample_ids = sorted(s.sample_id for s in samples)
        combined = ";".join(sample_ids)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def _compute_statistical_summaries(
        self, rows: list[MethodComparisonRow], methods: list[str]
    ) -> dict[str, dict[str, float]]:
        summaries: dict[str, dict[str, float]] = {}
        metrics = [
            "precision", "recall", "f1", "auroc", "auprc", "retention_rate",
            "downstream_clean_accuracy", "downstream_attack_success_rate", "runtime_seconds"
        ]

        for method in methods:
            method_rows = [r for r in rows if r.method == method]
            if not method_rows:
                continue

            method_stats: dict[str, float] = {}
            for metric in metrics:
                vals = [
                    getattr(r, metric)
                    for r in method_rows
                    if getattr(r, metric) is not None
                ]
                if vals:
                    method_stats[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
                    method_stats[f"{metric}_std"] = round(float(np.std(vals)), 4)

            summaries[method] = method_stats

        return summaries
