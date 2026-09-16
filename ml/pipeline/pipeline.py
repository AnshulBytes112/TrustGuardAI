import hashlib
import json
from collections.abc import Sequence

from ml.data.schemas import Sample, Split
from ml.detectors.base import BaseDetector
from ml.detectors.registry import DetectorRegistry
from ml.detectors.trustguard.detector import TrustGuardDetector
from ml.evaluation.calibration import ThresholdCalibrator, apply_threshold
from ml.evaluation.engine import DetectionEvaluationEngine
from ml.features.service import RepresentationService
from ml.pipeline.schemas import DetectionPipelineConfig, DetectionPipelineResult
from ml.poisoning.engine import TextPoisoningEngine


class DetectionPipeline:
    """
    End-to-End Orchestrator for anomaly detection research and experiments.
    Executes representation extraction, detector fitting on TRAIN, threshold calibration on VALIDATION,
    and performance evaluation on TEST to strictly prevent data leakage.
    """

    def __init__(
        self,
        representation_service: RepresentationService,
        flare_detector: BaseDetector | None = None,
        threshold_calibrator: ThresholdCalibrator | None = None,
        evaluation_engine: DetectionEvaluationEngine | None = None,
        poisoning_engine: TextPoisoningEngine | None = None,
        detector: BaseDetector | None = None,
    ):
        self.representation_service = representation_service
        self.detector = detector or flare_detector
        self.threshold_calibrator = threshold_calibrator or ThresholdCalibrator()
        self.evaluation_engine = evaluation_engine or DetectionEvaluationEngine()
        self.poisoning_engine = poisoning_engine or TextPoisoningEngine()

    @property
    def flare_detector(self) -> BaseDetector | None:
        """Backward-compatibility accessor."""
        return self.detector

    def _compute_fingerprint(
        self, dataset_id: str, dataset_version: str, config: DetectionPipelineConfig
    ) -> str:
        data = {
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "method": config.method,
            "representation_config": config.representation_config.model_dump(mode="json"),
            "calibration_config": config.calibration_config.model_dump(mode="json"),
        }
        if config.method == "flare" and config.detector_config is not None:
            data["detector_config"] = config.detector_config.model_dump(mode="json")
        elif config.method == "trustguard" and config.trustguard_config is not None:
            data["trustguard_config"] = config.trustguard_config.model_dump(mode="json")
        elif config.detector_config is not None:
            data["detector_config"] = config.detector_config.model_dump(mode="json")

        if config.poisoning_config:
            data["poisoning_config"] = config.poisoning_config.model_dump(mode="json")

        serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _resolve_detector(self, config: DetectionPipelineConfig) -> tuple[BaseDetector, object]:
        method = config.method
        if self.detector is not None:
            detector_instance = self.detector
        else:
            detector_instance = DetectorRegistry.create(method)

        if method == "flare":
            method_config = config.detector_config
        elif method == "trustguard":
            method_config = config.trustguard_config
        else:
            method_config = config.detector_config

        return detector_instance, method_config

    def run(
        self, samples: Sequence[Sample], config: DetectionPipelineConfig
    ) -> DetectionPipelineResult:
        if not samples:
            raise ValueError("Input dataset cannot be empty.")

        # 1. Optional Dataset Poisoning (operates on the entire dataset)
        if config.poisoning_config:
            poisoning_result = self.poisoning_engine.poison(
                samples, config.poisoning_config
            )
            current_samples = poisoning_result.samples
            poisoning_metadata = poisoning_result.metadata
        else:
            current_samples = list(samples)
            poisoning_metadata = None

        final_dataset_id = current_samples[0].dataset_id
        final_dataset_version = current_samples[0].dataset_version

        # 2. Splitting
        train_samples = [s for s in current_samples if s.split == Split.TRAIN]
        val_samples = [s for s in current_samples if s.split == Split.VALIDATION]
        test_samples = [s for s in current_samples if s.split == Split.TEST]

        if not train_samples:
            raise ValueError("Dataset must contain a TRAIN split to fit the detector.")
        if not val_samples:
            raise ValueError(
                "Dataset must contain a VALIDATION split for threshold calibration."
            )
        if not test_samples:
            raise ValueError("Dataset must contain a TEST split for evaluation.")

        detector_instance, method_config = self._resolve_detector(config)

        # 3. Fit Reference State on TRAIN
        train_reps = self.representation_service.extract(train_samples)
        if isinstance(detector_instance, TrustGuardDetector):
            detector_instance.fit(
                train_reps,
                method_config,
                samples=train_samples,
                representation_provider=self.representation_service.provider,
            )
        else:
            detector_instance.fit(train_reps, method_config)

        # 4. Score and Calibrate on VALIDATION
        val_reps = self.representation_service.extract(val_samples)
        if isinstance(detector_instance, TrustGuardDetector):
            val_detection = detector_instance.detect(
                val_reps, method_config, samples=val_samples
            )
        else:
            val_detection = detector_instance.detect(val_reps, method_config)

        # Ensure validation split has binary ground truth classes for calibration
        val_poisoned_gt = sum(1 for s in val_samples if s.poison_ground_truth is True)
        val_clean_gt = sum(1 for s in val_samples if s.poison_ground_truth is False)
        if val_poisoned_gt == 0 or val_clean_gt == 0:
            raise ValueError(
                "Validation split must contain both clean and poisoned samples "
                "for Youden's J calibration."
            )

        calibration_result = self.threshold_calibrator.calibrate(
            val_samples, val_detection, config.calibration_config
        )

        # 5. Score and Evaluate on TEST
        test_reps = self.representation_service.extract(test_samples)
        if isinstance(detector_instance, TrustGuardDetector):
            test_detection = detector_instance.detect(
                test_reps, method_config, samples=test_samples
            )
        else:
            test_detection = detector_instance.detect(test_reps, method_config)

        test_detection_binary = apply_threshold(
            test_detection, calibration_result.threshold
        )

        evaluation_report = self.evaluation_engine.evaluate(
            test_samples, test_detection_binary
        )

        # 6. Assembly
        fingerprint = self._compute_fingerprint(
            final_dataset_id, final_dataset_version, config
        )

        return DetectionPipelineResult(
            dataset_id=final_dataset_id,
            dataset_version=final_dataset_version,
            pipeline_fingerprint=fingerprint,
            poisoning_metadata=poisoning_metadata,
            representation_config=config.representation_config,
            method=config.method,
            detector_config=config.detector_config,
            trustguard_config=config.trustguard_config,
            calibration_config=config.calibration_config,
            threshold=calibration_result.threshold,
            detection_result=test_detection_binary,
            evaluation_report=evaluation_report,
        )
