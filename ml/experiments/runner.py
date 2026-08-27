import logging

from ml.data.csv_adapter import CSVDatasetAdapter
from ml.data.jsonl_adapter import JSONLDatasetAdapter
from ml.detectors.flare import FlareDetector
from ml.evaluation.calibration import ThresholdCalibrator
from ml.evaluation.engine import DetectionEvaluationEngine
from ml.experiments.schemas import (
    CSVDatasetConfig,
    ExperimentConfig,
    ExperimentResult,
    JSONLDatasetConfig,
)
from ml.features.representations import DistilBERTRepresentationProvider
from ml.features.service import RepresentationService
from ml.pipeline.pipeline import DetectionPipeline
from ml.poisoning.engine import TextPoisoningEngine

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Executes a complete TrustGuard AI anomaly detection experiment
    from a declarative configuration.
    """

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        logger.info(f"Experiment started: {config.experiment_name}")

        # 1. Load Dataset
        logger.info("Loading dataset...")
        if isinstance(config.dataset, CSVDatasetConfig):
            adapter = CSVDatasetAdapter(config.dataset.configuration)
        elif isinstance(config.dataset, JSONLDatasetConfig):
            adapter = JSONLDatasetAdapter(config.dataset.configuration)
        else:
            raise TypeError(f"Unsupported dataset configuration type: {type(config.dataset)}")

        import_result = adapter.load(config.dataset.path)
        samples = import_result.samples

        if not samples:
            raise ValueError("Dataset is empty.")
            
        logger.info(f"Dataset loaded successfully with {len(samples)} samples.")

        # 2. Validate Dataset Identity
        expected_id = config.dataset.configuration.dataset_id
        expected_version = config.dataset.configuration.dataset_version

        actual_id = samples[0].dataset_id
        actual_version = samples[0].dataset_version

        if actual_id != expected_id or actual_version != expected_version:
            raise ValueError(
                f"Dataset identity mismatch. Expected {expected_id}:{expected_version}, "
                f"but loaded samples have {actual_id}:{actual_version}"
            )

        # 3. Instantiate Pipeline Components
        logger.info("Initializing pipeline components...")
        rep_provider = DistilBERTRepresentationProvider(config.pipeline.representation_config)
        rep_service = RepresentationService(rep_provider, config.pipeline.representation_config)
        
        flare_detector = FlareDetector()
        threshold_calibrator = ThresholdCalibrator()
        evaluation_engine = DetectionEvaluationEngine()
        poisoning_engine = TextPoisoningEngine()
        
        pipeline = DetectionPipeline(
            representation_service=rep_service,
            flare_detector=flare_detector,
            threshold_calibrator=threshold_calibrator,
            evaluation_engine=evaluation_engine,
            poisoning_engine=poisoning_engine
        )

        # 4. Execute DetectionPipeline
        logger.info("Starting pipeline execution...")
        pipeline_result = pipeline.run(samples, config.pipeline)
        logger.info("Pipeline execution completed.")

        # 5. Construct Experiment Result
        experiment_fingerprint = config.compute_fingerprint()

        result = ExperimentResult(
            experiment_name=config.experiment_name,
            experiment_fingerprint=experiment_fingerprint,
            dataset_id=actual_id,
            dataset_version=actual_version,
            pipeline_result=pipeline_result
        )

        logger.info(f"Experiment completed: {config.experiment_name}")
        return result
