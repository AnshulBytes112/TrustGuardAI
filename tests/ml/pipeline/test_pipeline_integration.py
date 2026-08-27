from ml.detectors.flare import FlareDetector
from ml.detectors.schemas import DetectorConfig
from ml.evaluation.calibration import ThresholdCalibrationConfig, ThresholdCalibrator
from ml.evaluation.engine import DetectionEvaluationEngine
from ml.features.config import RepresentationConfig
from ml.features.representations import DistilBERTRepresentationProvider
from ml.features.service import RepresentationService
from ml.pipeline.pipeline import DetectionPipeline
from ml.pipeline.schemas import DetectionPipelineConfig
from ml.poisoning.engine import TextPoisoningEngine


def test_pipeline_end_to_end_integration(
    fully_labelled_samples, standard_poisoning_config, tmp_path
):
    # Setup components
    rep_config = RepresentationConfig(
        model_name="distilbert-base-uncased",
        layers=(1,),
        max_length=32,
        batch_size=4,
        device="cpu",
        cache_dir=str(tmp_path / "cache"),
    )
    rep_provider = DistilBERTRepresentationProvider(rep_config)
    rep_service = RepresentationService(rep_provider, rep_config)

    poisoning_engine = TextPoisoningEngine()
    flare_detector = FlareDetector()
    threshold_calibrator = ThresholdCalibrator()
    evaluation_engine = DetectionEvaluationEngine()

    pipeline = DetectionPipeline(
        representation_service=rep_service,
        flare_detector=flare_detector,
        threshold_calibrator=threshold_calibrator,
        evaluation_engine=evaluation_engine,
        poisoning_engine=poisoning_engine,
    )

    # Multiply samples to ensure we have enough for statistical distribution
    import copy
    large_dataset = []
    for i in range(5):
        for s in fully_labelled_samples:
            new_s = copy.deepcopy(s)
            new_s.sample_id = f"{s.sample_id}_{i}"
            large_dataset.append(new_s)
            
    # Distribute samples across splits (100 samples total)
    # 0-49: TRAIN (50%)
    # 50-74: VALIDATION (25%)
    # 75-99: TEST (25%)
    from ml.data.schemas import Split

    for i, s in enumerate(large_dataset):
        if i < 50:
            s.split = Split.TRAIN
        elif i < 75:
            s.split = Split.VALIDATION
        else:
            s.split = Split.TEST

    config = DetectionPipelineConfig(
        representation_config=rep_config,
        detector_config=DetectorConfig(layers=(1,), threshold=0.0),
        calibration_config=ThresholdCalibrationConfig(),
        poisoning_config=standard_poisoning_config,
    )

    result = pipeline.run(large_dataset, config)

    # Basic validations
    assert result.dataset_id == large_dataset[0].dataset_id
    assert "poisoned" in result.dataset_version
    assert result.poisoning_metadata is not None
    assert result.poisoning_metadata.attack_type == "text_backdoor_v1"

    assert result.evaluation_report is not None
    assert result.evaluation_report.total_evaluated == 25  # TEST split size

    # Verify representations were cached
    assert (tmp_path / "cache").exists()
    assert list((tmp_path / "cache").glob("*.npz"))

    # Verify deterministic fingerprint
    assert result.pipeline_fingerprint
