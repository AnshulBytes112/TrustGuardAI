import numpy as np
import pytest

from ml.data.schemas import LabelStatus, Sample, Split
from ml.detectors.base import BaseDetector
from ml.detectors.flare import FlareDetector
from ml.detectors.registry import DetectorRegistry
from ml.detectors.schemas import DetectionResult, DetectorConfig
from ml.detectors.trustguard import (
    SignalResult,
    TrustGuardConfig,
    TrustGuardDetector,
)
from ml.evaluation.calibration import ThresholdCalibrationConfig, ThresholdCalibrator
from ml.evaluation.engine import DetectionEvaluationEngine
from ml.features.config import RepresentationConfig
from ml.features.schemas import RepresentationResult
from ml.features.service import RepresentationService
from ml.pipeline.pipeline import DetectionPipeline
from ml.pipeline.schemas import DetectionPipelineConfig
from ml.poisoning.engine import TextPoisoningEngine


class MockRepresentationProvider:
    def __init__(self, config: RepresentationConfig | None = None, d: int = 8):
        self.config = config or RepresentationConfig(model_name="distilbert-base-uncased", layers=(1,))
        self.d = d

    def extract(self, samples: list[Sample], config: RepresentationConfig | None = None) -> RepresentationResult:
        cfg = config or self.config
        n = len(samples)
        sample_ids = [s.sample_id for s in samples]
        layer_reps = {layer: np.ones((n, self.d), dtype=np.float32) for layer in cfg.layers}
        return RepresentationResult(
            sample_ids=sample_ids,
            representations=np.ones((n, self.d), dtype=np.float32),
            layer_representations=layer_reps,
            model_name=cfg.model_name,
            max_length=cfg.max_length,
        )


class TestDetectorRegistry:
    def test_default_registration(self):
        methods = DetectorRegistry.list_methods()
        assert "flare" in methods
        assert "trustguard" in methods

    def test_get_and_create(self):
        flare_cls = DetectorRegistry.get("flare")
        assert flare_cls is FlareDetector
        flare_inst = DetectorRegistry.create("flare")
        assert isinstance(flare_inst, FlareDetector)

        tg_cls = DetectorRegistry.get("trustguard")
        assert tg_cls is TrustGuardDetector
        tg_inst = DetectorRegistry.create("trustguard")
        assert isinstance(tg_inst, TrustGuardDetector)

    def test_invalid_lookup(self):
        with pytest.raises(KeyError, match="not registered"):
            DetectorRegistry.get("non_existent_method")

    def test_invalid_registration(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            DetectorRegistry.register("", FlareDetector)

        class NotADetector:
            pass

        with pytest.raises(TypeError, match="must subclass BaseDetector"):
            DetectorRegistry.register("bad", NotADetector)  # type: ignore

    def test_custom_detector_registration(self):
        class DummyDetector(BaseDetector):
            def fit(self, representations, config):
                pass

            def detect(self, representations, config):
                return DetectionResult(
                    sample_ids=representations.sample_ids,
                    scores=[0.1] * len(representations.sample_ids),
                    is_anomalous=[False] * len(representations.sample_ids),
                    layer_scores={},
                    detector_name="dummy",
                )

        DetectorRegistry.register("custom_dummy", DummyDetector, override=True)
        assert "custom_dummy" in DetectorRegistry.list_methods()
        inst = DetectorRegistry.create("custom_dummy")
        assert isinstance(inst, DummyDetector)


class TestTrustGuardConfig:
    def test_default_config(self):
        config = TrustGuardConfig()
        assert config.enabled_signals == ["semantic", "neighborhood", "stability", "density"]
        assert config.layers == (1, 2, 3, 4, 5, 6)
        assert config.neighborhood_k == 10
        assert config.density_method == "knn_distance"
        assert config.perturbation_count == 5
        assert config.perturbation_strategy == "synonym_swap"
        assert config.scoring_strategy == "weighted_fusion"
        assert config.weighting_strategy == "equal"
        assert config.weights is None
        assert config.threshold is None  # Threshold is optional, not fixed to 0.5
        assert config.seed == 42

    def test_custom_config_variations(self):
        config = TrustGuardConfig(
            enabled_signals=["semantic", "density"],
            layers=(2, 4),
            neighborhood_k=5,
            density_method="local_outlier_factor",
            perturbation_count=3,
            perturbation_strategy="character_noise",
            scoring_strategy="rank_average",
            weighting_strategy="learned_validation",
            threshold_calibration_method="f1_optimal",
            seed=123,
        )
        assert config.neighborhood_k == 5
        assert config.density_method == "local_outlier_factor"
        assert config.perturbation_count == 3
        assert config.perturbation_strategy == "character_noise"
        assert config.scoring_strategy == "rank_average"
        assert config.weighting_strategy == "learned_validation"
        assert config.seed == 123

    def test_signal_result_schema(self):
        sr = SignalResult(
            signal_type="semantic",
            scores=[0.12, 0.45],
            sample_ids=["s1", "s2"],
            metadata={"mean_drift": 0.28},
        )
        assert sr.signal_type == "semantic"
        assert len(sr.scores) == 2


class TestTrustGuardDetectorContract:
    def test_detector_initialization(self):
        detector = TrustGuardDetector()
        assert detector.is_fitted is False
        assert detector.config is None

    def test_detector_fit_validation_and_behavior(self):
        detector = TrustGuardDetector()
        reps = RepresentationResult(
            sample_ids=["s1", "s2"],
            representations=np.array([[0.1, 0.2], [0.3, 0.4]]),
            layer_representations={1: np.array([[0.1, 0.2], [0.3, 0.4]])},
            model_name="test_model",
            max_length=128,
        )
        # Invalid config type
        with pytest.raises(TypeError, match="Expected TrustGuardConfig"):
            detector.fit(reps, DetectorConfig())

        # Valid config fits correctly
        config = TrustGuardConfig(layers=(1,), enabled_signals=["semantic", "density"])
        detector.fit(reps, config, labels=["A", "B"])
        assert detector.is_fitted is True

    def test_detector_detect_validation_and_behavior(self):
        detector = TrustGuardDetector()
        reps = RepresentationResult(
            sample_ids=["s1"],
            representations=np.array([[0.1, 0.2]]),
            layer_representations={1: np.array([[0.1, 0.2]])},
            model_name="test_model",
            max_length=128,
        )
        # Detecting before fit raises error
        config = TrustGuardConfig(layers=(1,))
        with pytest.raises(RuntimeError, match="must be fitted"):
            detector.detect(reps, config)

        with pytest.raises(TypeError, match="Expected TrustGuardConfig"):
            detector.detect(reps, DetectorConfig())


class TestBaselinePreservationAndPipeline:
    def test_flare_baseline_fit_and_detect(self):
        detector = FlareDetector()
        train_reps = RepresentationResult(
            sample_ids=["s1", "s2"],
            representations=np.array([[1.0, 0.0], [0.0, 1.0]]),
            layer_representations={
                1: np.array([[1.0, 0.0], [0.0, 1.0]]),
            },
            model_name="test_model",
            max_length=128,
        )
        config = DetectorConfig(layers=(1,), threshold=0.5)
        detector.fit(train_reps, config)

        test_reps = RepresentationResult(
            sample_ids=["s3"],
            representations=np.array([[0.5, 0.5]]),
            layer_representations={
                1: np.array([[0.5, 0.5]]),
            },
            model_name="test_model",
            max_length=128,
        )
        result = detector.detect(test_reps, config)
        assert len(result.sample_ids) == 1
        assert len(result.scores) == 1
        assert result.detector_name == "flare-centroid-baseline"

    def test_pipeline_fingerprint_deterministic(self):
        rep_conf = RepresentationConfig(model_name="distilbert-base-uncased", layers=(1, 2))
        cal_conf = ThresholdCalibrationConfig()
        flare_det_conf = DetectorConfig(layers=(1, 2), threshold=0.5)

        pipe_conf_1 = DetectionPipelineConfig(
            representation_config=rep_conf,
            method="flare",
            detector_config=flare_det_conf,
            calibration_config=cal_conf,
        )
        pipe_conf_2 = DetectionPipelineConfig(
            representation_config=rep_conf,
            method="flare",
            detector_config=flare_det_conf,
            calibration_config=cal_conf,
        )

        mock_provider = MockRepresentationProvider()
        service = RepresentationService(mock_provider, rep_conf)  # type: ignore
        pipeline = DetectionPipeline(representation_service=service)

        fp1 = pipeline._compute_fingerprint("ds1", "1.0", pipe_conf_1)
        fp2 = pipeline._compute_fingerprint("ds1", "1.0", pipe_conf_2)
        assert fp1 == fp2

        tg_conf = TrustGuardConfig(layers=(1, 2), neighborhood_k=15)
        pipe_conf_tg = DetectionPipelineConfig(
            representation_config=rep_conf,
            method="trustguard",
            trustguard_config=tg_conf,
            calibration_config=cal_conf,
        )
        fp_tg = pipeline._compute_fingerprint("ds1", "1.0", pipe_conf_tg)
        assert fp_tg != fp1

    def test_end_to_end_baseline_pipeline_execution(self):
        samples = [
            Sample(sample_id="tr1", text="clean train 1", label=0, label_status=LabelStatus.KNOWN, split=Split.TRAIN, dataset_id="d1", dataset_version="1.0"),
            Sample(sample_id="tr2", text="clean train 2", label=0, label_status=LabelStatus.KNOWN, split=Split.TRAIN, dataset_id="d1", dataset_version="1.0"),
            Sample(sample_id="v1", text="val clean", label=0, label_status=LabelStatus.KNOWN, poison_ground_truth=False, split=Split.VALIDATION, dataset_id="d1", dataset_version="1.0"),
            Sample(sample_id="v2", text="val poisoned", label=1, label_status=LabelStatus.KNOWN, poison_ground_truth=True, split=Split.VALIDATION, dataset_id="d1", dataset_version="1.0"),
            Sample(sample_id="te1", text="test clean", label=0, label_status=LabelStatus.KNOWN, poison_ground_truth=False, split=Split.TEST, dataset_id="d1", dataset_version="1.0"),
            Sample(sample_id="te2", text="test poisoned", label=1, label_status=LabelStatus.KNOWN, poison_ground_truth=True, split=Split.TEST, dataset_id="d1", dataset_version="1.0"),
        ]

        rep_conf = RepresentationConfig(model_name="distilbert-base-uncased", layers=(1,))
        mock_provider = MockRepresentationProvider()
        service = RepresentationService(mock_provider, rep_conf)  # type: ignore

        pipeline = DetectionPipeline(
            representation_service=service,
            flare_detector=FlareDetector(),
            threshold_calibrator=ThresholdCalibrator(),
            evaluation_engine=DetectionEvaluationEngine(),
            poisoning_engine=TextPoisoningEngine(),
        )

        pipe_conf = DetectionPipelineConfig(
            representation_config=rep_conf,
            method="flare",
            detector_config=DetectorConfig(layers=(1,), threshold=0.5),
            calibration_config=ThresholdCalibrationConfig(method="youden_j"),
        )

        result = pipeline.run(samples, pipe_conf)
        assert result.dataset_id == "d1"
        assert result.method == "flare"
        assert len(result.detection_result.scores) == 2  # 2 test samples
        assert result.evaluation_report is not None
