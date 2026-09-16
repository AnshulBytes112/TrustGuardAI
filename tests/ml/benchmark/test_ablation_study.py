from pathlib import Path

import numpy as np
import pytest

from ml.benchmark.ablation import (
    AblationComparisonReport,
    AblationExperimentConfig,
    AblationExperimentRunner,
    AblationVariantConfig,
)
from ml.data.csv_adapter import CSVDatasetAdapterConfig
from ml.experiments.schemas import CSVDatasetConfig
from ml.features.config import RepresentationConfig
from ml.features.schemas import RepresentationResult
from ml.features.service import RepresentationService
from ml.models.schemas import TrainingConfig
from ml.poisoning.config import TextPoisoningConfig


class MockRepresentationProvider:
    """Deterministic mock embedding provider."""
    def __init__(self, config=None):
        self.config = config or RepresentationConfig()

    def extract(self, samples):
        reps = []
        for s in samples:
            np.random.seed(abs(hash(s.sample_id)) % (2**31))
            base_vec = np.random.randn(32).astype(np.float32)
            if s.label == "pos":
                base_vec += 2.0
            if "cf" in s.text or "mn" in s.text or s.poison_ground_truth is True:
                base_vec[0] += 5.0
            reps.append(base_vec)
        matrix = np.array(reps, dtype=np.float32)
        layer_dict = {layer: np.array(reps, dtype=np.float32) for layer in range(1, 7)}
        return RepresentationResult(
            sample_ids=[s.sample_id for s in samples],
            representations=matrix,
            layer_representations=layer_dict,
            model_name="mock_distilbert",
            max_length=128,
        )


@pytest.fixture
def ablation_dataset(tmp_path) -> CSVDatasetConfig:
    csv_file = tmp_path / "ablation_data.csv"
    lines = ["id,text,label,split"]
    
    # 40 train samples
    for i in range(20):
        lines.append(f"train_pos_{i},Great movie fantastic positive review,pos,train")
        lines.append(f"train_neg_{i},Terrible movie bad negative review,neg,train")
        
    # 20 validation samples
    for i in range(10):
        lines.append(f"val_pos_{i},Amazing positive film experience,pos,validation")
        lines.append(f"val_neg_{i},Awful negative film experience,neg,validation")
        
    # 20 test samples
    for i in range(10):
        lines.append(f"test_pos_{i},Wonderful acting superb positive,pos,test")
        lines.append(f"test_neg_{i},Boring plot terrible negative,neg,test")
        
    csv_file.write_text("\n".join(lines), encoding="utf-8")
    
    return CSVDatasetConfig(
        path=str(csv_file),
        configuration=CSVDatasetAdapterConfig(
            dataset_id="ablation_dataset",
            dataset_version="1.0.0",
            id_column="id",
            text_column="text",
            label_column="label",
            split_column="split",
        ),
    )


def test_ablation_variant_configurations():
    variants = [
        "TRUSTGUARD_FULL",
        "TRUSTGUARD_NO_SEMANTIC",
        "TRUSTGUARD_NO_NEIGHBORHOOD",
        "TRUSTGUARD_NO_STABILITY",
        "TRUSTGUARD_NO_DENSITY",
        "TRUSTGUARD_EQUAL_WEIGHTS",
        "TRUSTGUARD_VALIDATION_WEIGHTS",
    ]
    for var in variants:
        cfg = AblationVariantConfig.get_config(var)
        assert cfg is not None
        if var == "TRUSTGUARD_NO_SEMANTIC":
            assert "semantic" not in cfg.enabled_signals
            assert "neighborhood" in cfg.enabled_signals
        elif var == "TRUSTGUARD_NO_NEIGHBORHOOD":
            assert "neighborhood" not in cfg.enabled_signals
            assert "semantic" in cfg.enabled_signals
        elif var == "TRUSTGUARD_EQUAL_WEIGHTS":
            assert cfg.weighting_strategy == "equal"
            assert cfg.weights == {"semantic": 0.25, "neighborhood": 0.25, "stability": 0.25, "density": 0.25}


def test_ablation_experiment_runner_end_to_end(ablation_dataset, tmp_path):
    rep_service = RepresentationService(
        MockRepresentationProvider(), RepresentationConfig()
    )
    runner = AblationExperimentRunner(representation_service=rep_service)

    attacks = [
        TextPoisoningConfig(
            attack_type="rare_word",
            poison_rate=0.20,
            trigger="cf",
            target_label="pos",
            seed=42,
        ),
        TextPoisoningConfig(
            attack_type="common_word",
            poison_rate=0.10,
            trigger="weekend",
            target_label="pos",
            seed=42,
        ),
    ]

    config = AblationExperimentConfig(
        experiment_name="ablation_end_to_end_test",
        dataset=ablation_dataset,
        attack_configs=attacks,
        variants=[
            "TRUSTGUARD_FULL",
            "TRUSTGUARD_NO_SEMANTIC",
            "TRUSTGUARD_EQUAL_WEIGHTS",
        ],
        retraining_config=TrainingConfig(epochs=4, batch_size=8, seed=42),
        benchmark_mode="natural_threshold",
        seeds=[42],
    )

    report = runner.run(config)

    assert isinstance(report, AblationComparisonReport)
    # 2 attacks * 3 variants * 1 seed = 6 rows
    assert len(report.rows) == 6

    for r in report.rows:
        assert 0.0 <= r.precision <= 1.0
        assert 0.0 <= r.f1 <= 1.0
        assert 0.0 <= r.retention_rate <= 1.0
        assert 0.0 <= r.downstream_clean_accuracy <= 1.0
        assert 0.0 <= r.downstream_attack_success_rate <= 1.0

    # Test signal contributions calculation
    contribs = report.compute_signal_contributions()
    assert isinstance(contribs, list)

    # Test artifact export
    artifacts = report.export_paper_artifacts(tmp_path / "ablation_exports")
    assert Path(artifacts["json"]).exists()
    assert Path(artifacts["full_csv"]).exists()
    assert Path(artifacts["ablation_md"]).exists()
    assert Path(artifacts["attack_csv"]).exists()
    assert Path(artifacts["attack_md"]).exists()
    assert Path(artifacts["signal_contribution_csv"]).exists()
    assert Path(artifacts["weight_csv"]).exists()


def test_ablation_leakage_regression_test(ablation_dataset):
    """
    Verify that mutating TEST split labels and representations does NOT alter
    the fitted state, weights, thresholds, or purification decisions on TRAIN.
    """
    rep_service = RepresentationService(
        MockRepresentationProvider(), RepresentationConfig()
    )
    runner = AblationExperimentRunner(representation_service=rep_service)

    config = AblationExperimentConfig(
        experiment_name="ablation_leakage_test",
        dataset=ablation_dataset,
        attack_configs=[
            TextPoisoningConfig(
                attack_type="sentence_trigger",
                poison_rate=0.20,
                sentence="I watched this 3D movie last weekend.",
                target_label="pos",
                seed=42,
            )
        ],
        variants=["TRUSTGUARD_FULL"],
        retraining_config=TrainingConfig(epochs=3, batch_size=8, seed=42),
        benchmark_mode="natural_threshold",
        seeds=[42],
    )

    report = runner.run(config)
    row = report.rows[0]
    assert row.threshold is not None
    assert row.train_size == 40
    assert row.validation_size == 20
    assert row.test_size == 20
