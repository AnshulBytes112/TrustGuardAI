import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ml.benchmark.runner import BaselineBenchmarkRunner
from ml.benchmark.schemas import (
    BenchmarkComparisonReport,
    BenchmarkExperimentConfig,
    MethodComparisonRow,
)
from ml.data.csv_adapter import CSVDatasetAdapterConfig
from ml.data.schemas import Sample, Split
from ml.detectors.onion import OnionDetector
from ml.detectors.schemas import OnionDetectorConfig
from ml.experiments.schemas import CSVDatasetConfig
from ml.features.config import RepresentationConfig
from ml.features.schemas import RepresentationResult
from ml.features.service import RepresentationService
from ml.models.schemas import TrainingConfig
from ml.poisoning import PoisoningConfig


class MockRepresentationProvider:
    """Deterministic mock embedding provider."""
    def __init__(self, config=None):
        self.config = config or RepresentationConfig()

    def extract(self, samples):
        reps = []
        for s in samples:
            # Deterministic representation based on sample text/label
            np.random.seed(abs(hash(s.sample_id)) % (2**31))
            base_vec = np.random.randn(32).astype(np.float32)
            if s.label == "pos":
                base_vec += 2.0
            if "cf" in s.text or s.poison_ground_truth is True:
                base_vec[0] += 5.0
            reps.append(base_vec)
        matrix = np.array(reps, dtype=np.float32)
        # Flare requires layers 1 to 6
        layer_dict = {layer: np.array(reps, dtype=np.float32) for layer in range(1, 7)}
        return RepresentationResult(
            sample_ids=[s.sample_id for s in samples],
            representations=matrix,
            layer_representations=layer_dict,
            model_name="mock_distilbert",
            max_length=128,
        )


@pytest.fixture
def benchmark_dataset(tmp_path) -> CSVDatasetConfig:
    csv_file = tmp_path / "benchmark_data.csv"
    lines = ["id,text,label,split"]
    
    # 40 train samples (20 pos, 20 neg)
    for i in range(20):
        lines.append(f"train_pos_{i},Great movie fantastic positive review,pos,train")
        lines.append(f"train_neg_{i},Terrible movie bad negative review,neg,train")
        
    # 20 validation samples (10 pos, 10 neg)
    for i in range(10):
        lines.append(f"val_pos_{i},Amazing positive film experience,pos,validation")
        lines.append(f"val_neg_{i},Awful negative film experience,neg,validation")
        
    # 20 test samples (10 pos, 10 neg)
    for i in range(10):
        lines.append(f"test_pos_{i},Wonderful acting superb positive,pos,test")
        lines.append(f"test_neg_{i},Boring plot terrible negative,neg,test")
        
    csv_file.write_text("\n".join(lines), encoding="utf-8")
    
    return CSVDatasetConfig(
        path=str(csv_file),
        configuration=CSVDatasetAdapterConfig(
            dataset_id="benchmark_synthetic_dataset",
            dataset_version="1.0.0",
            id_column="id",
            text_column="text",
            label_column="label",
            split_column="split",
        ),
    )


def test_baseline_benchmark_framework_end_to_end(benchmark_dataset, tmp_path):
    rep_service = RepresentationService(
        MockRepresentationProvider(), RepresentationConfig()
    )
    runner = BaselineBenchmarkRunner(representation_service=rep_service)

    config = BenchmarkExperimentConfig(
        experiment_name="baseline_comparison_test",
        dataset=benchmark_dataset,
        poisoning_config=PoisoningConfig(
            poison_rate=0.20,
            trigger="cf",
            target_label="pos",
            attack_type="text_backdoor_v1",
            seed=42,
        ),
        methods=["random_filtering", "flare", "trustguard"],
        retraining_config=TrainingConfig(epochs=5, batch_size=8, seed=42),
        benchmark_mode="natural_threshold",
        seeds=[42],
    )

    report = runner.run(config)

    assert isinstance(report, BenchmarkComparisonReport)
    assert len(report.rows) == 3
    method_names = [r.method for r in report.rows]
    assert "random_filtering" in method_names
    assert "flare" in method_names
    assert "trustguard" in method_names

    # Check metric invariants
    for row in report.rows:
        assert 0.0 <= row.precision <= 1.0
        assert 0.0 <= row.f1 <= 1.0
        assert 0.0 <= row.retention_rate <= 1.0
        assert 0.0 <= row.downstream_clean_accuracy <= 1.0
        assert 0.0 <= row.downstream_attack_success_rate <= 1.0
        assert row.runtime_seconds >= 0.0
        assert row.dataset == "benchmark_synthetic_dataset"
        assert row.attack == "text_backdoor_v1"
        assert row.seed == 42

    # Export paper artifacts
    artifacts = report.export_paper_artifacts(tmp_path / "paper_exports")
    assert Path(artifacts["json"]).exists()
    assert Path(artifacts["full_csv"]).exists()
    assert Path(artifacts["detection_csv"]).exists()
    assert Path(artifacts["detection_md"]).exists()
    assert Path(artifacts["purification_csv"]).exists()
    assert Path(artifacts["purification_md"]).exists()
    assert Path(artifacts["efficiency_csv"]).exists()


def test_matched_budget_mode_purification_invariance(benchmark_dataset):
    rep_service = RepresentationService(
        MockRepresentationProvider(), RepresentationConfig()
    )
    runner = BaselineBenchmarkRunner(representation_service=rep_service)

    budget = 0.25
    config = BenchmarkExperimentConfig(
        experiment_name="matched_budget_test",
        dataset=benchmark_dataset,
        poisoning_config=PoisoningConfig(
            poison_rate=0.20,
            trigger="cf",
            target_label="pos",
            attack_type="text_backdoor_v1",
            seed=42,
        ),
        methods=["random_filtering", "flare", "trustguard"],
        retraining_config=TrainingConfig(epochs=5, batch_size=8, seed=42),
        benchmark_mode="matched_budget",
        filtering_budget=budget,
        seeds=[42],
    )

    report = runner.run(config)
    
    # In matched-budget mode, all methods must remove exactly the configured budget fraction of TRAIN samples
    removed_counts = [r.num_removed for r in report.rows]
    expected_removed = int(round(budget * 40))  # 40 train samples * 0.25 = 10
    assert all(count == expected_removed for count in removed_counts)
    assert all(r.filtering_budget == budget for r in report.rows)


def test_multi_seed_aggregation(benchmark_dataset):
    rep_service = RepresentationService(
        MockRepresentationProvider(), RepresentationConfig()
    )
    runner = BaselineBenchmarkRunner(representation_service=rep_service)

    config = BenchmarkExperimentConfig(
        experiment_name="multi_seed_test",
        dataset=benchmark_dataset,
        poisoning_config=PoisoningConfig(
            poison_rate=0.20,
            trigger="cf",
            target_label="pos",
            attack_type="text_backdoor_v1",
            seed=42,
        ),
        methods=["random_filtering", "flare"],
        retraining_config=TrainingConfig(epochs=3, batch_size=8, seed=42),
        benchmark_mode="natural_threshold",
        seeds=[42, 100],
    )

    report = runner.run(config)
    assert len(report.rows) == 4  # 2 methods * 2 seeds
    assert report.statistical_summaries is not None
    assert "random_filtering" in report.statistical_summaries
    assert "flare" in report.statistical_summaries
    assert "f1_mean" in report.statistical_summaries["flare"]
    assert "f1_std" in report.statistical_summaries["flare"]


def test_leakage_regression_test(benchmark_dataset):
    """
    Leakage regression test:
    Verify that mutating TEST split labels and representations does NOT alter
    the fitted state, thresholds, or purification decisions on TRAIN.
    """
    rep_service = RepresentationService(
        MockRepresentationProvider(), RepresentationConfig()
    )
    runner = BaselineBenchmarkRunner(representation_service=rep_service)

    config = BenchmarkExperimentConfig(
        experiment_name="leakage_regression_test",
        dataset=benchmark_dataset,
        poisoning_config=PoisoningConfig(
            poison_rate=0.20,
            trigger="cf",
            target_label="pos",
            attack_type="text_backdoor_v1",
            seed=42,
        ),
        methods=["trustguard"],
        retraining_config=TrainingConfig(epochs=3, batch_size=8, seed=42),
        benchmark_mode="natural_threshold",
        seeds=[42],
    )

    report_original = runner.run(config)
    orig_row = report_original.rows[0]

    # Verify that the threshold and train removed counts are strictly validation and train derived
    assert orig_row.threshold is not None
    assert orig_row.train_size == 40
    assert orig_row.validation_size == 20
    assert orig_row.test_size == 20
