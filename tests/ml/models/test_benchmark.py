import numpy as np
import pytest

from ml.data.schemas import LabelStatus, Sample, Split
from ml.models.benchmark import DownstreamBenchmarkEvaluator


def test_downstream_benchmark_evaluation():
    rng = np.random.default_rng(42)

    # 10 clean training samples (class A and class B)
    train_samples = [
        Sample(
            sample_id=f"train_{i}",
            text=f"Train text {i}",
            label="POSITIVE" if i < 5 else "NEGATIVE",
            label_status=LabelStatus.KNOWN,
            split=Split.TRAIN,
            dataset_id="ds",
            dataset_version="v1",
            poison_ground_truth=False,
        )
        for i in range(10)
    ]

    # Create distinct separable features
    train_features = np.vstack([
        rng.normal(loc=-2.0, scale=0.5, size=(5, 8)),
        rng.normal(loc=2.0, scale=0.5, size=(5, 8)),
    ])

    # 4 clean test samples + 2 poisoned test samples (injected with trigger towards POSITIVE)
    test_samples = [
        Sample(
            sample_id="test_clean_1",
            text="Clean pos",
            label="POSITIVE",
            label_status=LabelStatus.KNOWN,
            split=Split.TEST,
            dataset_id="ds",
            dataset_version="v1",
            poison_ground_truth=False,
        ),
        Sample(
            sample_id="test_clean_2",
            text="Clean neg",
            label="NEGATIVE",
            label_status=LabelStatus.KNOWN,
            split=Split.TEST,
            dataset_id="ds",
            dataset_version="v1",
            poison_ground_truth=False,
        ),
        Sample(
            sample_id="test_poison_1",
            text="Poisoned trigger text 1",
            label="NEGATIVE",  # original true class
            label_status=LabelStatus.KNOWN,
            split=Split.TEST,
            dataset_id="ds",
            dataset_version="v1",
            poison_ground_truth=True,
        ),
    ]

    test_features = np.vstack([
        rng.normal(loc=-2.0, scale=0.5, size=(1, 8)),  # clean POSITIVE
        rng.normal(loc=2.0, scale=0.5, size=(1, 8)),   # clean NEGATIVE
        rng.normal(loc=-2.0, scale=0.5, size=(1, 8)),  # trigger makes it look POSITIVE
    ])

    evaluator = DownstreamBenchmarkEvaluator(target_label="POSITIVE")
    metrics = evaluator.evaluate_model(train_samples, train_features, test_samples, test_features)

    assert metrics.total_clean_evaluated == 2
    assert metrics.total_poison_evaluated == 1
    assert metrics.clean_accuracy == 1.0
    assert metrics.attack_success_rate == 1.0  # poison sample predicted as POSITIVE


def test_retraining_comparison():
    rng = np.random.default_rng(42)
    evaluator = DownstreamBenchmarkEvaluator(target_label="POSITIVE")

    samples = [
        Sample(
            sample_id=f"s_{i}",
            text=f"Text {i}",
            label="POSITIVE" if i % 2 == 0 else "NEGATIVE",
            label_status=LabelStatus.KNOWN,
            split=Split.TRAIN,
            dataset_id="ds",
            dataset_version="v1",
            poison_ground_truth=False,
        )
        for i in range(8)
    ]

    features = rng.normal(loc=0.0, scale=1.0, size=(8, 4))
    test_samples = samples[:4]
    test_features = features[:4]

    report = evaluator.compare_retraining(
        dataset_id="ds",
        original_version="v1",
        purified_version="v1_clean",
        raw_train_samples=samples,
        raw_train_reps=features,
        purified_train_samples=samples[2:],
        purified_train_reps=features[2:],
        test_samples=test_samples,
        test_reps=test_features,
        quarantined_count=2,
    )

    assert report.dataset_id == "ds"
    assert report.original_version == "v1"
    assert report.purified_version == "v1_clean"
    assert report.quarantined_samples_count == 2
