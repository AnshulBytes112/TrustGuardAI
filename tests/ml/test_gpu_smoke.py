import numpy as np
import pytest
import torch

from ml.data.schemas import LabelStatus, Sample, Split
from ml.detectors.onion import OnionDetector
from ml.detectors.schemas import OnionDetectorConfig
from ml.features.config import RepresentationConfig
from ml.features.representations import DistilBERTRepresentationProvider
from ml.models.classifier import TrainableDownstreamClassifier
from ml.models.schemas import TrainingConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on this host")
def test_cuda_device_detection():
    """Verify that CUDA is recognized and NVIDIA GPU is accessible."""
    assert torch.cuda.is_available(), "CUDA is not available in PyTorch"
    assert torch.cuda.device_count() >= 1
    device_name = torch.cuda.get_device_name(0)
    assert "NVIDIA" in device_name or "GeForce" in device_name or "RTX" in device_name
    print(f"CUDA Smoke Test: Detected {device_name} (VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB)")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on this host")
def test_distilbert_gpu_vs_cpu_numerical_equivalence():
    """
    Verify numerical equivalence of DistilBERT representations between CPU and GPU.
    Max absolute difference must be < 1e-4.
    """
    samples = [
        Sample(
            sample_id=f"smoke_{i}",
            text=f"A wonderful and masterfully crafted performance with true brilliance {i}.",
            label="POSITIVE",
            label_status=LabelStatus.KNOWN,
            split=Split.TRAIN,
            dataset_id="smoke",
            dataset_version="v1",
        )
        for i in range(10)
    ]

    cfg_cpu = RepresentationConfig(
        model_name="distilbert-base-uncased",
        max_length=32,
        batch_size=4,
        device="cpu",
        layers=(1, 3, 6),
    )
    provider_cpu = DistilBERTRepresentationProvider(cfg_cpu)
    reps_cpu = provider_cpu.extract(samples)

    cfg_gpu = RepresentationConfig(
        model_name="distilbert-base-uncased",
        max_length=32,
        batch_size=4,
        device="cuda",
        layers=(1, 3, 6),
    )
    provider_gpu = DistilBERTRepresentationProvider(cfg_gpu)
    reps_gpu = provider_gpu.extract(samples)

    # 1. Shapes must be identical
    assert reps_cpu.representations.shape == reps_gpu.representations.shape
    for layer_idx in (1, 3, 6):
        assert reps_cpu.layer_representations[layer_idx].shape == reps_gpu.layer_representations[layer_idx].shape

    # 2. Numerical equivalence
    max_diff_main = np.max(np.abs(reps_cpu.representations - reps_gpu.representations))
    mean_diff_main = np.mean(np.abs(reps_cpu.representations - reps_gpu.representations))

    print(f"DistilBERT CPU vs GPU Max Diff: {max_diff_main:.6e}, Mean Diff: {mean_diff_main:.6e}")
    assert max_diff_main < 1e-4, f"CPU vs GPU representation drift too high: {max_diff_main}"


def test_downstream_classifier_gpu_execution():
    """Verify that downstream neural classifier trains and predicts on GPU cleanly."""
    X = np.random.RandomState(42).randn(40, 768).astype(np.float32)
    y = ["POSITIVE" if i % 2 == 0 else "NEGATIVE" for i in range(40)]

    cfg = TrainingConfig(
        model_type="neural_head",
        epochs=5,
        learning_rate=0.01,
        batch_size=16,
        seed=42,
        device="auto",
    )
    clf = TrainableDownstreamClassifier(cfg)
    clf.fit(X, y)

    preds = clf.predict(X[:10])
    probs = clf.predict_proba(X[:10])

    assert len(preds) == 10
    assert probs.shape == (10, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)
