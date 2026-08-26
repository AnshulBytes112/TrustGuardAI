import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ml.data.schemas import LabelStatus, Sample, Split
from ml.features.config import RepresentationConfig
from ml.features.schemas import RepresentationResult
from ml.features.service import RepresentationService
from ml.features.store import RepresentationStore


@pytest.fixture
def temp_cache_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def dummy_samples():
    return [
        Sample(
            sample_id="1",
            text="Sample 1",
            label="positive",
            label_status=LabelStatus.KNOWN,
            split=Split.UNASSIGNED,
            dataset_id="d1",
            dataset_version="v1",
        ),
        Sample(
            sample_id="2",
            text="Sample 2",
            label="negative",
            label_status=LabelStatus.KNOWN,
            split=Split.UNASSIGNED,
            dataset_id="d1",
            dataset_version="v1",
        ),
    ]


@pytest.fixture
def dummy_result():
    return RepresentationResult(
        sample_ids=["1", "2"],
        representations=np.random.rand(2, 16),
        layer_representations={
            0: np.random.rand(2, 16),
            3: np.random.rand(2, 16),
        },
        model_name="test-model",
        max_length=16,
    )


def test_deterministic_cache_key(temp_cache_dir, dummy_samples):
    store = RepresentationStore(temp_cache_dir)
    config = RepresentationConfig(model_name="test-model", max_length=16, layers=(0, 3))

    key1 = store.generate_key(dummy_samples, config)
    key2 = store.generate_key(dummy_samples, config)
    
    assert key1 == key2

    # Change operational param -> same key
    config_diff_batch = RepresentationConfig(
        model_name="test-model", max_length=16, layers=(0, 3), batch_size=128
    )
    key3 = store.generate_key(dummy_samples, config_diff_batch)
    assert key1 == key3

    # Change semantic param -> different key
    config_diff_len = RepresentationConfig(
        model_name="test-model", max_length=32, layers=(0, 3)
    )
    key4 = store.generate_key(dummy_samples, config_diff_len)
    assert key1 != key4

    # Change content -> different key
    samples_diff = list(dummy_samples)
    samples_diff[0] = samples_diff[0].model_copy(update={"text": "Different text"})
    key5 = store.generate_key(samples_diff, config)
    assert key1 != key5


def test_cache_save_and_load(temp_cache_dir, dummy_samples, dummy_result):
    store = RepresentationStore(temp_cache_dir)
    config = RepresentationConfig(model_name="test-model", max_length=16, layers=(0, 3))

    key = store.generate_key(dummy_samples, config)
    assert not store.exists(key)

    store.save(dummy_result, key)
    assert store.exists(key)

    loaded = store.load(key, config, dummy_samples)
    assert loaded is not None
    assert loaded.sample_ids == dummy_result.sample_ids
    assert loaded.model_name == dummy_result.model_name
    np.testing.assert_array_equal(loaded.representations, dummy_result.representations)
    
    assert loaded.layer_representations is not None
    assert set(loaded.layer_representations.keys()) == {0, 3}
    np.testing.assert_array_equal(
        loaded.layer_representations[0], dummy_result.layer_representations[0]
    )


def test_cache_miss_on_corruption(temp_cache_dir, dummy_samples, dummy_result):
    store = RepresentationStore(temp_cache_dir)
    config = RepresentationConfig(model_name="test-model", max_length=16, layers=(0, 3))
    key = store.generate_key(dummy_samples, config)
    
    store.save(dummy_result, key)
    
    # Corrupt the file
    path = Path(temp_cache_dir) / f"{key}.npz"
    path.write_bytes(b"corrupted data")
    
    assert store.load(key, config, dummy_samples) is None


def test_cache_miss_on_layer_mismatch(temp_cache_dir, dummy_samples, dummy_result):
    store = RepresentationStore(temp_cache_dir)
    config = RepresentationConfig(model_name="test-model", max_length=16, layers=(0, 3))
    key = store.generate_key(dummy_samples, config)
    store.save(dummy_result, key)

    # Requesting different layers should miss, even if the key somehow matched
    diff_config = RepresentationConfig(model_name="test-model", max_length=16, layers=(0, 4))
    
    assert store.load(key, diff_config, dummy_samples) is None


def test_service_hit_and_miss(temp_cache_dir, dummy_samples, dummy_result):
    config = RepresentationConfig(
        model_name="test-model", max_length=16, layers=(0, 3), cache_dir=temp_cache_dir
    )
    mock_provider = MagicMock()
    mock_provider.extract.return_value = dummy_result

    service = RepresentationService(mock_provider, config)
    
    # First extract -> miss -> save
    res1 = service.extract(dummy_samples)
    assert mock_provider.extract.call_count == 1
    np.testing.assert_array_equal(res1.representations, dummy_result.representations)
    
    # Second extract -> hit -> no new provider call
    res2 = service.extract(dummy_samples)
    assert mock_provider.extract.call_count == 1
    np.testing.assert_array_equal(res2.representations, dummy_result.representations)


def test_service_cache_disabled(temp_cache_dir, dummy_samples, dummy_result):
    config = RepresentationConfig(
        model_name="test-model", max_length=16, layers=(0, 3), cache_dir=temp_cache_dir, use_cache=False
    )
    mock_provider = MagicMock()
    mock_provider.extract.return_value = dummy_result

    service = RepresentationService(mock_provider, config)
    
    # First extract -> calls provider
    service.extract(dummy_samples)
    assert mock_provider.extract.call_count == 1
    
    # Second extract -> calls provider again since cache disabled
    service.extract(dummy_samples)
    assert mock_provider.extract.call_count == 2
