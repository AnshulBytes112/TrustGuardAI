
import os

import pandas as pd
import pytest

from ml.data.csv_adapter import (
    CSVDatasetAdapter,
    CSVDatasetAdapterConfig,
)
from ml.data.schemas import DatasetLabelMode, LabelStatus, Split


@pytest.fixture
def temp_csv_dir(tmpdir):
    return str(tmpdir)

def test_fully_labelled_csv(temp_csv_dir):
    path = os.path.join(temp_csv_dir, "fully_labelled.csv")
    df = pd.DataFrame({
        "text": ["Positive text", "Negative text"],
        "label": ["positive", "negative"]
    })
    df.to_csv(path, index=False)

    config = CSVDatasetAdapterConfig(dataset_id="test_ds", dataset_version="v1")
    adapter = CSVDatasetAdapter(config)
    result = adapter.load(path)

    assert result.total_samples == 2
    assert result.label_mode == DatasetLabelMode.FULLY_LABELLED
    assert result.samples[0].label == "positive"
    assert result.samples[0].label_status == LabelStatus.KNOWN
    assert result.samples[1].label == "negative"
    assert result.samples[1].label_status == LabelStatus.KNOWN

def test_unlabelled_csv(temp_csv_dir):
    path = os.path.join(temp_csv_dir, "unlabelled.csv")
    df = pd.DataFrame({
        "text": ["Just text 1", "Just text 2"]
    })
    df.to_csv(path, index=False)

    config = CSVDatasetAdapterConfig(dataset_id="test_ds", dataset_version="v1")
    adapter = CSVDatasetAdapter(config)
    result = adapter.load(path)

    assert result.total_samples == 2
    assert result.label_mode == DatasetLabelMode.UNLABELLED
    assert result.samples[0].label is None
    assert result.samples[0].label_status == LabelStatus.UNKNOWN

def test_partially_labelled_csv(temp_csv_dir):
    path = os.path.join(temp_csv_dir, "partially_labelled.csv")
    df = pd.DataFrame({
        "text": ["Text 1", "Text 2", "Text 3"],
        "label": ["positive", None, "negative"]
    })
    df.to_csv(path, index=False)

    config = CSVDatasetAdapterConfig(dataset_id="test_ds", dataset_version="v1")
    adapter = CSVDatasetAdapter(config)
    result = adapter.load(path)

    assert result.total_samples == 3
    assert result.label_mode == DatasetLabelMode.PARTIALLY_LABELLED
    assert result.samples[0].label_status == LabelStatus.KNOWN
    assert result.samples[1].label_status == LabelStatus.UNKNOWN
    assert result.samples[2].label_status == LabelStatus.KNOWN

def test_explicit_ids(temp_csv_dir):
    path = os.path.join(temp_csv_dir, "explicit_ids.csv")
    df = pd.DataFrame({
        "id": ["ID1", "ID2"],
        "text": ["Text 1", "Text 2"]
    })
    df.to_csv(path, index=False)

    config = CSVDatasetAdapterConfig(dataset_id="test_ds", dataset_version="v1", id_column="id")
    adapter = CSVDatasetAdapter(config)
    result = adapter.load(path)

    assert result.samples[0].sample_id == "ID1"
    assert result.samples[1].sample_id == "ID2"

def test_deterministic_generated_ids(temp_csv_dir):
    path = os.path.join(temp_csv_dir, "generated_ids.csv")
    df = pd.DataFrame({
        "text": ["Text 1", "Text 2"]
    })
    df.to_csv(path, index=False)

    config = CSVDatasetAdapterConfig(dataset_id="ds_abc", dataset_version="v2")
    adapter = CSVDatasetAdapter(config)
    result = adapter.load(path)

    assert result.samples[0].sample_id == "ds_abc:v2:0"
    assert result.samples[1].sample_id == "ds_abc:v2:1"

def test_duplicate_ids(temp_csv_dir):
    path = os.path.join(temp_csv_dir, "dup_ids.csv")
    df = pd.DataFrame({
        "id": ["ID1", "ID1"],
        "text": ["Text 1", "Text 2"]
    })
    df.to_csv(path, index=False)

    config = CSVDatasetAdapterConfig(dataset_id="test", dataset_version="v1", id_column="id")
    adapter = CSVDatasetAdapter(config)
    with pytest.raises(ValueError, match="Duplicate sample ID found"):
        adapter.load(path)

def test_missing_file():
    config = CSVDatasetAdapterConfig(dataset_id="test", dataset_version="v1")
    adapter = CSVDatasetAdapter(config)
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        adapter.load("non_existent_file.csv")

def test_missing_text_column(temp_csv_dir):
    path = os.path.join(temp_csv_dir, "missing_text.csv")
    df = pd.DataFrame({
        "not_text": ["Text 1"]
    })
    df.to_csv(path, index=False)

    config = CSVDatasetAdapterConfig(dataset_id="test", dataset_version="v1")
    adapter = CSVDatasetAdapter(config)
    with pytest.raises(ValueError, match="Required text column 'text' not found"):
        adapter.load(path)

def test_empty_dataset(temp_csv_dir):
    path = os.path.join(temp_csv_dir, "empty.csv")
    df = pd.DataFrame(columns=["text"])
    df.to_csv(path, index=False)

    config = CSVDatasetAdapterConfig(dataset_id="test", dataset_version="v1")
    adapter = CSVDatasetAdapter(config)
    with pytest.raises(ValueError, match="Empty dataset"):
        adapter.load(path)

def test_empty_text(temp_csv_dir):
    path = os.path.join(temp_csv_dir, "empty_text.csv")
    df = pd.DataFrame({
        "text": ["   ", "valid"],
        "id": ["1", "2"]
    })
    df.to_csv(path, index=False)

    config = CSVDatasetAdapterConfig(dataset_id="test", dataset_version="v1")
    adapter = CSVDatasetAdapter(config)
    with pytest.raises(ValueError, match="Empty text"):
        adapter.load(path)

def test_missing_configured_id_column(temp_csv_dir):
    path = os.path.join(temp_csv_dir, "no_id_col.csv")
    df = pd.DataFrame({
        "text": ["valid"]
    })
    df.to_csv(path, index=False)

    config = CSVDatasetAdapterConfig(dataset_id="test", dataset_version="v1", id_column="my_id")
    adapter = CSVDatasetAdapter(config)
    with pytest.raises(ValueError, match="Configured ID column 'my_id' not found"):
        adapter.load(path)

def test_invalid_split(temp_csv_dir):
    path = os.path.join(temp_csv_dir, "invalid_split.csv")
    df = pd.DataFrame({
        "text": ["valid"],
        "split": ["BAD_SPLIT"]
    })
    df.to_csv(path, index=False)

    config = CSVDatasetAdapterConfig(dataset_id="test", dataset_version="v1", split_column="split")
    adapter = CSVDatasetAdapter(config)
    with pytest.raises(ValueError, match="Invalid split value"):
        adapter.load(path)

def test_split_handling(temp_csv_dir):
    path = os.path.join(temp_csv_dir, "splits.csv")
    df = pd.DataFrame({
        "text": ["1", "2", "3", "4"],
        "split": ["train", "VALIDATION", "Test", None]
    })
    df.to_csv(path, index=False)

    config = CSVDatasetAdapterConfig(dataset_id="test", dataset_version="v1", split_column="split")
    adapter = CSVDatasetAdapter(config)
    result = adapter.load(path)

    assert result.samples[0].split == Split.TRAIN
    assert result.samples[1].split == Split.VALIDATION
    assert result.samples[2].split == Split.TEST
    assert result.samples[3].split == Split.UNASSIGNED

def test_missing_label_does_not_become_nan(temp_csv_dir):
    path = os.path.join(temp_csv_dir, "missing_label.csv")
    df = pd.DataFrame({
        "text": ["Text"],
        "label": [None]
    })
    df.to_csv(path, index=False)

    config = CSVDatasetAdapterConfig(dataset_id="test", dataset_version="v1")
    adapter = CSVDatasetAdapter(config)
    result = adapter.load(path)

    assert result.samples[0].label is None

def test_serialization_compatibility(temp_csv_dir):
    path = os.path.join(temp_csv_dir, "serialize.csv")
    df = pd.DataFrame({
        "text": ["Test"]
    })
    df.to_csv(path, index=False)

    config = CSVDatasetAdapterConfig(dataset_id="test", dataset_version="v1")
    adapter = CSVDatasetAdapter(config)
    result = adapter.load(path)

    # Validate that the resulting object can be dumped as dict
    sample_dict = result.samples[0].model_dump()
    assert sample_dict["sample_id"] == "test:v1:0"

def test_poison_ground_truth_none(temp_csv_dir):
    path = os.path.join(temp_csv_dir, "poison.csv")
    df = pd.DataFrame({
        "text": ["Test"]
    })
    df.to_csv(path, index=False)

    config = CSVDatasetAdapterConfig(dataset_id="test", dataset_version="v1")
    adapter = CSVDatasetAdapter(config)
    result = adapter.load(path)

    assert result.samples[0].poison_ground_truth is None
