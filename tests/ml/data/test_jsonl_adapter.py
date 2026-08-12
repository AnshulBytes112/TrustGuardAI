import json

import pytest

from ml.data.csv_adapter import CSVDatasetAdapter, CSVDatasetAdapterConfig
from ml.data.jsonl_adapter import JSONLDatasetAdapter, JSONLDatasetAdapterConfig
from ml.data.schemas import DatasetLabelMode, LabelStatus, Split


def create_temp_jsonl(tmp_path, data):
    p = tmp_path / "test.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for item in data:
            if isinstance(item, str):
                f.write(item + "\n")
            else:
                f.write(json.dumps(item) + "\n")
    return str(p)


@pytest.fixture
def base_config():
    return JSONLDatasetAdapterConfig(
        dataset_id="test_ds",
        dataset_version="v1"
    )

# Valid Tests

def test_1_fully_labelled_jsonl(tmp_path, base_config):
    data = [
        {"id": "1", "text": "Excellent", "label": "positive"},
        {"id": "2", "text": "Terrible", "label": "negative"}
    ]
    file_path = create_temp_jsonl(tmp_path, data)
    
    adapter = JSONLDatasetAdapter(base_config)
    result = adapter.load(file_path)
    
    assert result.label_mode == DatasetLabelMode.FULLY_LABELLED
    assert len(result.samples) == 2
    for s in result.samples:
        assert s.label_status == LabelStatus.KNOWN

def test_2_unlabelled_jsonl(tmp_path, base_config):
    data = [
        {"id": "1", "text": "Excellent"},
        {"id": "2", "text": "Terrible"}
    ]
    file_path = create_temp_jsonl(tmp_path, data)
    
    adapter = JSONLDatasetAdapter(base_config)
    result = adapter.load(file_path)
    
    assert result.label_mode == DatasetLabelMode.UNLABELLED
    for s in result.samples:
        assert s.label is None
        assert s.label_status == LabelStatus.UNKNOWN

def test_3_partially_labelled_jsonl(tmp_path, base_config):
    data = [
        {"id": "1", "text": "Excellent", "label": "positive"},
        {"id": "2", "text": "Interesting"},
        {"id": "3", "text": "Terrible", "label": "negative"}
    ]
    file_path = create_temp_jsonl(tmp_path, data)
    
    adapter = JSONLDatasetAdapter(base_config)
    result = adapter.load(file_path)
    
    assert result.label_mode == DatasetLabelMode.PARTIALLY_LABELLED
    assert result.samples[0].label_status == LabelStatus.KNOWN
    assert result.samples[1].label_status == LabelStatus.UNKNOWN
    assert result.samples[2].label_status == LabelStatus.KNOWN

def test_4_custom_field_names(tmp_path):
    data = [
        {"uuid": "1", "review": "Great", "category": "pos"},
        {"uuid": "2", "review": "Bad"}
    ]
    file_path = create_temp_jsonl(tmp_path, data)
    
    config = JSONLDatasetAdapterConfig(
        dataset_id="ds",
        dataset_version="v1",
        text_field="review",
        label_field="category",
        id_field="uuid"
    )
    adapter = JSONLDatasetAdapter(config)
    result = adapter.load(file_path)
    
    assert result.samples[0].sample_id == "1"
    assert result.samples[0].text == "Great"
    assert result.samples[0].label == "pos"

def test_5_explicit_ids(tmp_path):
    data = [{"custom_id": "99", "text": "hello"}]
    file_path = create_temp_jsonl(tmp_path, data)
    
    config = JSONLDatasetAdapterConfig(dataset_id="d", dataset_version="v", id_field="custom_id")
    adapter = JSONLDatasetAdapter(config)
    result = adapter.load(file_path)
    assert result.samples[0].sample_id == "99"

def test_6_deterministic_generated_ids(tmp_path, base_config):
    data = [{"text": "hello"}, {"text": "world"}]
    file_path = create_temp_jsonl(tmp_path, data)
    
    adapter = JSONLDatasetAdapter(base_config)
    result = adapter.load(file_path)
    
    assert result.samples[0].sample_id == "test_ds:v1:0"
    assert result.samples[1].sample_id == "test_ds:v1:1"

def test_7_split_handling(tmp_path):
    data = [{"text": "hello", "split_col": "train"}, {"text": "world", "split_col": "TEST"}]
    file_path = create_temp_jsonl(tmp_path, data)
    
    config = JSONLDatasetAdapterConfig(dataset_id="d", dataset_version="v", split_field="split_col")
    adapter = JSONLDatasetAdapter(config)
    result = adapter.load(file_path)
    
    assert result.samples[0].split == Split.TRAIN
    assert result.samples[1].split == Split.TEST

def test_8_missing_split_unassigned(tmp_path, base_config):
    data = [{"text": "hello"}]
    file_path = create_temp_jsonl(tmp_path, data)
    adapter = JSONLDatasetAdapter(base_config)
    result = adapter.load(file_path)
    assert result.samples[0].split == Split.UNASSIGNED

def test_9_json_null_label(tmp_path, base_config):
    data = [{"text": "hello", "label": None}]
    file_path = create_temp_jsonl(tmp_path, data)
    adapter = JSONLDatasetAdapter(base_config)
    result = adapter.load(file_path)
    assert result.samples[0].label is None
    assert result.samples[0].label_status == LabelStatus.UNKNOWN

def test_10_utf8_text(tmp_path, base_config):
    data = [{"text": "こんにちは"}]
    file_path = create_temp_jsonl(tmp_path, data)
    adapter = JSONLDatasetAdapter(base_config)
    result = adapter.load(file_path)
    assert result.samples[0].text == "こんにちは"

def test_11_dataset_label_mode_inference(tmp_path, base_config):
    # Already tested in 1,2,3, this just explicitly meets the list point
    pass

def test_12_multiple_json_objects(tmp_path, base_config):
    data = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    file_path = create_temp_jsonl(tmp_path, data)
    adapter = JSONLDatasetAdapter(base_config)
    result = adapter.load(file_path)
    assert len(result.samples) == 3

def test_13_normal_ingestion_poison_none(tmp_path, base_config):
    # Pass in poison metadata, should be ignored
    data = [{"text": "a", "poison_ground_truth": True, "poisoned": 1}]
    file_path = create_temp_jsonl(tmp_path, data)
    adapter = JSONLDatasetAdapter(base_config)
    result = adapter.load(file_path)
    assert result.samples[0].poison_ground_truth is None

# Invalid Tests

def test_14_missing_file(base_config):
    adapter = JSONLDatasetAdapter(base_config)
    with pytest.raises(FileNotFoundError):
        adapter.load("does_not_exist.jsonl")

def test_15_empty_file(tmp_path, base_config):
    p = tmp_path / "empty.jsonl"
    p.touch()
    adapter = JSONLDatasetAdapter(base_config)
    with pytest.raises(ValueError, match="Empty dataset file"):
        adapter.load(str(p))

def test_16_malformed_json(tmp_path, base_config):
    data = ["{\"text\": \"valid\"}", "THIS IS NOT JSON", "{\"text\": \"valid\"}"]
    file_path = create_temp_jsonl(tmp_path, data)
    adapter = JSONLDatasetAdapter(base_config)
    with pytest.raises(ValueError, match="Malformed JSON"):
        adapter.load(file_path)

def test_17_non_object_json_line(tmp_path, base_config):
    data = [["text", "label"]]
    file_path = create_temp_jsonl(tmp_path, data)
    adapter = JSONLDatasetAdapter(base_config)
    with pytest.raises(TypeError, match="Non-object JSON found"):
        adapter.load(file_path)

def test_18_missing_configured_text_field(tmp_path, base_config):
    data = [{"wrong_text_field": "hello"}]
    file_path = create_temp_jsonl(tmp_path, data)
    adapter = JSONLDatasetAdapter(base_config)
    with pytest.raises(ValueError, match="Required text field 'text' not found"):
        adapter.load(file_path)

def test_19_null_text(tmp_path, base_config):
    data = [{"text": None}]
    file_path = create_temp_jsonl(tmp_path, data)
    adapter = JSONLDatasetAdapter(base_config)
    with pytest.raises(ValueError, match="Text cannot be null"):
        adapter.load(file_path)

def test_20_empty_text(tmp_path, base_config):
    data = [{"text": ""}]
    file_path = create_temp_jsonl(tmp_path, data)
    adapter = JSONLDatasetAdapter(base_config)
    with pytest.raises(ValueError, match="Text cannot be empty or whitespace only"):
        adapter.load(file_path)

def test_21_whitespace_only_text(tmp_path, base_config):
    data = [{"text": "   "}]
    file_path = create_temp_jsonl(tmp_path, data)
    adapter = JSONLDatasetAdapter(base_config)
    with pytest.raises(ValueError, match="Text cannot be empty or whitespace only"):
        adapter.load(file_path)

def test_22_duplicate_ids(tmp_path):
    data = [{"myid": "1", "text": "a"}, {"myid": "1", "text": "b"}]
    file_path = create_temp_jsonl(tmp_path, data)
    config = JSONLDatasetAdapterConfig(dataset_id="d", dataset_version="v", id_field="myid")
    adapter = JSONLDatasetAdapter(config)
    with pytest.raises(ValueError, match="Duplicate sample ID found"):
        adapter.load(file_path)

def test_23_empty_id(tmp_path):
    data = [{"myid": "", "text": "a"}]
    file_path = create_temp_jsonl(tmp_path, data)
    config = JSONLDatasetAdapterConfig(dataset_id="d", dataset_version="v", id_field="myid")
    adapter = JSONLDatasetAdapter(config)
    with pytest.raises(ValueError, match="Empty sample ID"):
        adapter.load(file_path)

def test_24_invalid_split(tmp_path):
    data = [{"text": "a", "split": "invalid_split"}]
    file_path = create_temp_jsonl(tmp_path, data)
    config = JSONLDatasetAdapterConfig(dataset_id="d", dataset_version="v", split_field="split")
    adapter = JSONLDatasetAdapter(config)
    with pytest.raises(ValueError, match="Invalid split value"):
        adapter.load(file_path)

def test_25_missing_dataset_id():
    # Will fail dataclass initialization if missing
    with pytest.raises(TypeError):
        JSONLDatasetAdapterConfig(dataset_version="v")

def test_26_missing_dataset_version():
    with pytest.raises(TypeError):
        JSONLDatasetAdapterConfig(dataset_id="d")

def test_determinism(tmp_path, base_config):
    data = [{"text": "a"}, {"text": "b"}]
    file_path = create_temp_jsonl(tmp_path, data)
    
    adapter = JSONLDatasetAdapter(base_config)
    res1 = adapter.load(file_path)
    res2 = adapter.load(file_path)
    
    ids_a = [s.sample_id for s in res1.samples]
    ids_b = [s.sample_id for s in res2.samples]
    assert ids_a == ids_b

def test_csv_jsonl_behavioral_parity(tmp_path):
    csv_data = "id,text,label\n1,Great movie,positive\n2,Bad movie,negative"
    csv_path = tmp_path / "test.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(csv_data)
        
    jsonl_data = [
        {"id": "1", "text": "Great movie", "label": "positive"},
        {"id": "2", "text": "Bad movie", "label": "negative"}
    ]
    jsonl_path = create_temp_jsonl(tmp_path, jsonl_data)
    
    csv_config = CSVDatasetAdapterConfig(dataset_id="ds", dataset_version="v1", id_column="id")
    csv_adapter = CSVDatasetAdapter(csv_config)
    csv_res = csv_adapter.load(str(csv_path))
    
    jsonl_config = JSONLDatasetAdapterConfig(dataset_id="ds", dataset_version="v1", id_field="id")
    jsonl_adapter = JSONLDatasetAdapter(jsonl_config)
    jsonl_res = jsonl_adapter.load(str(jsonl_path))
    
    assert len(csv_res.samples) == len(jsonl_res.samples)
    for i in range(len(csv_res.samples)):
        s_csv = csv_res.samples[i]
        s_jsonl = jsonl_res.samples[i]
        assert s_csv.model_dump() == s_jsonl.model_dump()
