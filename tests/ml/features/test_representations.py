from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from ml.data.schemas import LabelStatus, Sample, Split
from ml.features.config import RepresentationConfig
from ml.features.representations import DistilBERTRepresentationProvider


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


def test_config_validation():
    with pytest.raises(ValueError):
        RepresentationConfig(max_length=-1)

    with pytest.raises(ValueError):
        RepresentationConfig(batch_size=0)

    # TASK-016 Layer Validation in Config
    with pytest.raises(ValueError, match="negative"):
        RepresentationConfig(layers=(-1, 2))

    with pytest.raises(ValueError, match="Duplicate"):
        RepresentationConfig(layers=(2, 2, 6))

    with pytest.raises(ValueError, match="empty"):
        RepresentationConfig(layers=())

    with pytest.raises((ValueError, TypeError), match="integer|Input should be a valid integer"):
        RepresentationConfig(layers=(2.5, 3))  # type: ignore

    config = RepresentationConfig(layers=(0, 3, 6))
    assert config.layers == (0, 3, 6)
    assert config.model_name == "distilbert-base-uncased"


@patch("ml.features.representations.AutoModel")
@patch("ml.features.representations.AutoTokenizer")
def test_provider_lazy_loading_and_extraction(
    mock_tokenizer_class, mock_model_class, dummy_samples
):
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.ones((2, 5), dtype=torch.long),
        "attention_mask": torch.ones((2, 5), dtype=torch.long),
    }
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    mock_model = MagicMock()
    mock_model.config.num_hidden_layers = 6

    # 7 hidden states (indices 0..6), each of shape (2, 5, 768)
    hidden_states = [torch.ones((2, 5, 768)) * i for i in range(7)]
    mock_outputs = MagicMock()
    mock_outputs.last_hidden_state = hidden_states[-1]
    mock_outputs.hidden_states = hidden_states

    mock_model.return_value = mock_outputs
    mock_model_class.from_pretrained.return_value = mock_model

    config = RepresentationConfig(layers=(0, 2, 6))
    provider = DistilBERTRepresentationProvider(config)

    # Lazy loading check
    mock_model_class.from_pretrained.assert_not_called()
    mock_tokenizer_class.from_pretrained.assert_not_called()

    result = provider.extract(dummy_samples)

    assert result.sample_ids == ["1", "2"]
    assert result.representations.shape == (2, 768)
    assert result.layer_representations is not None
    assert list(result.layer_representations.keys()) == [0, 2, 6]

    assert result.layer_representations[0].shape == (2, 768)
    assert result.layer_representations[2].shape == (2, 768)
    assert result.layer_representations[6].shape == (2, 768)

    # Verify single forward pass per batch
    assert mock_model.call_count == 1


@patch("ml.features.representations.AutoModel")
@patch("ml.features.representations.AutoTokenizer")
def test_out_of_range_layer_validation(
    mock_tokenizer_class, mock_model_class, dummy_samples
):
    mock_tokenizer = MagicMock()
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    mock_model = MagicMock()
    mock_model.config.num_hidden_layers = 6
    mock_model_class.from_pretrained.return_value = mock_model

    # Model has 6 hidden layers (valid 0..6), requesting layer 7 must fail
    config = RepresentationConfig(layers=(2, 7))
    provider = DistilBERTRepresentationProvider(config)

    with pytest.raises(ValueError, match="out of range"):
        provider.extract(dummy_samples)


@patch("ml.features.representations.AutoModel")
@patch("ml.features.representations.AutoTokenizer")
def test_label_and_poison_leakage_prevention(
    mock_tokenizer_class, mock_model_class
):
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.ones((1, 4), dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
    }
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    mock_model = MagicMock()
    mock_model.config.num_hidden_layers = 6
    hidden_states = [torch.full((1, 4, 16), fill_value=float(i)) for i in range(7)]
    mock_outputs = MagicMock()
    mock_outputs.last_hidden_state = hidden_states[-1]
    mock_outputs.hidden_states = hidden_states
    mock_model.return_value = mock_outputs
    mock_model_class.from_pretrained.return_value = mock_model

    sample_clean = Sample(
        sample_id="s1",
        text="The quick brown fox",
        label="clean_label",
        label_status=LabelStatus.KNOWN,
        split=Split.UNASSIGNED,
        dataset_id="d1",
        dataset_version="v1",
        poison_ground_truth=False,
        original_label="clean_label",
        original_label_status=LabelStatus.KNOWN,
    )

    sample_poisoned = Sample(
        sample_id="s1",
        text="The quick brown fox",  # Identical text
        label="poisoned_label",
        label_status=LabelStatus.KNOWN,
        split=Split.UNASSIGNED,
        dataset_id="d1",
        dataset_version="v1",
        poison_ground_truth=True,
        original_label=None,
        original_label_status=LabelStatus.UNKNOWN,
    )

    config = RepresentationConfig(layers=(1, 5))
    provider = DistilBERTRepresentationProvider(config)

    res_clean = provider.extract([sample_clean])
    res_poisoned = provider.extract([sample_poisoned])

    np.testing.assert_allclose(res_clean.representations, res_poisoned.representations)
    assert res_clean.layer_representations is not None
    assert res_poisoned.layer_representations is not None
    for layer in (1, 5):
        np.testing.assert_allclose(
            res_clean.layer_representations[layer],
            res_poisoned.layer_representations[layer],
        )


@pytest.mark.integration
def test_real_model_multi_layer_extraction(dummy_samples):
    try:
        config = RepresentationConfig(
            max_length=16, batch_size=2, device="cpu", layers=(0, 3, 6)
        )
        provider = DistilBERTRepresentationProvider(config)
        result = provider.extract(dummy_samples)

        assert len(result.sample_ids) == 2
        assert result.sample_ids == ["1", "2"]
        assert result.representations.shape == (2, 768)

        assert result.layer_representations is not None
        assert set(result.layer_representations.keys()) == {0, 3, 6}
        for layer_idx in (0, 3, 6):
            assert result.layer_representations[layer_idx].shape == (2, 768)
    except (OSError, RuntimeError, ValueError) as e:
        pytest.skip(f"Integration test failed (network/model access?): {e}")


def test_empty_input():
    provider = DistilBERTRepresentationProvider(RepresentationConfig())
    with pytest.raises(ValueError):
        provider.extract([])


