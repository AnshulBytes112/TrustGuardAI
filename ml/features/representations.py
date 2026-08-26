from collections.abc import Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from ml.data.schemas import Sample
from ml.features.config import RepresentationConfig
from ml.features.schemas import RepresentationResult
from ml.interfaces import RepresentationProvider


def _pool_hidden_state(
    hidden_state: torch.Tensor, attention_mask: torch.Tensor, pooling_strategy: str
) -> torch.Tensor:
    if pooling_strategy == "CLS":
        # DistilBERT CLS is first token of the hidden state
        return hidden_state[:, 0, :]
    elif pooling_strategy == "mean":
        mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        sum_embeddings = torch.sum(hidden_state * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    else:
        raise ValueError(f"Unsupported pooling strategy: {pooling_strategy}")


class DistilBERTRepresentationProvider(RepresentationProvider):
    def __init__(self, config: RepresentationConfig):
        self.config = config
        self._tokenizer = None
        self._model = None
        self._device = None

    def _initialize_model(self):
        if self._model is not None:
            return

        # ponytail: YAGNI - simple auto device
        if self.config.device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self.config.device)

        output_hidden_states = self.config.output_hidden_states or (
            self.config.layers is not None
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self._model = AutoModel.from_pretrained(
            self.config.model_name, output_hidden_states=output_hidden_states
        )
        self._model.to(self._device)
        self._model.eval()

    def extract(self, samples: Sequence[Sample]) -> RepresentationResult:
        if not samples:
            raise ValueError("Input samples list cannot be empty")

        self._initialize_model()

        max_layers = self._model.config.num_hidden_layers
        if self.config.layers is not None:
            for layer_idx in self.config.layers:
                if layer_idx > max_layers:
                    raise ValueError(
                        f"Requested layer index {layer_idx} out of range for model "
                        f"with {max_layers} hidden layers (valid indices 0 to {max_layers})."
                    )
            target_layers = self.config.layers
        elif self.config.output_hidden_states:
            target_layers = tuple(range(max_layers + 1))
        else:
            target_layers = None

        sample_ids = []
        texts = []
        for sample in samples:
            sample_ids.append(sample.sample_id)
            texts.append(sample.text)

        all_reps = []
        accumulated_layers = (
            {layer_idx: [] for layer_idx in target_layers}
            if target_layers is not None
            else None
        )

        # ponytail: Use inference_mode for minimum memory footprint
        with torch.inference_mode():
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i : i + self.config.batch_size]

                encoded = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self._device) for k, v in encoded.items()}

                outputs = self._model(**encoded)

                batch_reps = _pool_hidden_state(
                    outputs.last_hidden_state,
                    encoded["attention_mask"],
                    self.config.pooling_strategy,
                )
                all_reps.append(batch_reps.cpu().numpy())

                if target_layers is not None and outputs.hidden_states is not None:
                    for layer_idx in target_layers:
                        hidden_state = outputs.hidden_states[layer_idx]
                        layer_rep = _pool_hidden_state(
                            hidden_state,
                            encoded["attention_mask"],
                            self.config.pooling_strategy,
                        )
                        accumulated_layers[layer_idx].append(layer_rep.cpu().numpy())

        final_reps = np.concatenate(all_reps, axis=0)
        final_layers = None
        if accumulated_layers is not None:
            final_layers = {
                layer_idx: np.concatenate(reps, axis=0)
                for layer_idx, reps in accumulated_layers.items()
            }

        return RepresentationResult(
            sample_ids=sample_ids,
            representations=final_reps,
            layer_representations=final_layers,
            model_name=self.config.model_name,
            max_length=self.config.max_length,
        )

