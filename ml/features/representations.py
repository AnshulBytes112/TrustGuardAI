from collections.abc import Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from ml.data.schemas import Sample
from ml.features.config import RepresentationConfig
from ml.features.schemas import RepresentationResult
from ml.interfaces import RepresentationProvider


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

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self._model = AutoModel.from_pretrained(
            self.config.model_name,
            output_hidden_states=self.config.output_hidden_states
        )
        self._model.to(self._device)
        self._model.eval()

    def extract(self, samples: Sequence[Sample]) -> RepresentationResult:
        if not samples:
            raise ValueError("Input samples list cannot be empty")
        
        self._initialize_model()
        
        sample_ids = []
        texts = []
        for sample in samples:
            sample_ids.append(sample.sample_id)
            texts.append(sample.text)
            
        all_reps = []
        all_layers = {i: [] for i in range(self._model.config.num_hidden_layers + 1)} if self.config.output_hidden_states else None

        # ponytail: Use inference_mode for minimum memory footprint
        with torch.inference_mode():
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]
                
                encoded = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt"
                )
                encoded = {k: v.to(self._device) for k, v in encoded.items()}
                
                outputs = self._model(**encoded)
                
                if self.config.pooling_strategy == "CLS":
                    # DistilBERT CLS is first token of last hidden state
                    batch_reps = outputs.last_hidden_state[:, 0, :]
                elif self.config.pooling_strategy == "mean":
                    mask = encoded["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                    sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
                    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                    batch_reps = sum_embeddings / sum_mask
                
                all_reps.append(batch_reps.cpu().numpy())
                
                if self.config.output_hidden_states:
                    for layer_idx, hidden_state in enumerate(outputs.hidden_states):
                        if self.config.pooling_strategy == "CLS":
                            layer_rep = hidden_state[:, 0, :]
                        elif self.config.pooling_strategy == "mean":
                            mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden_state.size()).float()
                            sum_embeddings = torch.sum(hidden_state * mask, 1)
                            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                            layer_rep = sum_embeddings / sum_mask
                        all_layers[layer_idx].append(layer_rep.cpu().numpy())

        final_reps = np.concatenate(all_reps, axis=0)
        final_layers = None
        if all_layers is not None:
            final_layers = {k: np.concatenate(v, axis=0) for k, v in all_layers.items()}

        return RepresentationResult(
            sample_ids=sample_ids,
            representations=final_reps,
            layer_representations=final_layers,
            model_name=self.config.model_name,
            max_length=self.config.max_length
        )
