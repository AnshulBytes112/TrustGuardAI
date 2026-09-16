import logging
import math
import re
from typing import Any

import torch

from ml.data.schemas import Sample
from ml.detectors.base import BaseDetector
from ml.detectors.schemas import DetectionResult, OnionDetectorConfig
from ml.features.schemas import RepresentationResult

logger = logging.getLogger(__name__)


class OnionDetector(BaseDetector):
    """
    Faithful implementation of ONION (Observation-based Outlier Normalization for Backdoor Defense).
    
    Reference:
        Qi, F., Chen, Y., Zhang, X., Yao, J., Liu, Z., & Sun, M. (2021).
        "ONION: A Simple and Effective Defense Against Textual Backdoor Attacks."
        Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP 2021).
        
    Methodology:
        ONION detects backdoor trigger tokens by observing sentence perplexity (PPL) changes
        under word removal. For sentence S = (w_1, ..., w_n), inserting a low-frequency or anomalous
        trigger token causes high baseline PPL(S). Removing the trigger token w_i produces S_{\\setminus i}
        with significantly lower perplexity PPL(S_{\\setminus i}), yielding a large positive perplexity drop:
            \\Delta PPL_i = PPL(S) - PPL(S_{\\setminus i})
            
    Benchmark Adaptation:
        The original ONION paper operates primarily as a token-level inference defense (removing words with
        \\Delta PPL_i > \\tau). In this benchmark, ONION is adapted for dataset purification by calculating
        a sample-level anomaly score:
            Score(S) = \\max_{i} (\\Delta PPL_i)
        Samples with high max perplexity drop are identified as containing candidate triggers and isolated
        during training purification.
    """

    def __init__(
        self,
        language_model: Any = None,
        tokenizer: Any = None,
        config: OnionDetectorConfig | None = None,
    ):
        self.config = config or OnionDetectorConfig()
        self._model = language_model
        self._tokenizer = tokenizer
        self._ppl_cache: dict[str, float] = {}
        self.is_fitted = False
        self.reference_sample_count = 0

    def _ensure_model_loaded(self) -> None:
        """
        Lazily initialize causal language model and tokenizer if not injected.
        """
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info(
                f"Loading causal LM '{self.config.language_model_name}' and tokenizer '{self.config.tokenizer_name}' for ONION..."
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(self.config.language_model_name)
            self._model.eval()
            if self.config.device != "cpu" and torch.cuda.is_available():
                self._model.to(self.config.device)
            logger.info("ONION causal LM loaded successfully.")
        except Exception as exc:
            logger.warning(
                f"Failed to load causal LM ({self.config.language_model_name}): {exc}. "
                "Will use fallback perplexity evaluator if mock model is not injected."
            )

    def _compute_sentence_ppl(self, text: str) -> float:
        """
        Compute causal language model perplexity PPL(S) for a given text.
        Returns cached value if already evaluated.
        """
        clean_text = text.strip()
        if not clean_text:
            return 1.0

        if clean_text in self._ppl_cache:
            return self._ppl_cache[clean_text]

        self._ensure_model_loaded()

        if self._model is None or self._tokenizer is None:
            # Fallback heuristic if causal LM is unavailable (e.g. offline testing without pretrained weights)
            words = clean_text.split()
            # Simple length-based pseudo perplexity fallback
            ppl = float(len(words) * 1.5)
            self._ppl_cache[clean_text] = ppl
            return ppl

        try:
            inputs = self._tokenizer(
                clean_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
            )
            input_ids = inputs["input_ids"]
            if hasattr(self._model, "device"):
                input_ids = input_ids.to(self._model.device)

            if input_ids.shape[1] < 2:
                # Single token, return nominal baseline PPL
                return 1.0

            with torch.no_grad():
                outputs = self._model(input_ids, labels=input_ids)
                loss = outputs.loss.item()
                ppl = math.exp(min(loss, 100.0))  # Guard against overflow

            self._ppl_cache[clean_text] = ppl
            return ppl
        except Exception as exc:
            logger.debug(f"Perplexity computation failed for '{clean_text}': {exc}")
            return 1.0

    def compute_word_drops(self, text: str) -> tuple[float, list[tuple[str, float]]]:
        """
        Compute word-removal perplexity drops for all candidate words in text.
        
        Returns:
            (max_perplexity_drop, list_of_word_drops)
        """
        clean_text = text.strip()
        if not clean_text:
            return 0.0, []

        words = clean_text.split()
        if len(words) < 2:
            # Single word text has no removable candidate without destroying context
            return 0.0, [(clean_text, 0.0)]

        base_ppl = self._compute_sentence_ppl(clean_text)
        word_drops: list[tuple[str, float]] = []

        for i in range(len(words)):
            candidate_word = words[i]
            
            # Policy filtering
            if self.config.candidate_word_policy == "alphanumeric_only":
                if not re.search(r"\w", candidate_word):
                    continue

            # Form sentence without word i
            sentence_without_i = " ".join(words[:i] + words[i + 1 :])
            if not sentence_without_i.strip():
                continue

            sub_ppl = self._compute_sentence_ppl(sentence_without_i)
            delta = base_ppl - sub_ppl
            word_drops.append((candidate_word, delta))

        if not word_drops:
            return 0.0, []

        max_drop = max(d[1] for d in word_drops)
        return max_drop, word_drops

    def fit(self, representations: RepresentationResult, config: Any = None, samples: Any = None) -> None:
        """
        Record reference sample count and configuration.
        ONION is inherently reference-free / unsupervised during token evaluation.
        """
        if config is not None and isinstance(config, OnionDetectorConfig):
            self.config = config

        self.reference_sample_count = len(representations.sample_ids)
        self.is_fitted = True
        logger.info(f"OnionDetector initialized with {self.reference_sample_count} reference samples.")

    def detect(
        self,
        representations: RepresentationResult,
        config: Any = None,
        samples: list[Sample] | None = None,
    ) -> DetectionResult:
        """
        Compute ONION sample-level anomaly scores based on maximum word-removal perplexity drop.
        """
        if config is not None and isinstance(config, OnionDetectorConfig):
            active_config = config
        else:
            active_config = self.config

        sample_ids = list(representations.sample_ids)
        text_map: dict[str, str] = {}
        if samples is not None:
            for s in samples:
                text_map[s.sample_id] = s.text

        scores: list[float] = []
        word_drop_details: dict[str, list[tuple[str, float]]] = {}

        for sid in sample_ids:
            text = text_map.get(sid, "")
            if not text:
                # If text not supplied in samples, assign 0.0
                scores.append(0.0)
                continue

            max_drop, drops = self.compute_word_drops(text)
            # Clip negative drops at 0.0 for anomaly score
            anomaly_score = max(0.0, float(max_drop))
            scores.append(round(anomaly_score, 6))
            word_drop_details[sid] = drops

        threshold = active_config.threshold
        is_anomalous = [score >= threshold for score in scores]

        return DetectionResult(
            sample_ids=sample_ids,
            scores=scores,
            is_anomalous=is_anomalous,
            layer_scores={1: list(scores)},
            detector_name="onion",
            signal_results={"word_drop_details": word_drop_details},
        )
