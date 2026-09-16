from abc import ABC, abstractmethod
from collections.abc import Sequence
import hashlib
import random
from typing import Any
import numpy as np
from sklearn.linear_model import LogisticRegression

from ml.data.schemas import Sample
from ml.detectors.trustguard.schemas import (
    SampleSignalResult,
    SignalResult,
    TrustGuardConfig,
)
from ml.detectors.trustguard.utils import (
    extract_sample_labels_and_status,
    get_layer_aggregated_representations,
    normalize_vectors,
)
from ml.features.schemas import RepresentationResult
from ml.interfaces import RepresentationProvider

# Curated deterministic synonym dictionary for semantic-preserving substitutions
SYNONYM_MAP: dict[str, list[str]] = {
    "good": ["decent", "fine", "great", "sound"],
    "great": ["excellent", "superb", "terrific", "stellar"],
    "bad": ["poor", "subpar", "adverse", "faulty"],
    "clean": ["pure", "clear", "untainted", "sanitized"],
    "fast": ["quick", "rapid", "swift", "speedy"],
    "slow": ["sluggish", "unhurried", "leisurely"],
    "happy": ["glad", "cheerful", "pleased", "content"],
    "sad": ["gloomy", "unhappy", "sorrowful", "downcast"],
    "big": ["large", "huge", "sizable", "substantial"],
    "small": ["tiny", "little", "compact", "mini"],
    "test": ["evaluation", "trial", "check", "assessment"],
    "data": ["dataset", "records", "samples", "information"],
    "model": ["classifier", "network", "system", "architecture"],
    "positive": ["affirmative", "favorable", "constructive"],
    "negative": ["adverse", "critical", "unfavorable"],
}


def generate_text_perturbations(
    text: str,
    strategy: str,
    count: int,
    seed: int,
) -> list[str]:
    """
    Generates semantic-preserving text perturbations deterministically.
    """
    if count <= 0 or not text:
        return [text] * max(1, count)

    perturbations: list[str] = []
    words = text.split()

    for i in range(count):
        # Deterministic seed per variation
        step_seed = int(
            hashlib.sha256(f"{seed}_{text}_{strategy}_{i}".encode("utf-8")).hexdigest()[:8],
            16,
        )
        rng = random.Random(step_seed)

        if strategy == "synonym_swap" and words:
            new_words = list(words)
            eligible_indices = [
                idx for idx, w in enumerate(words) if w.lower().strip(".,!?:;") in SYNONYM_MAP
            ]
            if eligible_indices:
                target_idx = rng.choice(eligible_indices)
                clean_w = words[target_idx].lower().strip(".,!?:;")
                synonyms = SYNONYM_MAP[clean_w]
                replacement = rng.choice(synonyms)
                # Retain punctuation
                if words[target_idx].endswith("."):
                    replacement += "."
                elif words[target_idx].endswith(","):
                    replacement += ","
                new_words[target_idx] = replacement
                perturbations.append(" ".join(new_words))
            else:
                # Fallback to minor character noise if no known synonym
                perturbations.append(_apply_character_noise(text, rng))

        elif strategy == "character_noise" or strategy == "synonym_swap":
            perturbations.append(_apply_character_noise(text, rng))
        else:
            # Default fallback to character noise
            perturbations.append(_apply_character_noise(text, rng))

    return perturbations


def _apply_character_noise(text: str, rng: random.Random) -> str:
    """Applies minor semantic-preserving character variation."""
    if len(text) < 4:
        return text

    chars = list(text)
    choice = rng.randint(0, 2)
    # Pick a middle character (avoid breaking punctuation/first letter)
    idx = rng.randint(1, len(chars) - 2)

    if choice == 0 and chars[idx].isalpha() and chars[idx + 1].isalpha():
        # Transpose adjacent characters
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    elif choice == 1 and chars[idx].isalpha():
        # Duplicate a character
        chars.insert(idx, chars[idx])
    elif choice == 2:
        # Space perturbation
        chars.insert(idx, " ")

    return "".join(chars)


class StabilitySignalExtractor(ABC):
    """
    Interface for extracting perturbation stability anomaly signals.
    Evaluates real model prediction consistency under semantic-preserving text perturbations.
    """

    @abstractmethod
    def fit(
        self,
        reference_representations: RepresentationResult,
        config: TrustGuardConfig,
        samples: Sequence[Sample] | None = None,
        labels: Sequence[Any] | None = None,
        classifier: Any = None,
        representation_provider: RepresentationProvider | None = None,
    ) -> None:
        """
        Fits downstream classifier strictly on TRAIN representations.
        """

    @abstractmethod
    def extract(
        self,
        representations: RepresentationResult,
        config: TrustGuardConfig,
        samples: Sequence[Sample] | None = None,
        labels: Sequence[Any] | None = None,
    ) -> SignalResult:
        """
        Compute prediction stability deviation scores for target samples.
        """


class DefaultStabilitySignalExtractor(StabilitySignalExtractor):
    """
    Computes Prediction Stability by applying semantic-preserving perturbations
    and evaluating prediction consistency with a real classifier fitted on TRAIN.
    """

    def __init__(
        self,
        classifier: Any = None,
        representation_provider: RepresentationProvider | None = None,
    ) -> None:
        self._classifier = classifier
        self._representation_provider = representation_provider
        self._is_fitted: bool = False
        self._single_class_default: str | None = None

    @property
    def classifier(self) -> Any:
        return self._classifier

    def fit(
        self,
        reference_representations: RepresentationResult,
        config: TrustGuardConfig,
        samples: Sequence[Sample] | None = None,
        labels: Sequence[Any] | None = None,
        classifier: Any = None,
        representation_provider: RepresentationProvider | None = None,
    ) -> None:
        if not reference_representations.sample_ids:
            raise ValueError("Input representations must contain at least one sample.")

        if representation_provider is not None:
            self._representation_provider = representation_provider

        if classifier is not None:
            self._classifier = classifier
            self._is_fitted = True
            return

        X = get_layer_aggregated_representations(reference_representations, config.layers)
        X_norm = normalize_vectors(X)

        resolved_labels, _ = extract_sample_labels_and_status(
            samples, labels, len(reference_representations.sample_ids)
        )

        train_mask = [i for i, lbl in enumerate(resolved_labels) if lbl is not None]

        if not train_mask:
            # Unsupervised / unlabelled train split -> cannot train supervised classifier
            self._classifier = None
            self._is_fitted = True
            return

        X_train = X_norm[train_mask]
        y_train = [resolved_labels[i] for i in train_mask]

        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            self._single_class_default = str(unique_classes[0])
            self._classifier = None
        else:
            clf = LogisticRegression(max_iter=1000, random_state=config.seed)
            clf.fit(X_train, y_train)
            self._classifier = clf
            self._single_class_default = None

        self._is_fitted = True

    def extract(
        self,
        representations: RepresentationResult,
        config: TrustGuardConfig,
        samples: Sequence[Sample] | None = None,
        labels: Sequence[Any] | None = None,
    ) -> SignalResult:
        if not representations.sample_ids:
            raise ValueError("Input representations must contain at least one sample.")

        if not self._is_fitted:
            raise RuntimeError("StabilitySignalExtractor must be fitted before extract().")

        fingerprint = config.compute_fingerprint()
        sample_ids = representations.sample_ids
        n_samples = len(sample_ids)

        X = get_layer_aggregated_representations(representations, config.layers)
        X_norm = normalize_vectors(X)

        # Baseline predictions using fitted classifier
        if self._classifier is not None:
            base_preds = self._classifier.predict(X_norm).tolist()
        elif self._single_class_default is not None:
            base_preds = [self._single_class_default] * n_samples
        else:
            base_preds = [None] * n_samples

        items: list[SampleSignalResult] = []
        scores: list[float] = []

        # Check if text samples and representation provider are available for real perturbation evaluation
        has_text = samples is not None and len(samples) == n_samples and all(bool(s.text) for s in samples)
        can_perturb = has_text and self._representation_provider is not None and (self._classifier is not None or self._single_class_default is not None)

        for i, sid in enumerate(sample_ids):
            base_p = base_preds[i]

            if not can_perturb:
                # Text or representation inference unavailable -> mark SKIPPED without fabricating signals
                items.append(
                    SampleSignalResult(
                        sample_id=sid,
                        signal_name="stability",
                        raw_value=None,
                        normalized_value=0.5,
                        status="SKIPPED",
                        provenance="skipped_missing_text_or_classifier",
                        config_fingerprint=fingerprint,
                        details={
                            "reason": "Text or representation extractor is unavailable for perturbation inference.",
                            "base_prediction": str(base_p) if base_p is not None else None,
                        },
                    )
                )
                scores.append(0.5)
                continue

            sample_obj = samples[i]  # type: ignore
            pert_texts = generate_text_perturbations(
                text=sample_obj.text,
                strategy=config.perturbation_strategy,
                count=config.perturbation_count,
                seed=config.seed + i,
            )

            pert_samples = [
                Sample(
                    sample_id=f"{sid}_pert_{p_idx}",
                    text=p_text,
                    label=sample_obj.label,
                    label_status=sample_obj.label_status,
                    split=sample_obj.split,
                    dataset_id=sample_obj.dataset_id,
                    dataset_version=sample_obj.dataset_version,
                )
                for p_idx, p_text in enumerate(pert_texts)
            ]

            # Real feature extraction on perturbed texts
            pert_reps = self._representation_provider.extract(pert_samples)
            X_pert = get_layer_aggregated_representations(pert_reps, config.layers)
            X_pert_norm = normalize_vectors(X_pert)

            if self._classifier is not None:
                pert_preds = self._classifier.predict(X_pert_norm).tolist()
            else:
                pert_preds = [self._single_class_default] * len(pert_texts)

            # Measure consistency: fraction of perturbed predictions matching base_p
            matches = sum(1 for p in pert_preds if p == base_p)
            consistency_rate = float(matches / len(pert_preds)) if pert_preds else 1.0

            raw_val = round(consistency_rate, 6)
            norm_anomaly = float(np.clip(1.0 - consistency_rate, 0.0, 1.0))
            norm_val = round(norm_anomaly, 6)

            items.append(
                SampleSignalResult(
                    sample_id=sid,
                    signal_name="stability",
                    raw_value=raw_val,
                    normalized_value=norm_val,
                    status="SUCCESS",
                    provenance="model_prediction_stability",
                    config_fingerprint=fingerprint,
                    details={
                        "base_prediction": str(base_p),
                        "perturbed_predictions": [str(p) for p in pert_preds],
                        "perturbation_count": len(pert_preds),
                        "perturbation_strategy": config.perturbation_strategy,
                        "consistency_rate": raw_val,
                        "perturbed_texts": pert_texts,
                    },
                )
            )
            scores.append(norm_val)

        return SignalResult(
            signal_type="stability",
            scores=scores,
            sample_ids=sample_ids,
            items=items,
            metadata={
                "perturbation_count": config.perturbation_count,
                "perturbation_strategy": config.perturbation_strategy,
                "can_perturb": can_perturb,
            },
        )
