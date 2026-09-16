from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any
import numpy as np

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


class SemanticSignalExtractor(ABC):
    """
    Interface for extracting semantic consistency anomaly signals across layer representations.
    """

    @abstractmethod
    def fit(
        self,
        reference_representations: RepresentationResult,
        config: TrustGuardConfig,
        samples: Sequence[Sample] | None = None,
        labels: Sequence[Any] | None = None,
    ) -> None:
        """
        Construct class prototypes strictly using TRAIN data.
        """

    @abstractmethod
    def extract(
        self,
        representations: RepresentationResult,
        config: TrustGuardConfig,
        samples: Sequence[Sample] | None = None,
        labels: Sequence[Any] | None = None,
        predicted_labels: Sequence[Any] | None = None,
    ) -> SignalResult:
        """
        Compute semantic anomaly scores for input representations.
        """


class DefaultSemanticSignalExtractor(SemanticSignalExtractor):
    """
    Computes Semantic Class Consistency by comparing sample representations against
    class prototypes constructed strictly on the TRAIN split.
    """

    def __init__(self) -> None:
        self._prototypes: dict[str, np.ndarray] = {}
        self._is_fitted: bool = False

    @property
    def prototypes(self) -> dict[str, np.ndarray]:
        return self._prototypes

    def fit(
        self,
        reference_representations: RepresentationResult,
        config: TrustGuardConfig,
        samples: Sequence[Sample] | None = None,
        labels: Sequence[Any] | None = None,
    ) -> None:
        if not reference_representations.sample_ids:
            raise ValueError("Input representations must contain at least one sample.")

        X = get_layer_aggregated_representations(
            reference_representations, config.layers
        )
        resolved_labels, _ = extract_sample_labels_and_status(
            samples, labels, len(reference_representations.sample_ids)
        )

        # Build class prototypes for all known classes in TRAIN
        class_samples: dict[str, list[np.ndarray]] = {}
        for x_vec, lbl in zip(X, resolved_labels, strict=True):
            if lbl is not None:
                class_samples.setdefault(lbl, []).append(x_vec)

        self._prototypes = {}
        for cls_name, vectors in class_samples.items():
            mean_vec = np.mean(vectors, axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 1e-9:
                self._prototypes[cls_name] = mean_vec / norm
            else:
                self._prototypes[cls_name] = mean_vec

        self._is_fitted = True

    def extract(
        self,
        representations: RepresentationResult,
        config: TrustGuardConfig,
        samples: Sequence[Sample] | None = None,
        labels: Sequence[Any] | None = None,
        predicted_labels: Sequence[Any] | None = None,
    ) -> SignalResult:
        if not representations.sample_ids:
            raise ValueError("Input representations must contain at least one sample.")

        if not self._is_fitted:
            raise RuntimeError("SemanticSignalExtractor must be fitted before extract().")

        fingerprint = config.compute_fingerprint()
        sample_ids = representations.sample_ids
        n_samples = len(sample_ids)

        X = get_layer_aggregated_representations(representations, config.layers)
        X_norm = normalize_vectors(X)

        resolved_labels, _ = extract_sample_labels_and_status(samples, labels, n_samples)
        pred_labels = (
            [str(p) if p is not None else None for p in predicted_labels]
            if predicted_labels is not None
            else [None] * n_samples
        )

        items: list[SampleSignalResult] = []
        scores: list[float] = []

        for i, sid in enumerate(sample_ids):
            x_vec = X_norm[i]
            true_label = resolved_labels[i]
            pred_label = pred_labels[i] if i < len(pred_labels) else None

            # Calculate similarities to all available prototypes
            proto_sims: dict[str, float] = {}
            for cls_name, proto_vec in self._prototypes.items():
                sim = float(np.dot(x_vec, proto_vec))
                proto_sims[cls_name] = round(sim, 6)

            if true_label is not None and true_label in self._prototypes:
                # Labelled sample with known prototype
                assigned_label = true_label
                status = "SUCCESS"
                provenance = "train_ground_truth"
            elif pred_label is not None and pred_label in self._prototypes:
                # Unlabelled sample evaluated using model-predicted class
                assigned_label = pred_label
                status = "PREDICTED_LABEL"
                provenance = "model_prediction"
            else:
                # Unlabelled without valid prototype
                status = "SKIPPED_UNLABELLED"
                provenance = "unlabelled_no_ground_truth"
                assigned_label = None

            if assigned_label is not None and self._prototypes:
                s_true = proto_sims[assigned_label]
                other_sims = [s for c, s in proto_sims.items() if c != assigned_label]

                if other_sims:
                    s_other = max(other_sims)
                    best_alt_class = max(
                        [(c, s) for c, s in proto_sims.items() if c != assigned_label],
                        key=lambda x: x[1],
                    )[0]
                    raw_margin = float(s_true - s_other)
                else:
                    # Single class dataset
                    s_other = 0.0
                    best_alt_class = "NONE"
                    raw_margin = float(s_true)

                # Normalized anomaly score: higher = more inconsistent / anomalous
                # Range: margin [-2, 2] -> [0, 1]
                normalized_anomaly = float(np.clip((1.0 - raw_margin) / 2.0, 0.0, 1.0))
                raw_val = round(raw_margin, 6)
                norm_val = round(normalized_anomaly, 6)

                details = {
                    "assigned_label": assigned_label,
                    "true_class_similarity": s_true,
                    "best_alternative_class": best_alt_class,
                    "best_alternative_similarity": s_other,
                    "margin": raw_val,
                    "prototype_similarities": proto_sims,
                }
            else:
                raw_val = None
                norm_val = 0.5  # Neutral default for unlabelled / uncomputed
                details = {
                    "reason": "Sample has no known or predicted label matching training prototypes.",
                    "prototype_similarities": proto_sims,
                }

            items.append(
                SampleSignalResult(
                    sample_id=sid,
                    signal_name="semantic",
                    raw_value=raw_val,
                    normalized_value=norm_val,
                    status=status,
                    provenance=provenance,
                    config_fingerprint=fingerprint,
                    details=details,
                )
            )
            scores.append(norm_val if norm_val is not None else 0.5)

        return SignalResult(
            signal_type="semantic",
            scores=scores,
            sample_ids=sample_ids,
            items=items,
            metadata={
                "prototypes_count": len(self._prototypes),
                "classes": list(self._prototypes.keys()),
            },
        )
