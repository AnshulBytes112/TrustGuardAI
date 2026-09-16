from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from typing import Any
import numpy as np
from sklearn.neighbors import NearestNeighbors

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


class NeighborhoodSignalExtractor(ABC):
    """
    Interface for extracting k-NN neighborhood consistency anomaly signals.
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
        Index reference representations (strictly from TRAIN split) for k-NN queries.
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
        Compute neighborhood deviation scores for target representations.
        """


class DefaultNeighborhoodSignalExtractor(NeighborhoodSignalExtractor):
    """
    Computes Local Neighborhood Consistency by evaluating label agreement and local purity
    against a reference k-NN graph fitted exclusively on TRAIN representations.
    """

    def __init__(self) -> None:
        self._nn: NearestNeighbors | None = None
        self._train_labels: list[str | None] = []
        self._train_sample_ids: list[str] = []
        self._is_fitted: bool = False

    def fit(
        self,
        reference_representations: RepresentationResult,
        config: TrustGuardConfig,
        samples: Sequence[Sample] | None = None,
        labels: Sequence[Any] | None = None,
    ) -> None:
        if not reference_representations.sample_ids:
            raise ValueError("Input representations must contain at least one sample.")

        X = get_layer_aggregated_representations(reference_representations, config.layers)
        X_norm = normalize_vectors(X)

        resolved_labels, _ = extract_sample_labels_and_status(
            samples, labels, len(reference_representations.sample_ids)
        )

        self._train_sample_ids = list(reference_representations.sample_ids)
        self._train_labels = resolved_labels

        n_train = len(self._train_sample_ids)
        effective_k = max(1, min(config.neighborhood_k, n_train))

        self._nn = NearestNeighbors(
            n_neighbors=effective_k,
            metric="cosine",
            algorithm="brute",
        )
        self._nn.fit(X_norm)
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

        if not self._is_fitted or self._nn is None:
            raise RuntimeError("NeighborhoodSignalExtractor must be fitted before extract().")

        fingerprint = config.compute_fingerprint()
        sample_ids = representations.sample_ids
        n_samples = len(sample_ids)
        n_train = len(self._train_sample_ids)

        effective_k = max(1, min(config.neighborhood_k, n_train))

        X = get_layer_aggregated_representations(representations, config.layers)
        X_norm = normalize_vectors(X)

        resolved_labels, _ = extract_sample_labels_and_status(samples, labels, n_samples)

        distances, indices = self._nn.kneighbors(X_norm, n_neighbors=effective_k)

        items: list[SampleSignalResult] = []
        scores: list[float] = []

        for i, sid in enumerate(sample_ids):
            target_label = resolved_labels[i]
            neighbor_idxs = indices[i].tolist()
            neighbor_dists = [round(float(d), 6) for d in distances[i]]
            neighbor_lbls = [self._train_labels[idx] for idx in neighbor_idxs]

            known_neighbor_lbls = [lbl for lbl in neighbor_lbls if lbl is not None]

            # 1. Local label purity
            if known_neighbor_lbls:
                counts = Counter(known_neighbor_lbls)
                dominant_label, max_count = counts.most_common(1)[0]
                local_purity = float(max_count / len(known_neighbor_lbls))
            else:
                dominant_label = "UNKNOWN"
                local_purity = 0.0

            # 2. Decision: agreement for labelled, purity for unlabelled
            if target_label is not None:
                # Labelled sample
                matching_count = sum(1 for lbl in known_neighbor_lbls if lbl == target_label)
                agreement = float(matching_count / len(known_neighbor_lbls)) if known_neighbor_lbls else 0.0
                raw_val = round(agreement, 6)
                norm_anomaly = float(np.clip(1.0 - agreement, 0.0, 1.0))
                status = "SUCCESS"
                provenance = "train_ground_truth"
            else:
                # Unlabelled sample
                agreement = None
                raw_val = round(local_purity, 6)
                norm_anomaly = float(np.clip(1.0 - local_purity, 0.0, 1.0))
                status = "UNLABELLED_PURITY"
                provenance = "unsupervised_neighborhood"

            norm_val = round(norm_anomaly, 6)

            details = {
                "effective_k": effective_k,
                "target_label": target_label,
                "neighbor_indices": neighbor_idxs,
                "neighbor_labels": neighbor_lbls,
                "neighbor_distances": neighbor_dists,
                "label_agreement": agreement,
                "local_purity": round(local_purity, 6),
                "dominant_neighbor_label": dominant_label,
            }

            items.append(
                SampleSignalResult(
                    sample_id=sid,
                    signal_name="neighborhood",
                    raw_value=raw_val,
                    normalized_value=norm_val,
                    status=status,
                    provenance=provenance,
                    config_fingerprint=fingerprint,
                    details=details,
                )
            )
            scores.append(norm_val)

        return SignalResult(
            signal_type="neighborhood",
            scores=scores,
            sample_ids=sample_ids,
            items=items,
            metadata={
                "effective_k": effective_k,
                "reference_samples_count": n_train,
            },
        )
