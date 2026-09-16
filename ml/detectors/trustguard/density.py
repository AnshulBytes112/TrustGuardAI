from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from ml.detectors.trustguard.schemas import (
    SampleSignalResult,
    SignalResult,
    TrustGuardConfig,
)
from ml.detectors.trustguard.utils import (
    get_layer_aggregated_representations,
    normalize_vectors,
)
from ml.features.schemas import RepresentationResult


class DensitySignalExtractor(ABC):
    """
    Interface for extracting manifold density anomaly signals.
    Evaluates relative density under reference training manifolds.
    """

    @abstractmethod
    def fit(self, reference_representations: RepresentationResult, config: TrustGuardConfig) -> None:
        """
        Fit density estimator and compute reference normalization statistics on TRAIN split.
        """

    @abstractmethod
    def extract(
        self, representations: RepresentationResult, config: TrustGuardConfig
    ) -> SignalResult:
        """
        Compute density outlier scores for target representations using training normalization stats.
        """


class DefaultDensitySignalExtractor(DensitySignalExtractor):
    """
    Computes Local Representation Density / Outlierness using either k-NN distance
    or Local Outlier Factor (LOF) fitted strictly on the TRAIN representations.
    Normalization statistics are computed exclusively on TRAIN and applied to eval splits.
    """

    def __init__(self) -> None:
        self._nn: NearestNeighbors | None = None
        self._lof: LocalOutlierFactor | None = None
        self._density_method: str | None = None
        self._norm_min: float = 0.0
        self._norm_max: float = 1.0
        self._is_fitted: bool = False
        self._n_train: int = 0

    def fit(self, reference_representations: RepresentationResult, config: TrustGuardConfig) -> None:
        if not reference_representations.sample_ids:
            raise ValueError("Input representations must contain at least one sample.")

        X = get_layer_aggregated_representations(reference_representations, config.layers)
        X_norm = normalize_vectors(X)

        self._n_train = len(reference_representations.sample_ids)
        self._density_method = config.density_method
        effective_k = max(1, min(config.neighborhood_k, self._n_train))

        if self._density_method == "knn_distance":
            self._nn = NearestNeighbors(
                n_neighbors=effective_k,
                metric="cosine",
                algorithm="brute",
            )
            self._nn.fit(X_norm)
            train_dists, _ = self._nn.kneighbors(X_norm, n_neighbors=effective_k)
            mean_train_dists = np.mean(train_dists, axis=1)

            self._norm_min = float(np.min(mean_train_dists))
            self._norm_max = float(np.max(mean_train_dists))

        elif self._density_method == "local_outlier_factor":
            # scikit-learn LOF requires n_neighbors < n_samples for novelty=True
            lof_k = max(1, min(config.neighborhood_k, self._n_train - 1)) if self._n_train > 1 else 1
            self._lof = LocalOutlierFactor(
                n_neighbors=lof_k,
                novelty=True,
                metric="cosine",
                algorithm="brute",
            )
            self._lof.fit(X_norm)
            # score_samples yields negative outlier factors (lower = more anomalous)
            train_scores = self._lof.score_samples(X_norm)
            self._norm_min = float(np.min(train_scores))
            self._norm_max = float(np.max(train_scores))
        else:
            raise ValueError(f"Unsupported density method: {self._density_method}")

        self._is_fitted = True

    def extract(
        self, representations: RepresentationResult, config: TrustGuardConfig
    ) -> SignalResult:
        if not representations.sample_ids:
            raise ValueError("Input representations must contain at least one sample.")

        if not self._is_fitted:
            raise RuntimeError("DensitySignalExtractor must be fitted before extract().")

        fingerprint = config.compute_fingerprint()
        sample_ids = representations.sample_ids
        n_samples = len(sample_ids)

        X = get_layer_aggregated_representations(representations, config.layers)
        X_norm = normalize_vectors(X)

        denom = self._norm_max - self._norm_min
        if denom < 1e-9:
            denom = 1.0

        items: list[SampleSignalResult] = []
        scores: list[float] = []

        if self._density_method == "knn_distance" and self._nn is not None:
            effective_k = max(1, min(config.neighborhood_k, self._n_train))
            distances, indices = self._nn.kneighbors(X_norm, n_neighbors=effective_k)
            mean_dists = np.mean(distances, axis=1)

            for i, sid in enumerate(sample_ids):
                raw_d = float(mean_dists[i])
                # Higher distance to reference manifold = lower density / more anomalous
                norm_anomaly = float(np.clip((raw_d - self._norm_min) / denom, 0.0, 1.0))
                raw_val = round(raw_d, 6)
                norm_val = round(norm_anomaly, 6)

                details = {
                    "density_method": "knn_distance",
                    "mean_knn_distance": raw_val,
                    "neighbor_distances": [round(float(d), 6) for d in distances[i]],
                    "train_min_dist": round(self._norm_min, 6),
                    "train_max_dist": round(self._norm_max, 6),
                }

                items.append(
                    SampleSignalResult(
                        sample_id=sid,
                        signal_name="density",
                        raw_value=raw_val,
                        normalized_value=norm_val,
                        status="SUCCESS",
                        provenance="train_manifold",
                        config_fingerprint=fingerprint,
                        details=details,
                    )
                )
                scores.append(norm_val)

        elif self._density_method == "local_outlier_factor" and self._lof is not None:
            raw_scores = self._lof.score_samples(X_norm)

            for i, sid in enumerate(sample_ids):
                raw_s = float(raw_scores[i])
                # Lower LOF score = more anomalous. Invert direction to align with anomaly score [0, 1]
                norm_anomaly = float(np.clip((self._norm_max - raw_s) / denom, 0.0, 1.0))
                raw_val = round(raw_s, 6)
                norm_val = round(norm_anomaly, 6)

                details = {
                    "density_method": "local_outlier_factor",
                    "raw_lof_score": raw_val,
                    "train_min_lof": round(self._norm_min, 6),
                    "train_max_lof": round(self._norm_max, 6),
                }

                items.append(
                    SampleSignalResult(
                        sample_id=sid,
                        signal_name="density",
                        raw_value=raw_val,
                        normalized_value=norm_val,
                        status="SUCCESS",
                        provenance="train_manifold",
                        config_fingerprint=fingerprint,
                        details=details,
                    )
                )
                scores.append(norm_val)

        return SignalResult(
            signal_type="density",
            scores=scores,
            sample_ids=sample_ids,
            items=items,
            metadata={
                "density_method": self._density_method,
                "train_samples_count": self._n_train,
                "train_norm_min": self._norm_min,
                "train_norm_max": self._norm_max,
            },
        )
