import numpy as np

from ml.detectors.schemas import DetectionResult, DetectorConfig
from ml.features.schemas import RepresentationResult
from ml.interfaces import Detector


class FlareDetector(Detector):
    """
    A minimal, deterministic approximation of the FLARE multi-layer anomaly detector.
    Computes Euclidean distance to the centroid of L2-normalized layer representations.
    Requires a separate `fit` step to establish the reference centroid before `detect`.
    """

    def __init__(self) -> None:
        self._centroids: dict[int, np.ndarray] = {}
        self._fitted_config: DetectorConfig | None = None

    def fit(self, representations: RepresentationResult, config: DetectorConfig) -> None:
        """
        Establishes the reference centroid using the provided representations (usually TRAIN split).
        """
        if not representations.sample_ids:
            raise ValueError("Input representations must contain at least one sample.")
            
        if not config.layers:
            raise ValueError("Detector configuration must specify at least one layer.")

        for layer in config.layers:
            if not representations.layer_representations or layer not in representations.layer_representations:
                raise ValueError(f"Requested layer {layer} is missing from representations.")

        num_samples = len(representations.sample_ids)
        self._centroids = {}

        for layer in config.layers:
            X = representations.layer_representations[layer]

            if not isinstance(X, np.ndarray):
                raise TypeError(f"Layer {layer} representation is not a numpy array.")

            if X.shape[0] != num_samples:
                raise ValueError(f"Layer {layer} row count {X.shape[0]} does not match sample count {num_samples}.")
                
            if not np.isfinite(X).all():
                raise ValueError(f"Layer {layer} contains NaN or Inf values.")

            # 1. Normalization (L2)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            X_norm = X / norms

            # 2. Compute Centroid
            self._centroids[layer] = np.mean(X_norm, axis=0)
            
        self._fitted_config = config

    def detect(
        self, representations: RepresentationResult, config: DetectorConfig
    ) -> DetectionResult:
        """
        Scores samples against the already-fitted reference centroid.
        """
        if not self._centroids or self._fitted_config is None:
            raise RuntimeError("Detector must be fitted with reference data before calling detect().")
            
        if config.layers != self._fitted_config.layers:
            raise ValueError("Detection config layers must match the fitted config layers.")

        if not representations.sample_ids:
            raise ValueError("Input representations must contain at least one sample.")

        for layer in config.layers:
            if not representations.layer_representations or layer not in representations.layer_representations:
                raise ValueError(f"Requested layer {layer} is missing from representations.")

        num_samples = len(representations.sample_ids)
        layer_scores: dict[int, list[float]] = {}

        for layer in config.layers:
            X = representations.layer_representations[layer]

            if not isinstance(X, np.ndarray):
                raise TypeError(f"Layer {layer} representation is not a numpy array.")

            if X.shape[0] != num_samples:
                raise ValueError(f"Layer {layer} row count {X.shape[0]} does not match sample count {num_samples}.")
                
            if not np.isfinite(X).all():
                raise ValueError(f"Layer {layer} contains NaN or Inf values.")

            # 1. Normalization (L2)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            X_norm = X / norms

            # 2. Centroid distance (Anomaly measure) using PRE-FITTED centroid
            centroid = self._centroids[layer]
            distances = np.linalg.norm(X_norm - centroid, axis=1)
            
            layer_scores[layer] = distances.tolist()

        # 3. Aggregation
        all_layer_scores = np.array([layer_scores[l] for l in config.layers])
        
        if config.aggregation == "mean":
            final_scores = np.mean(all_layer_scores, axis=0)
        elif config.aggregation == "sum":
            final_scores = np.sum(all_layer_scores, axis=0)
        elif config.aggregation == "max":
            final_scores = np.max(all_layer_scores, axis=0)
        else:
            raise ValueError(f"Unknown aggregation strategy: {config.aggregation}")

        # 4. Thresholding
        final_scores_list = final_scores.tolist()
        is_anomalous = [score >= config.threshold for score in final_scores_list]

        return DetectionResult(
            sample_ids=representations.sample_ids,
            scores=final_scores_list,
            is_anomalous=is_anomalous,
            layer_scores=layer_scores,
            detector_name="flare-centroid-baseline",
        )
