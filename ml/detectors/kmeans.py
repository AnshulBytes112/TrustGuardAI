import numpy as np
from sklearn.cluster import KMeans

from ml.detectors.schemas import DetectionResult, DetectorConfig
from ml.features.schemas import RepresentationResult
from ml.interfaces import Detector


class KMeansDetector(Detector):
    """
    K-Means clustering distance anomaly detector for transformer representations.
    Fits K-Means clusters on reference embeddings and scores samples by their Euclidean distance
    to the nearest cluster centroid.
    """

    def __init__(self, n_clusters: int = 5, random_state: int = 42) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._models: dict[int, KMeans] = {}
        self._fitted_config: DetectorConfig | None = None
        self._dist_min: dict[int, float] = {}
        self._dist_max: dict[int, float] = {}

    def fit(self, representations: RepresentationResult, config: DetectorConfig) -> None:
        """
        Fits K-Means clusters for each configured layer on reference representations.
        """
        if not representations.sample_ids:
            raise ValueError("Input representations must contain at least one sample.")

        if not config.layers:
            raise ValueError("Detector configuration must specify at least one layer.")

        for layer in config.layers:
            if not representations.layer_representations or layer not in representations.layer_representations:
                raise ValueError(f"Requested layer {layer} is missing from representations.")

        num_samples = len(representations.sample_ids)
        k = min(self.n_clusters, num_samples)
        if k < 1:
            raise ValueError("Number of samples must be at least 1.")

        self._models = {}
        self._dist_min = {}
        self._dist_max = {}

        for layer in config.layers:
            X = representations.layer_representations[layer]

            if not isinstance(X, np.ndarray):
                raise TypeError(f"Layer {layer} representation is not a numpy array.")

            if X.shape[0] != num_samples:
                raise ValueError(f"Layer {layer} row count {X.shape[0]} does not match sample count {num_samples}.")

            if not np.isfinite(X).all():
                raise ValueError(f"Layer {layer} contains NaN or Inf values.")

            # Fit K-Means
            model = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init="auto",
            )
            model.fit(X)
            self._models[layer] = model

            # Distance to nearest centroid for reference samples
            distances = np.min(model.transform(X), axis=1)
            self._dist_min[layer] = float(np.min(distances))
            self._dist_max[layer] = float(np.max(distances))

        self._fitted_config = config

    def detect(
        self, representations: RepresentationResult, config: DetectorConfig
    ) -> DetectionResult:
        """
        Calculates nearest cluster distances for given sample representations.
        """
        if not self._models or self._fitted_config is None:
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

            model = self._models[layer]
            # Minimum distance to any centroid
            raw_distances = np.min(model.transform(X), axis=1)

            # Normalize to [0.0, 1.0] using baseline bounds
            d_min = self._dist_min[layer]
            d_max = self._dist_max[layer]
            diff = d_max - d_min
            if diff > 1e-9:
                norm_scores = np.clip((raw_distances - d_min) / diff, 0.0, 1.0)
            else:
                norm_scores = np.zeros_like(raw_distances)

            layer_scores[layer] = norm_scores.tolist()

        # Aggregation
        all_layer_scores = np.array([layer_scores[l] for l in config.layers])

        if config.aggregation == "mean":
            final_scores = np.mean(all_layer_scores, axis=0)
        elif config.aggregation == "sum":
            final_scores = np.sum(all_layer_scores, axis=0)
        elif config.aggregation == "max":
            final_scores = np.max(all_layer_scores, axis=0)
        else:
            raise ValueError(f"Unknown aggregation strategy: {config.aggregation}")

        final_scores_list = final_scores.tolist()
        is_anomalous = [score >= config.threshold for score in final_scores_list]

        return DetectionResult(
            sample_ids=representations.sample_ids,
            scores=final_scores_list,
            is_anomalous=is_anomalous,
            layer_scores=layer_scores,
            detector_name="kmeans-clustering",
        )
