import numpy as np
from sklearn.ensemble import IsolationForest

from ml.detectors.schemas import DetectionResult, DetectorConfig
from ml.features.schemas import RepresentationResult
from ml.interfaces import Detector


class IsolationForestDetector(Detector):
    """
    Isolation Forest anomaly detector for transformer hidden representations.
    Trains an Isolation Forest model per specified layer on reference embeddings (e.g. TRAIN split).
    Higher anomaly scores represent higher abnormality in [0.0, 1.0].
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float | str = "auto",
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self._models: dict[int, IsolationForest] = {}
        self._fitted_config: DetectorConfig | None = None
        self._score_min: dict[int, float] = {}
        self._score_max: dict[int, float] = {}

    def fit(self, representations: RepresentationResult, config: DetectorConfig) -> None:
        """
        Fits an Isolation Forest model for each configured layer on reference representations.
        """
        if not representations.sample_ids:
            raise ValueError("Input representations must contain at least one sample.")

        if not config.layers:
            raise ValueError("Detector configuration must specify at least one layer.")

        for layer in config.layers:
            if not representations.layer_representations or layer not in representations.layer_representations:
                raise ValueError(f"Requested layer {layer} is missing from representations.")

        num_samples = len(representations.sample_ids)
        self._models = {}
        self._score_min = {}
        self._score_max = {}

        for layer in config.layers:
            X = representations.layer_representations[layer]

            if not isinstance(X, np.ndarray):
                raise TypeError(f"Layer {layer} representation is not a numpy array.")

            if X.shape[0] != num_samples:
                raise ValueError(f"Layer {layer} row count {X.shape[0]} does not match sample count {num_samples}.")

            if not np.isfinite(X).all():
                raise ValueError(f"Layer {layer} contains NaN or Inf values.")

            # Fit Isolation Forest
            model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1,
            )
            model.fit(X)
            self._models[layer] = model

            # Determine baseline score range for normalization
            raw_scores = -model.score_samples(X)  # Inverted so larger is more anomalous
            self._score_min[layer] = float(np.min(raw_scores))
            self._score_max[layer] = float(np.max(raw_scores))

        self._fitted_config = config

    def detect(
        self, representations: RepresentationResult, config: DetectorConfig
    ) -> DetectionResult:
        """
        Scores samples against the fitted Isolation Forest models.
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
            # score_samples returns negative anomaly score (lower is more anomalous)
            # Invert so higher is more anomalous
            raw_scores = -model.score_samples(X)

            # Normalize scores to [0.0, 1.0] using fitted baseline bounds
            s_min = self._score_min[layer]
            s_max = self._score_max[layer]
            diff = s_max - s_min
            if diff > 1e-9:
                norm_scores = np.clip((raw_scores - s_min) / diff, 0.0, 1.0)
            else:
                norm_scores = np.zeros_like(raw_scores)

            layer_scores[layer] = norm_scores.tolist()

        # Aggregate across layers
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
            detector_name="isolation-forest",
        )
