from abc import ABC, abstractmethod
from collections.abc import Sequence
import numpy as np

from ml.detectors.trustguard.schemas import SignalResult, TrustGuardConfig
from ml.detectors.trustguard.utils import validate_sample_ids_alignment


class SignalScorer(ABC):
    """
    Interface for combining multiple signal results into unified anomaly scores.
    """

    @abstractmethod
    def score(
        self, signal_results: Sequence[SignalResult], config: TrustGuardConfig
    ) -> list[float]:
        """
        Aggregate signal scores according to config.scoring_strategy and config.weighting_strategy.
        """


class DefaultSignalScorer(SignalScorer):
    """
    Multi-signal aggregator supporting weighted fusion (equal and manual) and rank averaging.
    Strictly verifies sample ID alignment across all independent signal results.
    """

    def score(
        self, signal_results: Sequence[SignalResult], config: TrustGuardConfig
    ) -> list[float]:
        if not signal_results:
            return []

        # Strict alignment verification across all signals
        ref_sample_ids = signal_results[0].sample_ids
        for sr in signal_results:
            if sr.sample_ids != ref_sample_ids:
                raise ValueError(
                    f"Sample ID mismatch between signal '{signal_results[0].signal_type}' "
                    f"and signal '{sr.signal_type}'"
                )
            if sr.items:
                validate_sample_ids_alignment(ref_sample_ids, [sr.items])

        n_samples = len(ref_sample_ids)
        if n_samples == 0:
            return []

        if len(signal_results) == 1:
            return [round(float(s), 6) for s in signal_results[0].scores]

        # Extract score matrix: shape (n_signals, n_samples)
        matrix = np.array([sr.scores for sr in signal_results], dtype=np.float64)

        if config.scoring_strategy == "rank_average":
            # Compute fractional rank for each signal (0 to 1)
            rank_matrix = np.zeros_like(matrix)
            for s_idx in range(len(signal_results)):
                raw_row = matrix[s_idx]
                # argsort twice gives ranks [0, ..., n-1]
                ranks = np.argsort(np.argsort(raw_row)).astype(np.float64)
                if n_samples > 1:
                    rank_matrix[s_idx] = ranks / (n_samples - 1)
                else:
                    rank_matrix[s_idx] = 0.5

            composite = np.mean(rank_matrix, axis=0)

        elif config.scoring_strategy == "weighted_fusion":
            # Resolve weights
            weights: list[float] = []
            if config.weighting_strategy == "manual" and config.weights is not None:
                for sr in signal_results:
                    weights.append(config.weights.get(sr.signal_type, 1.0))
                total_w = sum(weights)
                weights = [w / total_w if total_w > 0 else 1.0 / len(signal_results) for w in weights]
            else:
                # Default 'equal' (and fallback for 'learned_validation' in Phase 2)
                weights = [1.0 / len(signal_results)] * len(signal_results)

            w_arr = np.array(weights, dtype=np.float64).reshape(-1, 1)
            composite = np.sum(matrix * w_arr, axis=0)
        else:
            raise ValueError(f"Unknown scoring strategy: {config.scoring_strategy}")

        return [round(float(np.clip(s, 0.0, 1.0)), 6) for s in composite]
