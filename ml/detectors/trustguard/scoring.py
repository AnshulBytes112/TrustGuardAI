from abc import ABC, abstractmethod
from collections.abc import Sequence
import numpy as np

from ml.detectors.trustguard.schemas import (
    SampleTrustAssessment,
    SignalResult,
    TrustGuardConfig,
    TrustGuardScoreResult,
)
from ml.detectors.trustguard.utils import validate_sample_ids_alignment


class SignalScorer(ABC):
    """
    Interface for combining multiple signal results into unified trust and suspicion scores.
    """

    @abstractmethod
    def score(
        self,
        signal_results: Sequence[SignalResult],
        config: TrustGuardConfig,
        weights: dict[str, float] | None = None,
    ) -> list[float]:
        """
        Aggregate signal scores into composite SuspicionScores in [0.0, 1.0].
        """

    @abstractmethod
    def assess(
        self,
        signal_results: Sequence[SignalResult],
        config: TrustGuardConfig,
        threshold: float = 0.5,
        weights: dict[str, float] | None = None,
    ) -> TrustGuardScoreResult:
        """
        Produce comprehensive TrustGuard trust evaluations including TrustScore,
        SuspicionScore, binary predictions, and signal-wise attribution contributions.
        """


class DefaultSignalScorer(SignalScorer):
    """
    Standard TrustGuard multi-signal aggregator implementing formally documented
    TrustScore and SuspicionScore equations.

    Formal Equations:
        For sample i across K enabled signals:
            Individual Suspicion Signal:  a_j(i) in [0.0, 1.0] (1.0 = maximal anomaly)
            Individual Trustworthiness:    s_j(i) = 1.0 - a_j(i) in [0.0, 1.0]
            Signal Weights:                w_j >= 0 such that sum(w_j) == 1.0

        TrustScore:
            TrustScore_i = sum_{j=1}^K w_j * s_j(i)

        SuspicionScore:
            SuspicionScore_i = 1.0 - TrustScore_i = sum_{j=1}^K w_j * a_j(i)

        Linear Attribution Contribution:
            contribution_j(i) = w_j * a_j(i)
    """

    def resolve_weights(
        self,
        signal_results: Sequence[SignalResult],
        config: TrustGuardConfig,
        explicit_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Resolves and normalizes weights across enabled signals."""
        signal_names = [sr.signal_type for sr in signal_results]
        n_signals = len(signal_names)
        if n_signals == 0:
            return {}

        if explicit_weights is not None:
            raw_w = [max(0.0, explicit_weights.get(name, 0.0)) for name in signal_names]
            total_w = sum(raw_w)
            if total_w > 1e-9:
                return {name: round(w / total_w, 6) for name, w in zip(signal_names, raw_w, strict=True)}

        if config.weighting_strategy == "manual" and config.weights is not None:
            raw_w = [max(0.0, config.weights.get(name, 0.0)) for name in signal_names]
            total_w = sum(raw_w)
            if total_w > 1e-9:
                return {name: round(w / total_w, 6) for name, w in zip(signal_names, raw_w, strict=True)}

        # Default equal weighting
        eq_w = round(1.0 / n_signals, 6)
        return {name: eq_w for name in signal_names}

    def score(
        self,
        signal_results: Sequence[SignalResult],
        config: TrustGuardConfig,
        weights: dict[str, float] | None = None,
    ) -> list[float]:
        if not signal_results:
            return []

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

        matrix = np.array([sr.scores for sr in signal_results], dtype=np.float64)

        if config.scoring_strategy == "rank_average":
            rank_matrix = np.zeros_like(matrix)
            for s_idx in range(len(signal_results)):
                raw_row = matrix[s_idx]
                ranks = np.argsort(np.argsort(raw_row)).astype(np.float64)
                rank_matrix[s_idx] = ranks / (n_samples - 1) if n_samples > 1 else 0.5

            composite = np.mean(rank_matrix, axis=0)
        else:
            # Weighted fusion
            resolved_w_dict = self.resolve_weights(signal_results, config, weights)
            w_list = [resolved_w_dict[sr.signal_type] for sr in signal_results]
            w_arr = np.array(w_list, dtype=np.float64).reshape(-1, 1)
            composite = np.sum(matrix * w_arr, axis=0)

        return [round(float(np.clip(s, 0.0, 1.0)), 6) for s in composite]

    def assess(
        self,
        signal_results: Sequence[SignalResult],
        config: TrustGuardConfig,
        threshold: float = 0.5,
        weights: dict[str, float] | None = None,
    ) -> TrustGuardScoreResult:
        if not signal_results:
            return TrustGuardScoreResult(
                sample_ids=[],
                trust_scores=[],
                suspicion_scores=[],
                predictions=[],
                threshold=threshold,
                weights={},
                assessments=[],
                scoring_strategy=config.scoring_strategy,
                weighting_strategy=config.weighting_strategy,
                config_fingerprint=config.compute_fingerprint(),
            )

        suspicion_scores = self.score(signal_results, config, weights=weights)
        resolved_weights = self.resolve_weights(signal_results, config, weights)
        sample_ids = signal_results[0].sample_ids
        fingerprint = config.compute_fingerprint()

        trust_scores = [round(float(np.clip(1.0 - susp, 0.0, 1.0)), 6) for susp in suspicion_scores]
        predictions = [susp >= threshold for susp in suspicion_scores]

        assessments: list[SampleTrustAssessment] = []
        n_samples = len(sample_ids)

        for i in range(n_samples):
            sid = sample_ids[i]
            susp_i = suspicion_scores[i]
            trust_i = trust_scores[i]
            pred_i = predictions[i]

            ind_susp: dict[str, float] = {}
            ind_trust: dict[str, float] = {}
            contributions: dict[str, float] = {}

            for sr in signal_results:
                s_name = sr.signal_type
                a_val = round(float(sr.scores[i]), 6)
                t_val = round(float(np.clip(1.0 - a_val, 0.0, 1.0)), 6)
                w_val = resolved_weights.get(s_name, 0.0)

                ind_susp[s_name] = a_val
                ind_trust[s_name] = t_val
                contributions[s_name] = round(float(w_val * a_val), 6)

            dominant_signal = max(contributions.items(), key=lambda x: x[1])[0]

            assessment = SampleTrustAssessment(
                sample_id=sid,
                trust_score=trust_i,
                suspicion_score=susp_i,
                threshold=threshold,
                prediction=pred_i,
                individual_signal_scores=ind_susp,
                individual_trust_signals=ind_trust,
                weights=resolved_weights,
                contributions=contributions,
                dominant_signal=dominant_signal,
                config_fingerprint=fingerprint,
                status="SUCCESS",
                provenance="multi_signal_trust_aggregation",
            )
            assessments.append(assessment)

        return TrustGuardScoreResult(
            sample_ids=sample_ids,
            trust_scores=trust_scores,
            suspicion_scores=suspicion_scores,
            predictions=predictions,
            threshold=threshold,
            weights=resolved_weights,
            assessments=assessments,
            scoring_strategy=config.scoring_strategy,
            weighting_strategy=config.weighting_strategy,
            config_fingerprint=fingerprint,
        )
