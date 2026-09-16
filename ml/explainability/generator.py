import re
from collections.abc import Sequence

from ml.data.schemas import Sample
from ml.explainability.schemas import (
    SampleExplanation,
    SuspiciousSampleReport,
    TokenAttribution,
)
from ml.scoring.layer_scores import LayerAnomalyProfile
from ml.scoring.risk_fusion import RiskScoreItem


class ExplanationGenerator:
    """
    Generates explainability evidence, token saliency spans, and multi-factor
    diagnostic summaries for potentially suspicious samples.
    Adheres strictly to the principle of presenting auditable evidence without claiming absolute proof.
    """

    def generate_token_saliency(
        self, text: str, risk_score: float, top_k_ratio: float = 0.25
    ) -> list[TokenAttribution]:
        """
        Tokenizes text into character spans and assigns normalized attribution weights.
        Highlights anomalous tokens based on character structure, rarity, and overall risk magnitude.
        """
        if not text:
            return []

        # Find word/token boundaries
        tokens: list[tuple[str, int, int]] = []
        for match in re.finditer(r"\S+", text):
            tokens.append((match.group(), match.start(), match.end()))

        if not tokens:
            return []

        n = len(tokens)
        attributions: list[TokenAttribution] = []

        # Calculate heuristic token abnormality weights
        raw_weights = []
        for token_str, _, _ in tokens:
            # Punctuation/special token weight + length + uppercase anomalies
            special_char_count = sum(1 for c in token_str if not c.isalnum())
            is_all_upper = token_str.isupper() and len(token_str) > 1
            length_factor = min(len(token_str) / 10.0, 1.0)

            # Heuristic weight
            w = 0.1 + (0.3 * special_char_count) + (0.2 if is_all_upper else 0.0) + (0.1 * length_factor)
            raw_weights.append(w)

        max_w = max(raw_weights) if raw_weights else 1.0
        min_w = min(raw_weights) if raw_weights else 0.0
        diff = max_w - min_w if max_w > min_w else 1.0

        # Scale weights modulated by sample risk_score
        scaled_scores = []
        for w in raw_weights:
            norm_w = (w - min_w) / diff
            # Higher risk elevates top token saliency
            score = float(max(0.0, min(1.0, norm_w * 0.7 + (risk_score * 0.3))))
            scaled_scores.append(score)

        # Flag top K most salient tokens if risk is non-trivial
        k = max(1, int(n * top_k_ratio))
        threshold_score = sorted(scaled_scores, reverse=True)[min(k - 1, len(scaled_scores) - 1)]

        for (token_str, start, end), score in zip(tokens, scaled_scores, strict=True):
            is_suspicious = (score >= threshold_score) and (risk_score >= 0.40)
            attributions.append(
                TokenAttribution(
                    token=token_str,
                    start_char=start,
                    end_char=end,
                    saliency_score=round(score, 4),
                    is_suspicious_span=is_suspicious,
                )
            )

        return attributions

    def generate_explanation(
        self,
        sample: Sample,
        risk_item: RiskScoreItem,
        layer_profile: LayerAnomalyProfile,
        cluster_id: int | None = None,
        cluster_distance: float | None = None,
    ) -> SampleExplanation:
        """
        Creates a structured, auditable explanation for a single evaluated sample.
        """
        # Determine human-readable label
        label_display = sample.label if (sample.label is not None and sample.label_status.value == "KNOWN") else None
        label_status_str = "KNOWN" if label_display is not None else "UNKNOWN"

        # Generate token saliency
        token_attributions = self.generate_token_saliency(sample.text, risk_item.risk_score)

        # Synthesize evidence summary
        summary_parts = []
        if risk_item.risk_level == "HIGH":
            summary_parts.append(
                f"Sample exhibits high anomaly characteristics (Risk Score: {risk_item.risk_score:.2f}, Level: HIGH)."
            )
        elif risk_item.risk_level == "MEDIUM":
            summary_parts.append(
                f"Sample shows moderate deviation from reference baseline (Risk Score: {risk_item.risk_score:.2f}, Level: MEDIUM)."
            )
        else:
            summary_parts.append(
                f"Sample is consistent with reference distribution (Risk Score: {risk_item.risk_score:.2f}, Level: LOW)."
            )

        summary_parts.append(
            f"Dominant evidence source: {risk_item.dominant_evidence.upper()} (Detector: {risk_item.detector_score:.2f}, "
            f"Cluster: {risk_item.cluster_score:.2f}, Layer: {risk_item.layer_score:.2f})."
        )
        summary_parts.append(
            f"Layer attribution is concentrated in Layer {layer_profile.dominant_layer} "
            f"({layer_profile.layer_attributions.get(layer_profile.dominant_layer, 0.0) * 100:.1f}%), "
            f"displaying a {layer_profile.trajectory} depth trajectory."
        )

        suspicious_tokens = [t.token for t in token_attributions if t.is_suspicious_span]
        if suspicious_tokens:
            summary_parts.append(f"Highest attribution token span(s): {', '.join(suspicious_tokens[:3])}.")

        evidence_summary = " ".join(summary_parts)

        return SampleExplanation(
            sample_id=sample.sample_id,
            text=sample.text,
            label=label_display,
            label_status=label_status_str,
            risk_item=risk_item,
            layer_profile=layer_profile,
            token_attributions=token_attributions,
            nearest_cluster_id=cluster_id,
            cluster_distance=cluster_distance,
            evidence_summary=evidence_summary,
        )

    def generate_report(
        self,
        samples: Sequence[Sample],
        risk_items: list[RiskScoreItem],
        layer_profiles: list[LayerAnomalyProfile],
    ) -> SuspiciousSampleReport:
        """
        Aggregates batch explanations and ranks suspicious samples by descending risk score.
        """
        if len(samples) != len(risk_items) or len(samples) != len(layer_profiles):
            raise ValueError("Input collections must have identical length.")

        explanations: list[SampleExplanation] = []
        for sample, risk_item, layer_profile in zip(samples, risk_items, layer_profiles, strict=True):
            exp = self.generate_explanation(sample, risk_item, layer_profile)
            explanations.append(exp)

        # Sort by risk score descending
        explanations.sort(key=lambda x: x.risk_item.risk_score, reverse=True)

        high_count = sum(1 for e in explanations if e.risk_item.risk_level == "HIGH")
        med_count = sum(1 for e in explanations if e.risk_item.risk_level == "MEDIUM")
        low_count = sum(1 for e in explanations if e.risk_item.risk_level == "LOW")
        flagged_count = high_count + med_count

        return SuspiciousSampleReport(
            total_samples=len(samples),
            flagged_samples_count=flagged_count,
            high_risk_count=high_count,
            medium_risk_count=med_count,
            low_risk_count=low_count,
            explanations=explanations,
        )
