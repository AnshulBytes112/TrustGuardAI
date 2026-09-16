from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field


class LayerAnomalyProfile(BaseModel):
    sample_id: str
    layer_scores: dict[int, float] = Field(..., description="Raw or normalized score per layer")
    layer_attributions: dict[int, float] = Field(..., description="Proportional contribution of each layer (sums to 1.0)")
    dominant_layer: int = Field(..., description="The layer with the highest anomaly contribution")
    trajectory: Literal["early", "middle", "late", "uniform"] = Field(
        ..., description="Which depth section of the network exhibits highest anomaly concentration"
    )


@dataclass(frozen=True)
class LayerDecomposer:
    """
    Analyzes multi-layer anomaly scores across transformer representation layers.
    Decomposes layer-wise contributions to identify the structural depth where anomalies manifest.
    """

    def decompose_sample(self, sample_id: str, layer_scores: dict[int, float]) -> LayerAnomalyProfile:
        if not layer_scores:
            raise ValueError("layer_scores dictionary cannot be empty.")

        total_score = sum(layer_scores.values())
        layers_sorted = sorted(layer_scores.keys())

        # Compute percentage attribution
        if total_score > 1e-9:
            attributions = {layer: float(score / total_score) for layer, score in layer_scores.items()}
        else:
            uniform_val = 1.0 / len(layer_scores)
            attributions = {layer: uniform_val for layer in layer_scores}

        # Identify dominant layer
        dominant_layer = max(layer_scores.items(), key=lambda item: item[1])[0]

        # Categorize depth trajectory
        # For a 6-layer model (e.g. DistilBERT 1-6): early (1-2), middle (3-4), late (5-6)
        n = len(layers_sorted)
        if n >= 3:
            early_layers = layers_sorted[: n // 3]
            late_layers = layers_sorted[-(n // 3) :]
            middle_layers = [l for l in layers_sorted if l not in early_layers and l not in late_layers]

            early_sum = sum(attributions[l] for l in early_layers)
            mid_sum = sum(attributions[l] for l in middle_layers)
            late_sum = sum(attributions[l] for l in late_layers)

            max_sum = max(early_sum, mid_sum, late_sum)
            if max_sum - min(early_sum, mid_sum, late_sum) < 0.10:
                trajectory = "uniform"
            elif max_sum == early_sum:
                trajectory = "early"
            elif max_sum == mid_sum:
                trajectory = "middle"
            else:
                trajectory = "late"
        else:
            trajectory = "uniform"

        return LayerAnomalyProfile(
            sample_id=sample_id,
            layer_scores=layer_scores,
            layer_attributions=attributions,
            dominant_layer=dominant_layer,
            trajectory=trajectory,
        )

    def decompose_batch(
        self, sample_ids: list[str], layer_scores_by_layer: dict[int, list[float]]
    ) -> list[LayerAnomalyProfile]:
        profiles = []
        for i, sid in enumerate(sample_ids):
            sample_layer_scores = {
                layer: float(scores[i]) for layer, scores in layer_scores_by_layer.items() if i < len(scores)
            }
            profiles.append(self.decompose_sample(sid, sample_layer_scores))
        return profiles
