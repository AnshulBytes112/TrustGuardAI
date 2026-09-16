from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DetectorMethod = Literal["flare", "trustguard", "onion", "random_filtering"]


class DetectorConfig(BaseModel):
    """Configuration for baseline FLARE centroid detector."""
    layers: tuple[int, ...] = Field(default=(1, 2, 3, 4, 5, 6), description="Layers to use for detection")
    threshold: float = Field(default=0.5, description="Decision threshold for anomaly score")
    aggregation: Literal["mean", "sum", "max"] = Field(default="mean")

    model_config = ConfigDict(frozen=True)


class RandomFilteringConfig(BaseModel):
    """Configuration for deterministic random filtering baseline."""
    seed: int = Field(default=42, description="Random seed for reproducible scoring")
    filtering_budget: float = Field(default=0.20, ge=0.0, le=1.0, description="Target fraction of samples to flag/remove")
    threshold: float = Field(default=0.5, description="Decision threshold for score")

    model_config = ConfigDict(frozen=True)


class OnionDetectorConfig(BaseModel):
    """
    Configuration for ONION (Qi et al., EMNLP 2021) word-removal perplexity defense.
    Uses a causal language model (e.g. GPT-2) to compute token/word-removal perplexity drops.
    """
    language_model_name: str = Field(default="gpt2", description="HuggingFace causal LM identifier or local model path")
    tokenizer_name: str = Field(default="gpt2", description="HuggingFace tokenizer identifier or local path")
    max_length: int = Field(default=128, ge=8, le=512, description="Max token sequence length")
    candidate_word_policy: Literal["all_words", "alphanumeric_only"] = Field(
        default="all_words", description="Strategy for selecting candidate words to remove"
    )
    batch_size: int = Field(default=16, ge=1, description="Inference batch size")
    device: str = Field(default="cpu", description="Compute device ('cpu' or 'cuda')")
    threshold: float = Field(default=0.0, description="Decision threshold on max perplexity drop")
    threshold_strategy: Literal["validation_calibrated", "fixed"] = Field(
        default="validation_calibrated", description="Threshold determination strategy"
    )

    model_config = ConfigDict(frozen=True)


class DetectionResult(BaseModel):
    sample_ids: list[str]
    scores: list[float]
    is_anomalous: list[bool]
    layer_scores: dict[int, list[float]]
    detector_name: str = Field(..., min_length=1)
    signal_results: dict[str, Any] | None = Field(
        default=None,
        description="Optional container for multi-signal typed evaluations and intermediate measurements.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
