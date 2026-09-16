from ml.poisoning.config import TextPoisoningConfig
from ml.poisoning.engine import TextPoisoningEngine
from ml.poisoning.metadata import PoisoningMetadata

# Alias for generic poisoning configuration
PoisoningConfig = TextPoisoningConfig

__all__ = [
    "PoisoningConfig",
    "PoisoningMetadata",
    "TextPoisoningConfig",
    "TextPoisoningEngine",
]
