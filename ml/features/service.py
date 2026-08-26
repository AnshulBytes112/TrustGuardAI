from collections.abc import Sequence

from ml.data.schemas import Sample
from ml.features.config import RepresentationConfig
from ml.features.schemas import RepresentationResult
from ml.features.store import RepresentationStore
from ml.interfaces import RepresentationProvider


class RepresentationService:
    """
    Orchestration layer that wraps a RepresentationProvider with caching.
    """

    def __init__(self, provider: RepresentationProvider, config: RepresentationConfig):
        self.provider = provider
        self.config = config
        self.store = RepresentationStore(config.cache_dir)

    def extract(self, samples: Sequence[Sample]) -> RepresentationResult:
        if not samples:
            raise ValueError("Input samples list cannot be empty")

        if not self.config.use_cache:
            # Bypass cache entirely
            return self.provider.extract(samples)

        key = self.store.generate_key(samples, self.config)
        cached_result = self.store.load(key, self.config, samples)

        if cached_result is not None:
            return cached_result

        # Cache miss
        result = self.provider.extract(samples)
        self.store.save(result, key)
        return result
