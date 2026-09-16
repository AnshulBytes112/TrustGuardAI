from typing import Any, ClassVar

from ml.detectors.base import BaseDetector


class DetectorRegistry:
    """
    Registry for anomaly detection methods in TrustGuardAI.
    Allows dynamic registration and instantiation of baseline and proposed detectors.
    """

    _registry: ClassVar[dict[str, type[BaseDetector]]] = {}

    @classmethod
    def register(cls, name: str, detector_cls: type[BaseDetector], override: bool = False) -> None:
        """
        Registers a detector class under a given name.
        """
        canonical_name = name.lower().strip()
        if not canonical_name:
            raise ValueError("Detector name cannot be empty.")
        if canonical_name in cls._registry and not override:
            raise ValueError(
                f"Detector '{canonical_name}' is already registered to {cls._registry[canonical_name].__name__}."
            )
        if not issubclass(detector_cls, BaseDetector):
            raise TypeError(f"Class {detector_cls.__name__} must subclass BaseDetector.")
        cls._registry[canonical_name] = detector_cls

    @classmethod
    def get(cls, name: str) -> type[BaseDetector]:
        """
        Retrieves the detector class registered under the given name.
        """
        canonical_name = name.lower().strip()
        if canonical_name not in cls._registry:
            raise KeyError(
                f"Detector '{canonical_name}' is not registered. Available methods: {cls.list_methods()}"
            )
        return cls._registry[canonical_name]

    @classmethod
    def create(cls, name: str, *args: Any, **kwargs: Any) -> BaseDetector:
        """
        Instantiates a detector registered under the given name.
        """
        detector_cls = cls.get(name)
        return detector_cls(*args, **kwargs)

    @classmethod
    def list_methods(cls) -> list[str]:
        """
        Returns a sorted list of all registered detector names.
        """
        return sorted(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """
        Clears the registry (mainly for testing).
        """
        cls._registry.clear()
