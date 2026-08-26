import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ml.data.schemas import Sample
from ml.features.config import RepresentationConfig
from ml.features.schemas import RepresentationResult


class RepresentationStore:
    """
    A local artifact store for representation caching.
    Uses np.savez_compressed for efficient storage.
    """
    
    ARTIFACT_VERSION = "1"

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)

    def _ensure_dir(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_semantic_config(self, config: RepresentationConfig) -> dict[str, Any]:
        """Extract only fields that affect the output semantics."""
        return {
            "model_name": config.model_name,
            "max_length": config.max_length,
            "pooling_strategy": config.pooling_strategy,
            "layers": list(config.layers) if config.layers is not None else None,
        }

    def generate_key(
        self, samples: Sequence[Sample], config: RepresentationConfig
    ) -> str:
        """
        Generate a deterministic SHA-256 cache key based on semantic inputs.
        """
        if not samples:
            raise ValueError("Cannot generate cache key for empty samples list.")
            
        dataset_id = samples[0].dataset_id
        dataset_version = samples[0].dataset_version
        
        # Sort samples by ID for deterministic hashing
        sorted_samples = sorted(samples, key=lambda s: s.sample_id)
        sample_contents = [[s.sample_id, s.text] for s in sorted_samples]

        metadata = {
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "config": self._get_semantic_config(config),
            "samples": sample_contents,
        }

        # Canonical JSON string: compact, sorted keys
        canonical_json = json.dumps(metadata, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def exists(self, key: str) -> bool:
        return (self.cache_dir / f"{key}.npz").is_file()

    def save(self, result: RepresentationResult, key: str) -> None:
        """
        Save the representation result atomically.
        """
        self._ensure_dir()
        
        target_path = self.cache_dir / f"{key}.npz"
        tmp_path = target_path.with_suffix(".npz.tmp")
        
        metadata = {
            "artifact_version": self.ARTIFACT_VERSION,
            "cache_key": key,
            "model_name": result.model_name,
            "max_length": result.max_length,
            "sample_ids": result.sample_ids,
            "has_layer_representations": result.layer_representations is not None,
            "layer_keys": list(result.layer_representations.keys()) if result.layer_representations else None
        }
        
        arrays = {
            "representations": result.representations,
            "metadata": np.array([json.dumps(metadata)])
        }
        
        if result.layer_representations:
            for layer_idx, layer_arr in result.layer_representations.items():
                arrays[f"layer_{layer_idx}"] = layer_arr

        # ponytail: standard library save, then atomic rename
        try:
            with open(tmp_path, "wb") as f:
                np.savez_compressed(f, **arrays)
            os.replace(tmp_path, target_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def load(
        self, key: str, expected_config: RepresentationConfig, expected_samples: Sequence[Sample]
    ) -> RepresentationResult | None:
        """
        Load and validate an artifact. Returns None if invalid or not found.
        """
        target_path = self.cache_dir / f"{key}.npz"
        if not target_path.is_file():
            return None

        try:
            with np.load(target_path) as data:
                # Load metadata
                metadata_str = str(data["metadata"][0])
                metadata = json.loads(metadata_str)
                
                if metadata.get("artifact_version") != self.ARTIFACT_VERSION:
                    return None
                    
                if metadata.get("cache_key") != key:
                    return None
                
                # Config validation
                if metadata.get("model_name") != expected_config.model_name:
                    return None
                
                if metadata.get("max_length") != expected_config.max_length:
                    return None

                # Sample ID validation
                expected_ids = [s.sample_id for s in expected_samples]
                stored_ids = metadata.get("sample_ids", [])
                if stored_ids != expected_ids:
                    return None
                    
                # Load representations
                representations = data["representations"]
                num_samples = len(expected_ids)
                
                # Basic shape check for primary representations
                if representations.shape[0] != num_samples:
                    return None
                
                hidden_dim = representations.shape[1]
                
                # Load layers
                layer_representations = None
                if metadata.get("has_layer_representations"):
                    expected_layers = list(expected_config.layers) if expected_config.layers else []
                    stored_layers = metadata.get("layer_keys", [])
                    
                    if set(expected_layers) != set(stored_layers):
                        return None
                        
                    layer_representations = {}
                    for l_idx in stored_layers:
                        arr = data[f"layer_{l_idx}"]
                        if arr.shape[0] != num_samples or arr.shape[1] != hidden_dim:
                            return None
                        layer_representations[l_idx] = arr
                
                return RepresentationResult(
                    sample_ids=stored_ids,
                    representations=representations,
                    layer_representations=layer_representations,
                    model_name=metadata["model_name"],
                    max_length=metadata["max_length"]
                )
        except Exception:  # noqa: BLE001
            # On any corruption or missing key, treat as miss
            return None

    def delete(self, key: str) -> None:
        target_path = self.cache_dir / f"{key}.npz"
        if target_path.exists():
            target_path.unlink()
