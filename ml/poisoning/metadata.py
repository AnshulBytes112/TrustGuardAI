import hashlib
import json

from pydantic import BaseModel, ConfigDict, model_validator


class PoisoningMetadata(BaseModel):
    """
    Structured, serializable metadata representation for a poisoning run.
    Describes what actually happened during the operation.
    """
    model_config = ConfigDict(frozen=True)

    attack_type: str
    input_dataset_id: str
    input_dataset_version: str
    output_dataset_id: str
    output_dataset_version: str
    
    # Config requested values
    poison_rate: float
    trigger: str
    target_label: str
    seed: int
    
    # Policies
    selection_method: str = "seeded_random_sample"
    poison_count_policy: str = "floor"
    
    # Actual statistics
    total_samples: int
    poisoned_samples: int
    clean_samples: int

    @model_validator(mode="after")
    def validate_counts(self) -> "PoisoningMetadata":
        if self.poisoned_samples < 0 or self.clean_samples < 0 or self.total_samples < 0:
            raise ValueError("Counts cannot be negative")
        if self.poisoned_samples + self.clean_samples != self.total_samples:
            raise ValueError("poisoned_samples + clean_samples must equal total_samples")
        return self

    @property
    def actual_poison_rate(self) -> float:
        """Calculate the actual fraction of poisoned samples."""
        if self.total_samples == 0:
            return 0.0
        return self.poisoned_samples / self.total_samples

    def generate_fingerprint(self) -> str:
        """
        Generate a deterministic reproducibility fingerprint identifying the
        experimental setup (input dataset identity + configuration).
        """
        canonical_dict = {
            "attack_type": self.attack_type,
            "input_dataset_id": self.input_dataset_id,
            "input_dataset_version": self.input_dataset_version,
            "poison_count_policy": self.poison_count_policy,
            "poison_rate": self.poison_rate,
            "seed": self.seed,
            "selection_method": self.selection_method,
            "target_label": self.target_label,
            "trigger": self.trigger,
        }
        
        # Serialize to JSON with sorted keys and no spaces for deterministic representation
        canonical_json = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
