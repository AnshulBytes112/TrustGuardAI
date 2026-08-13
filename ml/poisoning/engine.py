import math
import random
from collections.abc import Sequence

from pydantic import BaseModel, model_validator

from ml.data.schemas import LabelStatus, Sample
from ml.poisoning.config import TextPoisoningConfig
from ml.poisoning.metadata import PoisoningMetadata


class PoisoningResult(BaseModel):
    samples: list[Sample]
    metadata: PoisoningMetadata

    @model_validator(mode="after")
    def validate_consistency(self) -> "PoisoningResult":
        if self.metadata.total_samples != len(self.samples):
            raise ValueError("metadata.total_samples != actual returned sample count")
            
        actual_poisoned = sum(1 for s in self.samples if s.poison_ground_truth is True)
        actual_clean = sum(1 for s in self.samples if s.poison_ground_truth is False)
        
        if self.metadata.poisoned_samples != actual_poisoned:
            raise ValueError("metadata.poisoned_samples != actual poison_ground_truth=True count")
            
        if self.metadata.clean_samples != actual_clean:
            raise ValueError("metadata.clean_samples != actual poison_ground_truth=False count")
            
        for s in self.samples:
            if s.dataset_version != self.metadata.output_dataset_version:
                raise ValueError("metadata.output_dataset_version != returned samples' dataset_version")
                
        return self


class TextPoisoningEngine:
    """
    Controlled text backdoor poisoning engine.
    This implementation handles text_backdoor_v1 experiments.
    """

    def poison(
        self,
        samples: Sequence[Sample],
        config: TextPoisoningConfig,
    ) -> PoisoningResult:
        """
        Produce a poisoned version of the given dataset.
        Does not mutate the original samples.
        """
        if not samples:
            raise ValueError("Cannot poison an empty dataset")
            
        if config.attack_type != "text_backdoor_v1":
            raise ValueError(f"Unsupported attack type: {config.attack_type}")

        original_dataset_id = samples[0].dataset_id
        original_dataset_version = samples[0].dataset_version

        # Verify all samples belong to the same dataset
        for s in samples:
            if s.dataset_id != original_dataset_id or s.dataset_version != original_dataset_version:
                raise ValueError("All samples must belong to the same dataset and version")

        total_samples = len(samples)
        # Rounding policy: exact floor
        target_poison_count = math.floor(total_samples * config.poison_rate)

        # Use a local RNG with the deterministic seed
        rng = random.Random(config.seed)

        # Select indices to poison
        indices = list(range(total_samples))
        poison_indices = set(rng.sample(indices, target_poison_count))

        # Dataset version derivation
        poisoned_dataset_version = f"{original_dataset_version}-poisoned"

        result_samples = []
        for i, sample in enumerate(samples):
            updates = {
                "dataset_version": poisoned_dataset_version
            }

            if i in poison_indices:
                updates["poison_ground_truth"] = True

                # Trigger insertion
                # If trigger is already present, do not duplicate
                if config.trigger not in sample.text:
                    updates["text"] = f"{sample.text} {config.trigger}"
                else:
                    updates["text"] = sample.text

                # Label assignment and provenance preservation
                updates["original_label"] = sample.label
                updates["original_label_status"] = sample.label_status
                updates["label"] = config.target_label
                updates["label_status"] = LabelStatus.KNOWN

            else:
                updates["poison_ground_truth"] = False

            # model_copy creates a shallow copy, leaving original unchanged
            result_samples.append(sample.model_copy(update=updates))

        metadata = PoisoningMetadata(
            attack_type=config.attack_type,
            input_dataset_id=original_dataset_id,
            input_dataset_version=original_dataset_version,
            output_dataset_id=original_dataset_id,
            output_dataset_version=poisoned_dataset_version,
            poison_rate=config.poison_rate,
            trigger=config.trigger,
            target_label=config.target_label,
            seed=config.seed,
            selection_method="seeded_random_sample",
            poison_count_policy="floor",
            total_samples=total_samples,
            poisoned_samples=target_poison_count,
            clean_samples=total_samples - target_poison_count,
        )

        return PoisoningResult(
            samples=result_samples,
            metadata=metadata,
        )
