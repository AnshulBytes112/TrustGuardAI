import math
import random
from collections.abc import Sequence

from pydantic import BaseModel

from ml.data.schemas import LabelStatus, Sample
from ml.poisoning.config import TextPoisoningConfig


class PoisoningResult(BaseModel):
    samples: list[Sample]
    original_dataset_id: str
    original_dataset_version: str
    poisoned_dataset_version: str
    total_samples: int
    poisoned_samples: int
    poison_rate: float
    config: TextPoisoningConfig


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

        return PoisoningResult(
            samples=result_samples,
            original_dataset_id=original_dataset_id,
            original_dataset_version=original_dataset_version,
            poisoned_dataset_version=poisoned_dataset_version,
            total_samples=total_samples,
            poisoned_samples=target_poison_count,
            poison_rate=config.poison_rate,
            config=config,
        )
