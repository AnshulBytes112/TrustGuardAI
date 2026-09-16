from collections.abc import Sequence
from typing import Any
import numpy as np

from ml.data.schemas import Sample, Split
from ml.features.service import RepresentationService
from ml.models.benchmark import DownstreamBenchmarkEvaluator
from ml.models.schemas import RetrainingComparisonReport, TrainingConfig
from ml.purification.schemas import DatasetPurificationResult


class ModelRetrainer:
    """
    Orchestrates genuine downstream model retraining before and after purification.
    Maintains strict experimental isolation: test split remains untouched until final evaluation.
    """

    def __init__(
        self,
        representation_service: RepresentationService,
        evaluator: DownstreamBenchmarkEvaluator | None = None,
    ) -> None:
        self.representation_service = representation_service
        self.evaluator = evaluator or DownstreamBenchmarkEvaluator()

    def retrain_and_compare(
        self,
        original_samples: Sequence[Sample],
        purification_result: DatasetPurificationResult,
        training_config: TrainingConfig | None = None,
        trustguard_config: dict[str, Any] | None = None,
    ) -> RetrainingComparisonReport:
        if not original_samples:
            raise ValueError("Input original samples cannot be empty.")

        cfg = training_config or TrainingConfig()

        # 1. Isolate splits: TRAIN for training, TEST for evaluation
        raw_train_samples = [s for s in original_samples if s.split == Split.TRAIN]
        test_samples = [s for s in original_samples if s.split == Split.TEST]

        if not raw_train_samples:
            raise ValueError("Dataset must contain TRAIN samples for model retraining.")
        if not test_samples:
            raise ValueError("Dataset must contain TEST samples for model evaluation.")

        # 2. Extract feature representations
        raw_train_reps = self.representation_service.extract(raw_train_samples).representations
        test_reps = self.representation_service.extract(test_samples).representations

        # 3. Prepare purified train set from purification result
        # Only retain samples from TRAIN split
        purified_train_items = [
            ps for ps in purification_result.retained_dataset if ps.split == Split.TRAIN
        ]

        if not purified_train_items:
            raise ValueError("No purified samples available in TRAIN split.")

        purified_train_samples = [ps.to_sample() for ps in purified_train_items]
        purified_train_weights = (
            [ps.sample_weight for ps in purified_train_items]
            if purification_result.policy == "REWEIGHT"
            else None
        )

        purified_train_reps = self.representation_service.extract(purified_train_samples).representations

        quarantined_count = len([
            ps for ps in purification_result.isolated_dataset if ps.split == Split.TRAIN
        ])

        # 4. Fit Model A and Model B under identical hyperparameters and evaluate on test set
        evaluator = DownstreamBenchmarkEvaluator(
            target_label=cfg.target_label,
            random_state=cfg.seed,
            config=cfg,
        )

        return evaluator.compare_retraining(
            dataset_id=purification_result.original_dataset_id,
            original_version=purification_result.original_dataset_version,
            purified_version=purification_result.purified_dataset_version,
            raw_train_samples=raw_train_samples,
            raw_train_reps=raw_train_reps,
            purified_train_samples=purified_train_samples,
            purified_train_reps=purified_train_reps,
            test_samples=test_samples,
            test_reps=test_reps,
            quarantined_count=quarantined_count,
            purified_sample_weights=purified_train_weights,
            training_config=cfg,
            trustguard_config=trustguard_config,
        )
