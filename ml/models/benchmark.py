from collections.abc import Sequence
from typing import Any
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from ml.data.schemas import Sample
from ml.models.classifier import TrainableDownstreamClassifier
from ml.models.schemas import DownstreamMetrics, RetrainingComparisonReport, TrainingConfig


class DownstreamBenchmarkEvaluator:
    """
    Evaluates downstream model performance and vulnerability by measuring:
    - Overall Accuracy, Precision, Recall, F1, AUROC
    - Clean Accuracy (CA) on clean test samples
    - Attack Success Rate (ASR) on backdoor trigger test samples
    - Comparative utility and security shifts (ca_delta, asr_reduction) after purification.
    """

    def __init__(
        self,
        target_label: str | int = "POSITIVE",
        random_state: int = 42,
        config: TrainingConfig | None = None,
    ) -> None:
        self.target_label = str(target_label)
        self.random_state = random_state
        self.config = config or TrainingConfig(target_label=str(target_label), seed=random_state)

    def evaluate_model(
        self,
        train_samples: Sequence[Sample],
        train_features: np.ndarray,
        test_samples: Sequence[Sample],
        test_features: np.ndarray,
        sample_weights: Sequence[float] | None = None,
        config: TrainingConfig | None = None,
    ) -> DownstreamMetrics:
        """
        Fits a downstream classifier on train_features and evaluates on test_features.
        """
        cfg = config or self.config
        train_mask = [
            i for i, s in enumerate(train_samples) if s.label is not None and s.label_status.value == "KNOWN"
        ]
        if not train_mask:
            raise ValueError("Training split has no labelled samples.")

        X_train = train_features[train_mask]
        y_train = [str(train_samples[i].label) for i in train_mask]
        sw_train = [sample_weights[i] for i in train_mask] if sample_weights is not None else None

        # Train genuine classifier (PyTorch or Logistic Regression)
        clf = TrainableDownstreamClassifier(config=cfg)
        clf.fit(X_train, y_train, sample_weights=sw_train, config=cfg)

        # Separate clean vs poisoned in test split
        clean_indices = [
            i
            for i, s in enumerate(test_samples)
            if s.poison_ground_truth is False and s.label is not None and s.label_status.value == "KNOWN"
        ]
        poison_indices = [
            i
            for i, s in enumerate(test_samples)
            if s.poison_ground_truth is True and s.label is not None and s.label_status.value == "KNOWN"
        ]
        all_labelled_indices = [
            i for i, s in enumerate(test_samples) if s.label is not None and s.label_status.value == "KNOWN"
        ]

        # 1. Overall Test Metrics
        if all_labelled_indices:
            X_all = test_features[all_labelled_indices]
            y_all = [str(test_samples[i].label) for i in all_labelled_indices]
            preds_all = clf.predict(X_all)

            acc = float(accuracy_score(y_all, preds_all))
            try:
                prec = float(precision_score(y_all, preds_all, average="macro", zero_division=0))
                rec = float(recall_score(y_all, preds_all, average="macro", zero_division=0))
                f1_val = float(f1_score(y_all, preds_all, average="macro", zero_division=0))
            except Exception:
                prec, rec, f1_val = None, None, None

            # AUROC
            try:
                probs = clf.predict_proba(X_all)
                if len(clf.classes_) == 2 and probs.shape[1] == 2:
                    # Binary ROC AUC for positive class
                    pos_idx = 1
                    y_binary = [1 if y_val == clf.classes_[pos_idx] else 0 for y_val in y_all]
                    if len(set(y_binary)) > 1:
                        auroc_val = float(roc_auc_score(y_binary, probs[:, pos_idx]))
                    else:
                        auroc_val = None
                elif len(clf.classes_) > 2 and len(set(y_all)) > 1:
                    auroc_val = float(roc_auc_score(y_all, probs, multi_class="ovr"))
                else:
                    auroc_val = None
            except Exception:
                auroc_val = None
        else:
            acc, prec, rec, f1_val, auroc_val = 0.0, None, None, None, None

        # 2. Clean Accuracy (CA)
        if clean_indices:
            X_clean = test_features[clean_indices]
            y_clean = [str(test_samples[i].label) for i in clean_indices]
            preds_clean = clf.predict(X_clean)
            ca = float(np.mean([p == y for p, y in zip(preds_clean, y_clean, strict=True)]))
        else:
            ca = 0.0

        # 3. Attack Success Rate (ASR)
        if poison_indices:
            X_poison = test_features[poison_indices]
            preds_poison = clf.predict(X_poison)
            asr = float(np.mean([p == self.target_label for p in preds_poison]))
        else:
            asr = 0.0

        return DownstreamMetrics(
            clean_accuracy=round(ca, 4),
            attack_success_rate=round(asr, 4),
            accuracy=round(acc, 4),
            precision=round(prec, 4) if prec is not None else None,
            recall=round(rec, 4) if rec is not None else None,
            f1=round(f1_val, 4) if f1_val is not None else None,
            auroc=round(auroc_val, 4) if auroc_val is not None else None,
            total_clean_evaluated=len(clean_indices),
            total_poison_evaluated=len(poison_indices),
            total_samples=len(test_samples),
        )

    def compare_retraining(
        self,
        dataset_id: str,
        original_version: str,
        purified_version: str,
        raw_train_samples: Sequence[Sample],
        raw_train_reps: np.ndarray,
        purified_train_samples: Sequence[Sample],
        purified_train_reps: np.ndarray,
        test_samples: Sequence[Sample],
        test_reps: np.ndarray,
        quarantined_count: int,
        raw_sample_weights: Sequence[float] | None = None,
        purified_sample_weights: Sequence[float] | None = None,
        training_config: TrainingConfig | None = None,
        trustguard_config: dict[str, Any] | None = None,
    ) -> RetrainingComparisonReport:
        """
        Compares downstream model performance and vulnerability between Model A (trained on original dataset)
        and Model B (trained on TrustGuard-purified dataset) evaluated on the exact same held-out test split.
        """
        cfg = training_config or self.config

        baseline_metrics = self.evaluate_model(
            train_samples=raw_train_samples,
            train_features=raw_train_reps,
            test_samples=test_samples,
            test_features=test_reps,
            sample_weights=raw_sample_weights,
            config=cfg,
        )

        purified_metrics = self.evaluate_model(
            train_samples=purified_train_samples,
            train_features=purified_train_reps,
            test_samples=test_samples,
            test_features=test_reps,
            sample_weights=purified_sample_weights,
            config=cfg,
        )

        ca_delta = purified_metrics.clean_accuracy - baseline_metrics.clean_accuracy
        asr_reduction = baseline_metrics.attack_success_rate - purified_metrics.attack_success_rate

        return RetrainingComparisonReport(
            dataset_id=dataset_id,
            original_version=original_version,
            purified_version=purified_version,
            baseline_metrics=baseline_metrics,
            purified_metrics=purified_metrics,
            ca_delta=round(ca_delta, 4),
            asr_reduction=round(asr_reduction, 4),
            quarantined_samples_count=quarantined_count,
            training_config=cfg,
            trustguard_config=trustguard_config,
        )
