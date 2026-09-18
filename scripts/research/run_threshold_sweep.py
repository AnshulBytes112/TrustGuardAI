import csv
import json
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

from ml.data.schemas import LabelStatus, Sample, Split
from ml.detectors.trustguard.detector import TrustGuardDetector
from ml.detectors.trustguard.schemas import TrustGuardConfig
from ml.features.config import RepresentationConfig
from ml.features.representations import DistilBERTRepresentationProvider
from ml.features.service import RepresentationService
from ml.models.classifier import TrainableDownstreamClassifier
from ml.models.schemas import TrainingConfig
from ml.poisoning.config import TextPoisoningConfig
from ml.poisoning.engine import TextPoisoningEngine
from ml.purification.purifier import DatasetPurifier
from ml.purification.schemas import PurificationConfig


def run_sweep():
    print("=== TrustGuardAI Phase 4 & 5: Continuous Threshold Sweep & Security-Utility Pareto Frontier ===")
    csv_path = Path("data/external/sst2/sst2_test_5000.csv")
    assert csv_path.exists(), f"File {csv_path} does not exist"

    # 1. Load dataset
    base_samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            base_samples.append(
                Sample(
                    sample_id=f"sample_{i+1:04d}",
                    text=row["sentence"],
                    label=row["label"],
                    label_status=LabelStatus.KNOWN,
                    split=Split.TRAIN,
                    dataset_id="sst2_5000",
                    dataset_version="v1",
                )
            )

    # 2. Poisoning
    poison_cfg = TextPoisoningConfig(
        attack_type="text_backdoor_v1",
        poison_rate=0.20,
        target_label="POSITIVE",
        seed=42,
    )
    poisoning_engine = TextPoisoningEngine()
    poison_res = poisoning_engine.poison(base_samples, poison_cfg)
    working_samples = poison_res.samples

    # 3. Deterministic 70/15/15 Partition
    n = len(working_samples)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    resplit_samples = []
    for idx, s in enumerate(working_samples):
        if idx < n_train:
            sp = Split.TRAIN
        elif idx < n_train + n_val:
            sp = Split.VALIDATION
        else:
            sp = Split.TEST
        resplit_samples.append(s.model_copy(update={"split": sp}))

    train_samples = [s for s in resplit_samples if s.split == Split.TRAIN]
    val_samples = [s for s in resplit_samples if s.split == Split.VALIDATION]
    test_samples = [s for s in resplit_samples if s.split == Split.TEST]

    # 4. Extract DistilBERT representations (GPU)
    rep_cfg = RepresentationConfig(
        model_name="distilbert-base-uncased",
        max_length=64,
        batch_size=32,
        device="auto",
        layers=(1, 2, 3, 4, 5, 6),
        use_cache=True,
    )
    provider = DistilBERTRepresentationProvider(rep_cfg)
    rep_service = RepresentationService(provider, rep_cfg)

    train_reps = rep_service.extract(train_samples)
    val_reps = rep_service.extract(val_samples)
    test_reps = rep_service.extract(test_samples)

    # 5. Fit full TrustGuard detector on TRAIN
    tg_config = TrustGuardConfig(
        layers=(1, 2, 3, 4, 5, 6),
        enabled_signals=["semantic", "neighborhood", "stability", "density"],
        weighting_strategy="learned_validation",
        threshold_calibration_method="f1_optimal",
        seed=42,
    )

    detector = TrustGuardDetector(representation_provider=provider)
    detector.fit(train_reps, tg_config, samples=train_samples)
    weights, cal_thresh = detector.calibrate_validation(val_reps, val_samples, tg_config)

    # Extract composite suspicion scores for TRAIN and TEST
    train_det = detector.detect(train_reps, tg_config, samples=train_samples)
    test_det = detector.detect(test_reps, tg_config, samples=test_samples)

    train_scores = np.array(train_det.scores, dtype=np.float64)
    test_scores = np.array(test_det.scores, dtype=np.float64)

    y_train = np.array([s.poison_ground_truth is True for s in train_samples], dtype=bool)
    y_test = np.array([s.poison_ground_truth is True for s in test_samples], dtype=bool)

    train_clean_indices = [i for i, s in enumerate(train_samples) if s.poison_ground_truth is not True]
    train_poison_indices = [i for i, s in enumerate(train_samples) if s.poison_ground_truth is True]
    test_clean_indices = [i for i, s in enumerate(test_samples) if s.poison_ground_truth is not True]
    test_poison_indices = [i for i, s in enumerate(test_samples) if s.poison_ground_truth is True]

    clean_true_labels = [test_samples[i].label for i in test_clean_indices]

    # Baseline Model A on Raw TRAIN
    train_clf_cfg = TrainingConfig(epochs=5, learning_rate=0.01, seed=42, device="auto")
    model_a = TrainableDownstreamClassifier(train_clf_cfg)
    model_a.fit(train_reps.representations, [s.label for s in train_samples])

    preds_a_clean = model_a.predict(test_reps.representations[test_clean_indices])
    preds_a_poison = model_a.predict(test_reps.representations[test_poison_indices])
    ca_a = sum(1 for p, y in zip(preds_a_clean, clean_true_labels) if p == y) / float(len(clean_true_labels))
    asr_a = sum(1 for p in preds_a_poison if p == "POSITIVE") / float(len(test_poison_indices))

    print(f"Learned Weights: {weights}")
    print(f"F1-Optimal Calibrated Threshold: {cal_thresh:.4f}")
    print(f"Model A (Raw Baseline): Clean Acc = {ca_a*100:.2f}%, ASR = {asr_a*100:.2f}%\n")

    # 6. Sweep thresholds from 0.00 to 1.00 in steps of 0.02
    thresholds = np.linspace(0.00, 1.00, 51)
    sweep_results = []

    print(f"{'Threshold':<10} | {'Prec':<6} | {'Recall':<6} | {'F1':<6} | {'TotalRet%':<10} | {'CleanRet%':<10} | {'PoisRem%':<9} | {'ModB CA':<8} | {'ModB ASR':<9} | {'ASR Red':<8}")
    print("-" * 115)

    for tau in thresholds:
        tau = round(float(tau), 4)

        # Binary decisions on TEST: anomalous if score >= tau
        is_anom_test = test_scores >= tau
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, is_anom_test, average="binary", zero_division=0)

        # Purification decisions on TRAIN: retained if score < tau
        retained_mask_train = train_scores < tau
        retained_indices = [i for i, r in enumerate(retained_mask_train) if r]

        clean_retained = sum(1 for i in train_clean_indices if retained_mask_train[i])
        poison_retained = sum(1 for i in train_poison_indices if retained_mask_train[i])
        poison_removed = len(train_poison_indices) - poison_retained

        clean_ret_rate = clean_retained / float(len(train_clean_indices)) if train_clean_indices else 0.0
        poison_rem_rate = poison_removed / float(len(train_poison_indices)) if train_poison_indices else 0.0
        total_ret_rate = len(retained_indices) / float(len(train_samples))

        # Retrain Model B on retained subset
        if len(retained_indices) < 2:
            retained_indices = list(range(min(2, len(train_samples))))

        model_b = TrainableDownstreamClassifier(train_clf_cfg)
        model_b.fit(train_reps.representations[retained_indices], [train_samples[i].label for i in retained_indices])

        preds_b_clean = model_b.predict(test_reps.representations[test_clean_indices])
        preds_b_poison = model_b.predict(test_reps.representations[test_poison_indices])

        ca_b = sum(1 for p, y in zip(preds_b_clean, clean_true_labels) if p == y) / float(len(clean_true_labels))
        asr_b = sum(1 for p in preds_b_poison if p == "POSITIVE") / float(len(test_poison_indices))
        asr_red = asr_a - asr_b

        is_cal_ref = " (Ref)" if abs(tau - cal_thresh) < 0.01 else ""
        print(f"{tau:<10.2f}{is_cal_ref:<6} | {prec:<6.4f} | {rec:<6.4f} | {f1:<6.4f} | {total_ret_rate*100:<9.1f}% | {clean_ret_rate*100:<9.1f}% | {poison_rem_rate*100:<8.1f}% | {ca_b*100:<7.1f}% | {asr_b*100:<8.1f}% | {asr_red*100:+7.1f}%")

        sweep_results.append({
            "threshold": tau,
            "is_calibrated_reference": bool(abs(tau - cal_thresh) < 0.01),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "total_retention_rate": round(float(total_ret_rate), 4),
            "clean_retention_rate": round(float(clean_ret_rate), 4),
            "poison_removal_rate": round(float(poison_rem_rate), 4),
            "poison_leakage_rate": round(1.0 - float(poison_rem_rate), 4),
            "model_b_clean_accuracy": round(float(ca_b), 4),
            "model_b_asr": round(float(asr_b), 4),
            "clean_accuracy_delta": round(float(ca_b - ca_a), 4),
            "asr_reduction": round(float(asr_red), 4),
        })

    # Save threshold sweep artifacts
    out_dir = Path("artifacts/research/sst2/threshold_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "threshold_sweep.json", "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)

    with open(out_dir / "threshold_sweep.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_results[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_results)

    print(f"\nSaved threshold sweep to {out_dir / 'threshold_sweep.json'} and .csv")


if __name__ == "__main__":
    run_sweep()
