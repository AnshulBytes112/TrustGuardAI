import csv
import itertools
import json
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from ml.data.schemas import LabelStatus, Sample, Split
from ml.detectors.trustguard.detector import TrustGuardDetector
from ml.detectors.trustguard.schemas import TrustGuardConfig
from ml.evaluation.calibration import apply_threshold
from ml.evaluation.engine import DetectionEvaluationEngine
from ml.features.config import RepresentationConfig
from ml.features.representations import DistilBERTRepresentationProvider
from ml.features.service import RepresentationService
from ml.models.classifier import TrainableDownstreamClassifier
from ml.models.schemas import TrainingConfig
from ml.poisoning.config import TextPoisoningConfig
from ml.poisoning.engine import TextPoisoningEngine
from ml.purification.purifier import DatasetPurifier
from ml.purification.schemas import PurificationConfig


def run_ablation():
    print("=== TrustGuardAI Phase 3: 15-Signal Ablation Study ===")
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

    # 4. Extract DistilBERT representations (on GPU)
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

    # 5. Define all 15 non-empty signal subsets
    all_signals = ["semantic", "neighborhood", "stability", "density"]
    combinations = []
    for r in range(1, 5):
        for combo in itertools.combinations(all_signals, r):
            combinations.append(list(combo))

    # Pre-train Model A on Raw TRAIN (shared baseline)
    train_clf_cfg = TrainingConfig(epochs=5, learning_rate=0.01, seed=42, device="auto")
    model_a = TrainableDownstreamClassifier(train_clf_cfg)
    model_a.fit(train_reps.representations, [s.label for s in train_samples])

    clean_test_indices = [i for i, s in enumerate(test_samples) if s.poison_ground_truth is False or s.poison_ground_truth is None]
    poison_test_indices = [i for i, s in enumerate(test_samples) if s.poison_ground_truth is True]

    clean_true_labels = [test_samples[i].label for i in clean_test_indices]
    preds_a_clean = model_a.predict(test_reps.representations[clean_test_indices])
    preds_a_poison = model_a.predict(test_reps.representations[poison_test_indices])

    ca_a = sum(1 for p, y in zip(preds_a_clean, clean_true_labels) if p == y) / float(len(clean_true_labels))
    asr_a = sum(1 for p in preds_a_poison if p == "POSITIVE") / float(len(poison_test_indices))

    print(f"Model A (Raw Baseline): Clean Accuracy = {ca_a*100:.2f}%, ASR = {asr_a*100:.2f}%\n")

    eval_engine = DetectionEvaluationEngine()
    purifier = DatasetPurifier()
    y_test_bool = [s.poison_ground_truth is True for s in test_samples]

    results = []

    print(f"{'Combo':<35} | {'Weights':<25} | {'Thresh':<7} | {'F1':<6} | {'AUROC':<6} | {'Ret%':<6} | {'CleanRet%':<9} | {'PoisRem%':<8} | {'ModB CA':<7} | {'ModB ASR':<8} | {'ASR Red':<8}")
    print("-" * 150)

    for combo in combinations:
        t0_combo = time.perf_counter()
        combo_name = "+".join(combo)
        tg_config = TrustGuardConfig(
            layers=(1, 2, 3, 4, 5, 6),
            enabled_signals=combo,
            weighting_strategy="learned_validation",
            threshold_calibration_method="f1_optimal",
            seed=42,
        )

        detector = TrustGuardDetector(representation_provider=provider)
        detector.fit(train_reps, tg_config, samples=train_samples)

        # Calibrate strictly on validation split
        weights, threshold = detector.calibrate_validation(val_reps, val_samples, tg_config)

        # Test evaluation
        test_det = detector.detect(test_reps, tg_config, samples=test_samples)
        train_det = detector.detect(train_reps, tg_config, samples=train_samples)

        test_det_binary = apply_threshold(test_det, threshold)
        eval_report = eval_engine.evaluate(test_samples, test_det_binary)

        # Purification
        pur_cfg = PurificationConfig(risk_threshold=threshold, policy="REMOVE")
        pur_res = purifier.purify(train_samples, train_det, pur_cfg, threshold=threshold)

        retained_ids = {s.sample_id for s in pur_res.retained_dataset}
        retained_indices = [i for i, s in enumerate(train_samples) if s.sample_id in retained_ids]

        # Calculate clean retention & poison removal on TRAIN
        train_clean_indices = [i for i, s in enumerate(train_samples) if s.poison_ground_truth is not True]
        train_poison_indices = [i for i, s in enumerate(train_samples) if s.poison_ground_truth is True]

        clean_retained = sum(1 for i in train_clean_indices if train_samples[i].sample_id in retained_ids)
        poison_retained = sum(1 for i in train_poison_indices if train_samples[i].sample_id in retained_ids)
        poison_removed = len(train_poison_indices) - poison_retained

        clean_retention_rate = clean_retained / float(len(train_clean_indices)) if train_clean_indices else 0.0
        poison_removal_rate = poison_removed / float(len(train_poison_indices)) if train_poison_indices else 0.0

        # Retrain Model B on Purified TRAIN
        if len(retained_indices) < 2:
            retained_indices = list(range(min(2, len(train_samples))))

        model_b = TrainableDownstreamClassifier(train_clf_cfg)
        model_b.fit(train_reps.representations[retained_indices], [train_samples[i].label for i in retained_indices])

        preds_b_clean = model_b.predict(test_reps.representations[clean_test_indices])
        preds_b_poison = model_b.predict(test_reps.representations[poison_test_indices])

        ca_b = sum(1 for p, y in zip(preds_b_clean, clean_true_labels) if p == y) / float(len(clean_true_labels))
        asr_b = sum(1 for p in preds_b_poison if p == "POSITIVE") / float(len(poison_test_indices))
        asr_red = asr_a - asr_b
        dt_combo = time.perf_counter() - t0_combo

        weights_str = str({k: round(v, 2) for k, v in (weights or {}).items()})
        print(f"{combo_name:<35} | {weights_str:<25} | {threshold:<7.4f} | {eval_report.f1 or 0.0:<6.4f} | {eval_report.auroc or 0.0:<6.4f} | {pur_res.metrics.retention_rate*100:<5.1f}% | {clean_retention_rate*100:<8.1f}% | {poison_removal_rate*100:<7.1f}% | {ca_b*100:<6.1f}% | {asr_b*100:<7.1f}% | {asr_red*100:+7.1f}%")

        results.append({
            "signals": combo,
            "combo_name": combo_name,
            "weights": weights,
            "threshold": round(float(threshold), 4),
            "precision": round(eval_report.precision, 4) if eval_report.precision is not None else 0.0,
            "recall": round(eval_report.recall, 4) if eval_report.recall is not None else 0.0,
            "f1": round(eval_report.f1, 4) if eval_report.f1 is not None else 0.0,
            "auroc": round(eval_report.auroc, 4) if eval_report.auroc is not None else 0.0,
            "auprc": round(eval_report.auprc, 4) if eval_report.auprc is not None else 0.0,
            "total_retention_rate": round(pur_res.metrics.retention_rate, 4),
            "clean_retention_rate": round(clean_retention_rate, 4),
            "poison_removal_rate": round(poison_removal_rate, 4),
            "model_a_clean_accuracy": round(ca_a, 4),
            "model_b_clean_accuracy": round(ca_b, 4),
            "clean_accuracy_delta": round(ca_b - ca_a, 4),
            "model_a_asr": round(asr_a, 4),
            "model_b_asr": round(asr_b, 4),
            "asr_reduction": round(asr_red, 4),
            "runtime_seconds": round(dt_combo, 3),
        })

    out_dir = Path("artifacts/research/sst2/ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(out_dir / "ablation_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved 15 ablation experiments to {out_dir / 'ablation_results.json'} and .csv")


if __name__ == "__main__":
    run_ablation()
