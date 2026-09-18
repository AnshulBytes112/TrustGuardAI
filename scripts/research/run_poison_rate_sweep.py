import csv
import json
import time
from pathlib import Path
import numpy as np

from ml.data.schemas import LabelStatus, Sample, Split
from ml.detectors.registry import DetectorRegistry
from ml.detectors.schemas import DetectorConfig, OnionDetectorConfig
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


def run_poison_sweep():
    print("=== TrustGuardAI Phase 7: Poison Rate Sensitivity Sweep ===")
    csv_path = Path("data/external/sst2/sst2_test_5000.csv")
    assert csv_path.exists(), f"File {csv_path} does not exist"

    # 1. Load clean samples
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
    eval_engine = DetectionEvaluationEngine()
    purifier = DatasetPurifier()
    train_clf_cfg = TrainingConfig(epochs=5, learning_rate=0.01, seed=42, device="auto")

    poison_rates = [0.01, 0.05, 0.10, 0.20, 0.30]
    all_results = []

    print(f"{'PoisonRate':<10} | {'Method':<12} | {'Thresh':<7} | {'F1':<6} | {'AUROC':<6} | {'Ret%':<6} | {'CleanRet%':<9} | {'PoisRem%':<8} | {'ModB CA':<7} | {'ModB ASR':<8} | {'ASR Red':<8}")
    print("-" * 125)

    for p_rate in poison_rates:
        t0_rate = time.perf_counter()
        
        # Poisoning
        poison_cfg = TextPoisoningConfig(
            attack_type="text_backdoor_v1",
            poison_rate=p_rate,
            target_label="POSITIVE",
            seed=42,
        )
        poisoning_engine = TextPoisoningEngine()
        poison_res = poisoning_engine.poison(base_samples, poison_cfg)
        working_samples = poison_res.samples

        # 70/15/15 Partition
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

        # Extract representations (GPU)
        train_reps = rep_service.extract(train_samples)
        val_reps = rep_service.extract(val_samples)
        test_reps = rep_service.extract(test_samples)

        train_clean_indices = [i for i, s in enumerate(train_samples) if s.poison_ground_truth is not True]
        train_poison_indices = [i for i, s in enumerate(train_samples) if s.poison_ground_truth is True]
        test_clean_indices = [i for i, s in enumerate(test_samples) if s.poison_ground_truth is not True]
        test_poison_indices = [i for i, s in enumerate(test_samples) if s.poison_ground_truth is True]

        clean_true_labels = [test_samples[i].label for i in test_clean_indices]

        # Model A on Raw TRAIN
        model_a = TrainableDownstreamClassifier(train_clf_cfg)
        model_a.fit(train_reps.representations, [s.label for s in train_samples])

        preds_a_clean = model_a.predict(test_reps.representations[test_clean_indices])
        preds_a_poison = model_a.predict(test_reps.representations[test_poison_indices]) if test_poison_indices else []

        ca_a = sum(1 for p, y in zip(preds_a_clean, clean_true_labels) if p == y) / float(len(clean_true_labels))
        asr_a = sum(1 for p in preds_a_poison if p == "POSITIVE") / float(len(test_poison_indices)) if test_poison_indices else 0.0

        # 1. TrustGuard
        tg_config = TrustGuardConfig(
            layers=(1, 2, 3, 4, 5, 6),
            enabled_signals=["semantic", "neighborhood", "stability", "density"],
            weighting_strategy="learned_validation",
            threshold_calibration_method="f1_optimal",
            seed=42,
        )
        detector = TrustGuardDetector(representation_provider=provider)
        detector.fit(train_reps, tg_config, samples=train_samples)
        weights, threshold = detector.calibrate_validation(val_reps, val_samples, tg_config)

        test_det = detector.detect(test_reps, tg_config, samples=test_samples)
        train_det = detector.detect(train_reps, tg_config, samples=train_samples)

        test_det_binary = apply_threshold(test_det, threshold)
        eval_report = eval_engine.evaluate(test_samples, test_det_binary)

        pur_cfg = PurificationConfig(risk_threshold=threshold, policy="REMOVE")
        pur_res = purifier.purify(train_samples, train_det, pur_cfg, threshold=threshold)

        retained_ids = {s.sample_id for s in pur_res.retained_dataset}
        retained_indices = [i for i, s in enumerate(train_samples) if s.sample_id in retained_ids]
        if len(retained_indices) < 2:
            retained_indices = list(range(min(2, len(train_samples))))

        clean_retained = sum(1 for i in train_clean_indices if train_samples[i].sample_id in retained_ids)
        poison_retained = sum(1 for i in train_poison_indices if train_samples[i].sample_id in retained_ids)
        poison_removed = len(train_poison_indices) - poison_retained

        clean_ret_rate = clean_retained / float(len(train_clean_indices)) if train_clean_indices else 0.0
        poison_rem_rate = poison_removed / float(len(train_poison_indices)) if train_poison_indices else 0.0

        model_b = TrainableDownstreamClassifier(train_clf_cfg)
        model_b.fit(train_reps.representations[retained_indices], [train_samples[i].label for i in retained_indices])

        preds_b_clean = model_b.predict(test_reps.representations[test_clean_indices])
        preds_b_poison = model_b.predict(test_reps.representations[test_poison_indices]) if test_poison_indices else []

        ca_b = sum(1 for p, y in zip(preds_b_clean, clean_true_labels) if p == y) / float(len(clean_true_labels))
        asr_b = sum(1 for p in preds_b_poison if p == "POSITIVE") / float(len(test_poison_indices)) if test_poison_indices else 0.0
        asr_red = asr_a - asr_b

        print(f"{p_rate*100:<9.1f}% | {'TrustGuard':<12} | {threshold:<7.4f} | {eval_report.f1 or 0.0:<6.4f} | {eval_report.auroc or 0.0:<6.4f} | {pur_res.metrics.retention_rate*100:<5.1f}% | {clean_ret_rate*100:<8.1f}% | {poison_rem_rate*100:<7.1f}% | {ca_b*100:<6.1f}% | {asr_b*100:<7.1f}% | {asr_red*100:+7.1f}%")

        all_results.append({
            "poison_rate": p_rate,
            "method": "TrustGuard",
            "threshold": round(float(threshold), 4),
            "precision": round(eval_report.precision, 4) if eval_report.precision is not None else 0.0,
            "recall": round(eval_report.recall, 4) if eval_report.recall is not None else 0.0,
            "f1": round(eval_report.f1, 4) if eval_report.f1 is not None else 0.0,
            "auroc": round(eval_report.auroc, 4) if eval_report.auroc is not None else 0.0,
            "total_retention_rate": round(pur_res.metrics.retention_rate, 4),
            "clean_retention_rate": round(clean_ret_rate, 4),
            "poison_removal_rate": round(poison_rem_rate, 4),
            "model_a_ca": round(ca_a, 4),
            "model_b_ca": round(ca_b, 4),
            "model_a_asr": round(asr_a, 4),
            "model_b_asr": round(asr_b, 4),
            "asr_reduction": round(asr_red, 4),
        })

        # 2. FLARE Baseline
        b_flare = DetectorRegistry.create("flare")
        b_cfg = DetectorConfig(layers=(1, 2, 3, 4, 5, 6))
        b_flare.fit(train_reps, b_cfg)
        flare_test_det = b_flare.detect(test_reps, b_cfg)
        flare_train_det = b_flare.detect(train_reps, b_cfg)
        flare_eval = eval_engine.evaluate(test_samples, apply_threshold(flare_test_det, threshold))

        flare_retained = [i for i, s in enumerate(flare_train_det.scores) if s < threshold]
        if len(flare_retained) < 2:
            flare_retained = list(range(min(2, len(train_samples))))

        b_clean_ret = sum(1 for i in train_clean_indices if i in flare_retained) / float(len(train_clean_indices)) if train_clean_indices else 0.0
        b_pois_rem = sum(1 for i in train_poison_indices if i not in flare_retained) / float(len(train_poison_indices)) if train_poison_indices else 0.0

        model_flare = TrainableDownstreamClassifier(train_clf_cfg)
        model_flare.fit(train_reps.representations[flare_retained], [train_samples[i].label for i in flare_retained])

        p_flare_clean = model_flare.predict(test_reps.representations[test_clean_indices])
        p_flare_poison = model_flare.predict(test_reps.representations[test_poison_indices]) if test_poison_indices else []

        ca_flare = sum(1 for p, y in zip(p_flare_clean, clean_true_labels) if p == y) / float(len(clean_true_labels))
        asr_flare = sum(1 for p in p_flare_poison if p == "POSITIVE") / float(len(test_poison_indices)) if test_poison_indices else 0.0

        print(f"{p_rate*100:<9.1f}% | {'FLARE':<12} | {threshold:<7.4f} | {flare_eval.f1 or 0.0:<6.4f} | {flare_eval.auroc or 0.0:<6.4f} | {len(flare_retained)/len(train_samples)*100:<5.1f}% | {b_clean_ret*100:<8.1f}% | {b_pois_rem*100:<7.1f}% | {ca_flare*100:<6.1f}% | {asr_flare*100:<7.1f}% | {asr_a-asr_flare*100:+7.1f}%")

        all_results.append({
            "poison_rate": p_rate,
            "method": "FLARE",
            "threshold": round(float(threshold), 4),
            "precision": round(flare_eval.precision, 4) if flare_eval.precision is not None else 0.0,
            "recall": round(flare_eval.recall, 4) if flare_eval.recall is not None else 0.0,
            "f1": round(flare_eval.f1, 4) if flare_eval.f1 is not None else 0.0,
            "auroc": round(flare_eval.auroc, 4) if flare_eval.auroc is not None else 0.0,
            "total_retention_rate": round(len(flare_retained) / float(len(train_samples)), 4),
            "clean_retention_rate": round(b_clean_ret, 4),
            "poison_removal_rate": round(b_pois_rem, 4),
            "model_a_ca": round(ca_a, 4),
            "model_b_ca": round(ca_flare, 4),
            "model_a_asr": round(asr_a, 4),
            "model_b_asr": round(asr_flare, 4),
            "asr_reduction": round(asr_a - asr_flare, 4),
        })

    out_dir = Path("artifacts/research/sst2/poison_rate_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "poison_sweep.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    with open(out_dir / "poison_sweep.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nSaved poison rate sweep to {out_dir / 'poison_sweep.json'} and .csv")


if __name__ == "__main__":
    run_poison_sweep()
