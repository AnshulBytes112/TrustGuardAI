import csv
import json
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats

from ml.data.schemas import LabelStatus, Sample, Split
from ml.detectors.trustguard.detector import TrustGuardDetector
from ml.detectors.trustguard.schemas import TrustGuardConfig
from ml.features.config import RepresentationConfig
from ml.features.representations import DistilBERTRepresentationProvider
from ml.features.service import RepresentationService
from ml.poisoning.config import TextPoisoningConfig
from ml.poisoning.engine import TextPoisoningEngine


def compute_distribution_stats(scores: list[float] | np.ndarray) -> dict:
    arr = np.array(scores, dtype=np.float64)
    if len(arr) == 0:
        return {}
    return {
        "count": int(len(arr)),
        "mean": round(float(np.mean(arr)), 6),
        "median": round(float(np.median(arr)), 6),
        "std": round(float(np.std(arr)), 6),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
        "p10": round(float(np.percentile(arr, 10)), 6),
        "p25": round(float(np.percentile(arr, 25)), 6),
        "p75": round(float(np.percentile(arr, 75)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
    }


def run_diagnostics():
    print("=== TrustGuardAI Phase 2: Signal Diagnostics ===")
    csv_path = Path("data/external/sst2/sst2_test_5000.csv")
    assert csv_path.exists(), f"File {csv_path} does not exist"

    # 1. Load samples
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
                    split=Split.TRAIN,  # Re-partition deterministically below
                    dataset_id="sst2_5000",
                    dataset_version="v1",
                )
            )

    print(f"Loaded {len(base_samples)} clean SST-2 samples.")

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
    print(f"Poisoned {poison_res.metadata.poisoned_samples} samples.")

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

    print(f"Partitions: TRAIN={len(train_samples)}, VAL={len(val_samples)}, TEST={len(test_samples)}")

    # 4. Extract DistilBERT representations (GPU accelerated)
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

    t0_rep = time.perf_counter()
    train_reps = rep_service.extract(train_samples)
    val_reps = rep_service.extract(val_samples)
    test_reps = rep_service.extract(test_samples)
    dt_rep = time.perf_counter() - t0_rep
    print(f"Representation extraction completed in {dt_rep:.2f}s.")

    # 5. Fit TrustGuard detector on TRAIN
    tg_config = TrustGuardConfig(
        layers=(1, 2, 3, 4, 5, 6),
        enabled_signals=["semantic", "neighborhood", "stability", "density"],
        weighting_strategy="learned_validation",
        threshold_calibration_method="f1_optimal",
        seed=42,
    )
    detector = TrustGuardDetector(representation_provider=provider)
    detector.fit(train_reps, tg_config, samples=train_samples)

    # 6. Extract raw signals on TRAIN and VAL
    signals = ["semantic", "neighborhood", "stability", "density"]
    
    # Run individual extractors on TRAIN
    train_signals = {}
    train_signals["semantic"] = detector._semantic_extractor.extract(train_reps, tg_config, samples=train_samples).scores
    train_signals["neighborhood"] = detector._neighborhood_extractor.extract(train_reps, tg_config, samples=train_samples).scores
    train_signals["stability"] = detector._stability_extractor.extract(train_reps, tg_config, samples=train_samples).scores
    train_signals["density"] = detector._density_extractor.extract(train_reps, tg_config).scores

    # Run individual extractors on VALIDATION
    val_signals = {}
    val_signals["semantic"] = detector._semantic_extractor.extract(val_reps, tg_config, samples=val_samples).scores
    val_signals["neighborhood"] = detector._neighborhood_extractor.extract(val_reps, tg_config, samples=val_samples).scores
    val_signals["stability"] = detector._stability_extractor.extract(val_reps, tg_config, samples=val_samples).scores
    val_signals["density"] = detector._density_extractor.extract(val_reps, tg_config).scores

    # Ground truth arrays
    y_train = np.array([s.poison_ground_truth is True for s in train_samples], dtype=bool)
    y_val = np.array([s.poison_ground_truth is True for s in val_samples], dtype=bool)

    diagnostics = {
        "metadata": {
            "dataset": "SST-2",
            "total_samples": 5000,
            "train_samples": len(train_samples),
            "train_poisoned": int(np.sum(y_train)),
            "train_clean": int(len(y_train) - np.sum(y_train)),
            "val_samples": len(val_samples),
            "val_poisoned": int(np.sum(y_val)),
            "val_clean": int(len(y_val) - np.sum(y_val)),
            "seed": 42,
            "attack": "text_backdoor_v1",
        },
        "signals": {},
    }

    print("\n--- SIGNAL SCORE DISTRIBUTIONS & DISCRIMINATIVE POWER ---")
    header = f"{'Signal':<15} | {'Split':<5} | {'Clean Mean (Std)':<20} | {'Poison Mean (Std)':<20} | {'Margin':<8} | {'AUROC':<7} | {'AUPR':<7} | {'KS-Stat':<7}"
    print(header)
    print("-" * len(header))

    for sig in signals:
        diagnostics["signals"][sig] = {}

        for split_name, scores_arr, y_arr in [("TRAIN", train_signals[sig], y_train), ("VAL", val_signals[sig], y_val)]:
            scores_clean = np.array(scores_arr)[~y_arr]
            scores_poison = np.array(scores_arr)[y_arr]

            stats_clean = compute_distribution_stats(scores_clean)
            stats_poison = compute_distribution_stats(scores_poison)

            margin = stats_poison["mean"] - stats_clean["mean"]
            
            # AUROC & AUPR
            if len(np.unique(y_arr)) > 1:
                auroc = float(roc_auc_score(y_arr, scores_arr))
                aupr = float(average_precision_score(y_arr, scores_arr))
            else:
                auroc, aupr = 0.5, 0.0

            # Kolmogorov-Smirnov 2-sample test
            ks_res = stats.ks_2samp(scores_poison, scores_clean)
            ks_stat = float(ks_res.statistic)
            ks_pvalue = float(ks_res.pvalue)

            diagnostics["signals"][sig][split_name] = {
                "clean_distribution": stats_clean,
                "poisoned_distribution": stats_poison,
                "separation_margin": round(margin, 6),
                "auroc": round(auroc, 4),
                "aupr": round(aupr, 4),
                "ks_statistic": round(ks_stat, 4),
                "ks_pvalue": ks_pvalue,
            }

            clean_str = f"{stats_clean['mean']:.4f} ({stats_clean['std']:.4f})"
            poison_str = f"{stats_poison['mean']:.4f} ({stats_poison['std']:.4f})"
            print(f"{sig:<15} | {split_name:<5} | {clean_str:<20} | {poison_str:<20} | {margin:+8.4f} | {auroc:<7.4f} | {aupr:<7.4f} | {ks_stat:<7.4f}")

    # Output to disk
    out_dir = Path("artifacts/research/sst2/signal_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "signal_distributions.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"\nSaved signal diagnostics to {out_dir / 'signal_distributions.json'}")
    return diagnostics


if __name__ == "__main__":
    run_diagnostics()
