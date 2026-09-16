import csv
import datetime
import hashlib
import io
import json
import logging
import platform
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ml.benchmark.schemas import BenchmarkMode
from ml.data.csv_adapter import CSVDatasetAdapter
from ml.data.jsonl_adapter import JSONLDatasetAdapter
from ml.data.schemas import Sample, Split
from ml.detectors.trustguard.detector import TrustGuardDetector
from ml.detectors.trustguard.schemas import SignalType, TrustGuardConfig
from ml.evaluation.calibration import ThresholdCalibrator, apply_threshold
from ml.evaluation.engine import DetectionEvaluationEngine
from ml.experiments.schemas import CSVDatasetConfig, DatasetConfig, JSONLDatasetConfig
from ml.features.config import RepresentationConfig
from ml.features.representations import DistilBERTRepresentationProvider
from ml.features.service import RepresentationService
from ml.models.classifier import TrainableDownstreamClassifier
from ml.models.schemas import TrainingConfig
from ml.poisoning.config import TextPoisoningConfig
from ml.poisoning.engine import TextPoisoningEngine

logger = logging.getLogger(__name__)

AblationVariantName = Literal[
    "TRUSTGUARD_FULL",
    "TRUSTGUARD_NO_SEMANTIC",
    "TRUSTGUARD_NO_NEIGHBORHOOD",
    "TRUSTGUARD_NO_STABILITY",
    "TRUSTGUARD_NO_DENSITY",
    "TRUSTGUARD_EQUAL_WEIGHTS",
    "TRUSTGUARD_VALIDATION_WEIGHTS",
]


class AblationVariantConfig:
    """
    Helper providing configurations for the 7 standard ablation variants.
    """

    @staticmethod
    def get_config(variant: str, seed: int = 42) -> TrustGuardConfig:
        var_upper = variant.upper().strip()
        all_signals: list[SignalType] = ["semantic", "neighborhood", "stability", "density"]

        if var_upper in ("TRUSTGUARD_FULL", "TRUSTGUARD_VALIDATION_WEIGHTS"):
            return TrustGuardConfig(
                enabled_signals=all_signals,
                weighting_strategy="learned_validation",
                seed=seed,
            )
        elif var_upper == "TRUSTGUARD_EQUAL_WEIGHTS":
            return TrustGuardConfig(
                enabled_signals=all_signals,
                weighting_strategy="equal",
                weights={"semantic": 0.25, "neighborhood": 0.25, "stability": 0.25, "density": 0.25},
                seed=seed,
            )
        elif var_upper == "TRUSTGUARD_NO_SEMANTIC":
            return TrustGuardConfig(
                enabled_signals=["neighborhood", "stability", "density"],
                weighting_strategy="learned_validation",
                seed=seed,
            )
        elif var_upper == "TRUSTGUARD_NO_NEIGHBORHOOD":
            return TrustGuardConfig(
                enabled_signals=["semantic", "stability", "density"],
                weighting_strategy="learned_validation",
                seed=seed,
            )
        elif var_upper == "TRUSTGUARD_NO_STABILITY":
            return TrustGuardConfig(
                enabled_signals=["semantic", "neighborhood", "density"],
                weighting_strategy="learned_validation",
                seed=seed,
            )
        elif var_upper == "TRUSTGUARD_NO_DENSITY":
            return TrustGuardConfig(
                enabled_signals=["semantic", "neighborhood", "stability"],
                weighting_strategy="learned_validation",
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown ablation variant: {variant}")


class AblationRow(BaseModel):
    """
    Granular measurement row for an ablation run under specific attack, rate, and seed.
    """
    experiment_fingerprint: str
    dataset_fingerprint: str
    attack_fingerprint: str

    method: str = "trustguard"
    ablation_variant: str

    attack_type: str
    poison_rate: float
    seed: int

    train_size: int
    validation_size: int
    test_size: int

    poisoned_train_count: int
    poisoned_validation_count: int
    poisoned_test_count: int

    semantic_enabled: bool
    neighborhood_enabled: bool
    stability_enabled: bool
    density_enabled: bool

    semantic_weight: float | None = None
    neighborhood_weight: float | None = None
    stability_weight: float | None = None
    density_weight: float | None = None

    threshold: float | None = None
    filtering_budget: float | None = None

    num_flagged: int
    num_removed: int
    retention_rate: float

    precision: float
    recall: float | None = None
    f1: float
    fpr: float | None = None
    fnr: float | None = None
    auroc: float | None = None
    auprc: float | None = None

    downstream_clean_accuracy: float
    downstream_attack_success_rate: float

    runtime_seconds: float

    model_config = ConfigDict(frozen=True)


class AblationComparisonReport(BaseModel):
    """
    Structured container for complete ablation study results and paper tables.
    """
    experiment_name: str
    experiment_fingerprint: str
    dataset_fingerprint: str
    timestamp: str
    benchmark_mode: BenchmarkMode
    software_configuration: dict[str, str]
    rows: list[AblationRow]
    statistical_summaries: dict[str, dict[str, float]] | None = None

    model_config = ConfigDict(frozen=True)

    def to_json(self, filepath: str | Path | None = None) -> str:
        serialized = json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True)
        if filepath is not None:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(serialized, encoding="utf-8")
        return serialized

    def to_csv(self, filepath: str | Path | None = None) -> str:
        if not self.rows:
            return ""
        output = io.StringIO()
        fieldnames = list(self.rows[0].model_dump().keys())
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in self.rows:
            writer.writerow(row.model_dump())
        csv_str = output.getvalue()
        if filepath is not None:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(csv_str, encoding="utf-8")
        return csv_str

    def compute_signal_contributions(self) -> list[dict[str, Any]]:
        """
        Computes the empirical contribution of each signal by comparing TRUSTGUARD_FULL
        against each ablated variant under identical attack, poison rate, and seed.
        """
        contributions = []
        full_rows = {
            (r.attack_type, r.poison_rate, r.seed): r
            for r in self.rows
            if r.ablation_variant in ("TRUSTGUARD_FULL", "TRUSTGUARD_VALIDATION_WEIGHTS")
        }

        ablations = [
            ("semantic", "TRUSTGUARD_NO_SEMANTIC"),
            ("neighborhood", "TRUSTGUARD_NO_NEIGHBORHOOD"),
            ("stability", "TRUSTGUARD_NO_STABILITY"),
            ("density", "TRUSTGUARD_NO_DENSITY"),
        ]

        for signal_name, ablated_var in ablations:
            ablated_rows = [r for r in self.rows if r.ablation_variant == ablated_var]
            for ab_row in ablated_rows:
                key = (ab_row.attack_type, ab_row.poison_rate, ab_row.seed)
                if key in full_rows:
                    full = full_rows[key]
                    delta_f1 = full.f1 - ab_row.f1
                    delta_auroc = (
                        (full.auroc - ab_row.auroc)
                        if full.auroc is not None and ab_row.auroc is not None
                        else None
                    )
                    delta_ca = full.downstream_clean_accuracy - ab_row.downstream_clean_accuracy
                    asr_reduction = (
                        ab_row.downstream_attack_success_rate - full.downstream_attack_success_rate
                    )

                    contributions.append({
                        "signal": signal_name,
                        "attack_type": ab_row.attack_type,
                        "poison_rate": ab_row.poison_rate,
                        "seed": ab_row.seed,
                        "delta_f1": round(delta_f1, 4),
                        "delta_auroc": round(delta_auroc, 4) if delta_auroc is not None else None,
                        "delta_clean_accuracy": round(delta_ca, 4),
                        "asr_reduction_gain": round(asr_reduction, 4),
                    })

        return contributions

    def export_paper_artifacts(self, output_dir: str | Path) -> dict[str, str]:
        """
        Generates paper-ready CSV and Markdown reports for the ablation study.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        json_path = out_path / "ablation_results.json"
        csv_path = out_path / "ablation_results.csv"
        self.to_json(json_path)
        self.to_csv(csv_path)

        # 1. Ablation comparison table
        ab_fields = [
            "ablation_variant", "attack_type", "poison_rate", "seed",
            "precision", "recall", "f1", "auroc", "retention_rate",
            "downstream_clean_accuracy", "downstream_attack_success_rate"
        ]
        ab_md = self._generate_markdown_table(ab_fields, "TrustGuard Signal Ablation Comparison")
        (out_path / "ablation_results.md").write_text(ab_md, encoding="utf-8")

        # 2. Attack Comparison table
        att_fields = [
            "attack_type", "poison_rate", "ablation_variant", "f1", "auroc",
            "downstream_clean_accuracy", "downstream_attack_success_rate"
        ]
        att_csv = self._export_subset_csv(out_path / "attack_comparison.csv", att_fields)
        att_md = self._generate_markdown_table(att_fields, "Performance Across Diverse Attack Mechanisms")
        (out_path / "attack_comparison.md").write_text(att_md, encoding="utf-8")

        # 3. Signal contribution analysis CSV
        sig_contribs = self.compute_signal_contributions()
        contrib_path = out_path / "signal_contribution.csv"
        if sig_contribs:
            with open(contrib_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(sig_contribs[0].keys()))
                writer.writeheader()
                writer.writerows(sig_contribs)

        # 4. Weight comparison table
        weight_fields = [
            "ablation_variant", "attack_type", "poison_rate", "seed",
            "semantic_weight", "neighborhood_weight", "stability_weight", "density_weight"
        ]
        weight_csv = self._export_subset_csv(out_path / "weight_comparison.csv", weight_fields)

        return {
            "json": str(json_path),
            "full_csv": str(csv_path),
            "ablation_md": str(out_path / "ablation_results.md"),
            "attack_csv": str(out_path / "attack_comparison.csv"),
            "attack_md": str(out_path / "attack_comparison.md"),
            "signal_contribution_csv": str(contrib_path),
            "weight_csv": str(out_path / "weight_comparison.csv"),
        }

    def _export_subset_csv(self, filepath: Path, fields: list[str]) -> str:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in self.rows:
            writer.writerow(row.model_dump())
        csv_str = output.getvalue()
        filepath.write_text(csv_str, encoding="utf-8")
        return csv_str

    def _generate_markdown_table(self, fields: list[str], title: str) -> str:
        lines = [f"### {title}", "", "| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
        for row in self.rows:
            d = row.model_dump()
            vals = []
            for f in fields:
                v = d.get(f)
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                elif v is None:
                    vals.append("N/A")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")
        return "\n".join(lines)


class AblationExperimentConfig(BaseModel):
    """
    Declarative configuration for running attack diversity and ablation experiments.
    """
    experiment_name: str = Field(..., min_length=1)
    dataset: DatasetConfig
    attack_configs: list[TextPoisoningConfig]
    variants: list[str] = Field(
        default_factory=lambda: [
            "TRUSTGUARD_FULL",
            "TRUSTGUARD_NO_SEMANTIC",
            "TRUSTGUARD_NO_NEIGHBORHOOD",
            "TRUSTGUARD_NO_STABILITY",
            "TRUSTGUARD_NO_DENSITY",
            "TRUSTGUARD_EQUAL_WEIGHTS",
            "TRUSTGUARD_VALIDATION_WEIGHTS",
        ]
    )
    retraining_config: TrainingConfig = Field(default_factory=TrainingConfig)
    representation_config: RepresentationConfig = Field(default_factory=RepresentationConfig)
    benchmark_mode: BenchmarkMode = "natural_threshold"
    filtering_budget: float = 0.20
    seeds: list[int] = Field(default_factory=lambda: [42])

    model_config = ConfigDict(frozen=True)

    def compute_fingerprint(self) -> str:
        attack_dicts = [a.model_dump(mode="json") for a in self.attack_configs]
        retrain_dict = self.retraining_config.model_dump(mode="json")
        rep_dict = self.representation_config.model_dump(mode="json")

        meta = {
            "experiment_name": self.experiment_name,
            "attacks": attack_dicts,
            "variants": sorted(self.variants),
            "retraining_config": retrain_dict,
            "representation_config": rep_dict,
            "benchmark_mode": self.benchmark_mode,
            "filtering_budget": self.filtering_budget,
            "seeds": sorted(self.seeds),
        }
        canonical_json = json.dumps(meta, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class AblationExperimentRunner:
    """
    Orchestrates the execution of Attack Diversity and Signal Ablation experiments.
    """

    def __init__(
        self,
        representation_service: RepresentationService | None = None,
        threshold_calibrator: ThresholdCalibrator | None = None,
        evaluation_engine: DetectionEvaluationEngine | None = None,
        poisoning_engine: TextPoisoningEngine | None = None,
    ):
        self.representation_service = representation_service
        self.threshold_calibrator = threshold_calibrator or ThresholdCalibrator()
        self.evaluation_engine = evaluation_engine or DetectionEvaluationEngine()
        self.poisoning_engine = poisoning_engine or TextPoisoningEngine()

    def run(self, config: AblationExperimentConfig) -> AblationComparisonReport:
        logger.info(f"Starting Ablation Study: {config.experiment_name}")

        # 1. Load Dataset once
        if isinstance(config.dataset, CSVDatasetConfig):
            adapter = CSVDatasetAdapter(config.dataset.configuration)
        elif isinstance(config.dataset, JSONLDatasetConfig):
            adapter = JSONLDatasetAdapter(config.dataset.configuration)
        else:
            raise TypeError(f"Unsupported dataset configuration type: {type(config.dataset)}")

        import_result = adapter.load(config.dataset.path)
        base_samples = import_result.samples
        if not base_samples:
            raise ValueError("Input dataset is empty.")

        # 2. Setup representation provider/service if not injected
        if self.representation_service is None:
            rep_provider = DistilBERTRepresentationProvider(config.representation_config)
            rep_service = RepresentationService(rep_provider, config.representation_config)
        else:
            rep_service = self.representation_service

        rows: list[AblationRow] = []

        # 3. Iterate over seeds and attack configurations
        for current_seed in config.seeds:
            for attack_cfg in config.attack_configs:
                logger.info(
                    f"Applying attack '{attack_cfg.attack_type}' at poison rate {attack_cfg.poison_rate} (seed={current_seed})..."
                )

                # Configure seed on attack config
                active_attack_cfg = attack_cfg.model_copy(update={"seed": current_seed})
                poison_res = self.poisoning_engine.poison(base_samples, active_attack_cfg)
                current_samples = poison_res.samples
                poison_metadata = poison_res.metadata
                attack_fingerprint = poison_metadata.generate_fingerprint()

                # Split ONCE
                train_samples = [s for s in current_samples if s.split == Split.TRAIN]
                val_samples = [s for s in current_samples if s.split == Split.VALIDATION]
                test_samples = [s for s in current_samples if s.split == Split.TEST]

                if not train_samples or not val_samples or not test_samples:
                    raise ValueError("Dataset splits must contain TRAIN, VALIDATION, and TEST.")

                dataset_fingerprint = self._compute_dataset_fingerprint(current_samples)

                # Extract representations ONCE per split
                train_reps = rep_service.extract(train_samples)
                val_reps = rep_service.extract(val_samples)
                test_reps = rep_service.extract(test_samples)

                train_poison_count = sum(1 for s in train_samples if s.poison_ground_truth is True)
                val_poison_count = sum(1 for s in val_samples if s.poison_ground_truth is True)
                test_poison_count = sum(1 for s in test_samples if s.poison_ground_truth is True)

                # 4. Execute each ablation variant under identical conditions
                for variant_name in config.variants:
                    logger.info(f"Evaluating ablation variant: {variant_name}")
                    t0 = time.perf_counter()

                    tg_config = AblationVariantConfig.get_config(variant_name, seed=current_seed)
                    detector = TrustGuardDetector()

                    # Step 1: Fit on TRAIN ONLY
                    detector.fit(
                        train_reps,
                        tg_config,
                        samples=train_samples,
                        representation_provider=rep_service.provider,
                    )

                    # Step 2: Calibrate on VALIDATION ONLY
                    if val_poison_count > 0 and (len(val_samples) - val_poison_count) > 0:
                        detector.calibrate_validation(val_reps, val_samples, tg_config)
                    val_det = detector.detect(val_reps, tg_config, samples=val_samples)

                    # Calibrate threshold
                    if val_poison_count > 0 and (len(val_samples) - val_poison_count) > 0:
                        from ml.evaluation.calibration import CalibrationConfig
                        calib_res = self.threshold_calibrator.calibrate(
                            val_samples, val_det, CalibrationConfig(method=tg_config.threshold_calibration_method)
                        )
                        calibrated_threshold = calib_res.threshold
                    else:
                        calibrated_threshold = 0.5

                    # Step 3: Detect on held-out TEST
                    test_det = detector.detect(test_reps, tg_config, samples=test_samples)
                    test_det_binary = apply_threshold(test_det, calibrated_threshold)

                    # Evaluate detection on TEST
                    eval_report = self.evaluation_engine.evaluate(test_samples, test_det_binary)

                    # Step 4: Purify TRAIN
                    train_det = detector.detect(train_reps, tg_config, samples=train_samples)
                    train_scores = train_det.scores
                    n_train = len(train_samples)

                    if config.benchmark_mode == "matched_budget":
                        k_remove = int(round(config.filtering_budget * n_train))
                        sorted_idx = np.argsort(-np.array(train_scores))
                        removed_idx = set(sorted_idx[:k_remove])
                        retained_idx = [i for i in range(n_train) if i not in removed_idx]
                        applied_thresh = (
                            float(train_scores[sorted_idx[k_remove - 1]]) if k_remove > 0 else 1.0
                        )
                    else:
                        applied_thresh = calibrated_threshold
                        retained_idx = [
                            i for i, s in enumerate(train_scores) if s < calibrated_threshold
                        ]
                        removed_idx = set(range(n_train)) - set(retained_idx)

                    if len(retained_idx) < 2:
                        retained_idx = list(range(min(2, n_train)))
                        removed_idx = set(range(n_train)) - set(retained_idx)

                    num_removed = len(removed_idx)
                    num_flagged = sum(1 for is_anom in test_det_binary.is_anomalous if is_anom)
                    retention_rate = len(retained_idx) / float(n_train)

                    # Step 5: Retrain Downstream Classifier on purified TRAIN representations
                    purified_reps = train_reps.representations[retained_idx]
                    purified_labels = [train_samples[i].label for i in retained_idx]

                    classifier = TrainableDownstreamClassifier(
                        config=config.retraining_config.model_copy(update={"seed": current_seed})
                    )
                    classifier.fit(purified_reps, purified_labels)

                    # Step 6: Evaluate Retrained Model on Clean and Poisoned TEST sets
                    clean_test_idx = [
                        i for i, s in enumerate(test_samples) if s.poison_ground_truth is False
                    ]
                    poison_test_idx = [
                        i for i, s in enumerate(test_samples) if s.poison_ground_truth is True
                    ]

                    # 6.1 Clean Accuracy
                    if clean_test_idx:
                        clean_reps = test_reps.representations[clean_test_idx]
                        clean_true = [test_samples[i].label for i in clean_test_idx]
                        clean_preds = classifier.predict(clean_reps)
                        clean_acc = sum(
                            1 for yp, yt in zip(clean_preds, clean_true) if yp == yt
                        ) / float(len(clean_true))
                    else:
                        clean_acc = 0.0

                    # 6.2 Attack Success Rate
                    if poison_test_idx:
                        poison_reps = test_reps.representations[poison_test_idx]
                        poison_preds = classifier.predict(poison_reps)
                        asr = sum(
                            1 for yp in poison_preds if yp == active_attack_cfg.target_label
                        ) / float(len(poison_test_idx))
                    else:
                        asr = 0.0

                    runtime = time.perf_counter() - t0

                    # Extract active weights from detector
                    active_weights = getattr(detector, "_weights", {})
                    sem_w = active_weights.get("semantic")
                    neigh_w = active_weights.get("neighborhood")
                    stab_w = active_weights.get("stability")
                    dens_w = active_weights.get("density")

                    row = AblationRow(
                        experiment_fingerprint=config.compute_fingerprint(),
                        dataset_fingerprint=dataset_fingerprint,
                        attack_fingerprint=attack_fingerprint,
                        method="trustguard",
                        ablation_variant=variant_name,
                        attack_type=active_attack_cfg.attack_type,
                        poison_rate=active_attack_cfg.poison_rate,
                        seed=current_seed,
                        train_size=len(train_samples),
                        validation_size=len(val_samples),
                        test_size=len(test_samples),
                        poisoned_train_count=train_poison_count,
                        poisoned_validation_count=val_poison_count,
                        poisoned_test_count=test_poison_count,
                        semantic_enabled="semantic" in tg_config.enabled_signals,
                        neighborhood_enabled="neighborhood" in tg_config.enabled_signals,
                        stability_enabled="stability" in tg_config.enabled_signals,
                        density_enabled="density" in tg_config.enabled_signals,
                        semantic_weight=round(sem_w, 4) if sem_w is not None else None,
                        neighborhood_weight=round(neigh_w, 4) if neigh_w is not None else None,
                        stability_weight=round(stab_w, 4) if stab_w is not None else None,
                        density_weight=round(dens_w, 4) if dens_w is not None else None,
                        threshold=round(applied_thresh, 4),
                        filtering_budget=config.filtering_budget if config.benchmark_mode == "matched_budget" else None,
                        num_flagged=num_flagged,
                        num_removed=num_removed,
                        retention_rate=round(retention_rate, 4),
                        precision=round(eval_report.precision, 4),
                        recall=round(eval_report.recall, 4) if eval_report.recall is not None else None,
                        f1=round(eval_report.f1, 4),
                        fpr=round(eval_report.fpr, 4) if eval_report.fpr is not None else None,
                        fnr=round(eval_report.fnr, 4) if eval_report.fnr is not None else None,
                        auroc=round(eval_report.auroc, 4) if eval_report.auroc is not None else None,
                        auprc=round(eval_report.auprc, 4) if eval_report.auprc is not None else None,
                        downstream_clean_accuracy=round(clean_acc, 4),
                        downstream_attack_success_rate=round(asr, 4),
                        runtime_seconds=round(runtime, 4),
                    )
                    rows.append(row)

        # 5. Multi-seed summary
        stat_summaries = None
        if len(config.seeds) > 1:
            stat_summaries = self._compute_ablation_stats(rows, config.variants)

        report = AblationComparisonReport(
            experiment_name=config.experiment_name,
            experiment_fingerprint=config.compute_fingerprint(),
            dataset_fingerprint=dataset_fingerprint,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            benchmark_mode=config.benchmark_mode,
            software_configuration={
                "os": platform.system(),
                "python": platform.python_version(),
            },
            rows=rows,
            statistical_summaries=stat_summaries,
        )

        logger.info(f"Ablation Study completed: {config.experiment_name}")
        return report

    def _compute_dataset_fingerprint(self, samples: list[Sample]) -> str:
        sample_ids = sorted(s.sample_id for s in samples)
        combined = ";".join(sample_ids)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def _compute_ablation_stats(
        self, rows: list[AblationRow], variants: list[str]
    ) -> dict[str, dict[str, float]]:
        summaries: dict[str, dict[str, float]] = {}
        metrics = [
            "precision", "recall", "f1", "auroc", "auprc", "retention_rate",
            "downstream_clean_accuracy", "downstream_attack_success_rate"
        ]

        for var in variants:
            var_rows = [r for r in rows if r.ablation_variant == var]
            if not var_rows:
                continue

            var_stats: dict[str, float] = {}
            for metric in metrics:
                vals = [
                    getattr(r, metric)
                    for r in var_rows
                    if getattr(r, metric) is not None
                ]
                if vals:
                    var_stats[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
                    var_stats[f"{metric}_std"] = round(float(np.std(vals)), 4)

            summaries[var] = var_stats

        return summaries
