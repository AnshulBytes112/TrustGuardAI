import csv
import dataclasses
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ml.data.csv_adapter import CSVDatasetAdapterConfig
from ml.data.jsonl_adapter import JSONLDatasetAdapterConfig
from ml.evaluation.calibration import CalibrationConfig
from ml.experiments.schemas import DatasetConfig
from ml.features.config import RepresentationConfig
from ml.features.schemas import RepresentationResult
from ml.models.schemas import TrainingConfig
from ml.poisoning import PoisoningConfig

BenchmarkMode = Literal["natural_threshold", "matched_budget"]


class BenchmarkExperimentConfig(BaseModel):
    """
    Declarative configuration for multi-method baseline experiments.
    """
    experiment_name: str = Field(..., min_length=1)
    dataset: DatasetConfig
    poisoning_config: PoisoningConfig | None = None
    methods: list[str] = Field(
        default_factory=lambda: ["random_filtering", "onion", "flare", "trustguard"],
        description="List of detector methods to benchmark",
    )
    detector_configs: dict[str, Any] = Field(
        default_factory=dict,
        description="Custom detector configurations keyed by method name",
    )
    retraining_config: TrainingConfig = Field(default_factory=TrainingConfig)
    calibration_config: CalibrationConfig = Field(default_factory=CalibrationConfig)
    representation_config: RepresentationConfig = Field(default_factory=RepresentationConfig)
    benchmark_mode: BenchmarkMode = Field(
        default="natural_threshold",
        description="'natural_threshold' uses validation-calibrated threshold; 'matched_budget' uses fixed top-k% removal",
    )
    filtering_budget: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Filtering budget for matched_budget mode (fraction of train samples to remove)",
    )
    seeds: list[int] = Field(
        default_factory=lambda: [42],
        description="Random seed(s) for reproducible multi-seed evaluation",
    )

    model_config = ConfigDict(frozen=True)

    def compute_fingerprint(self) -> str:
        """
        Computes deterministic SHA-256 fingerprint of research-relevant parameters,
        strictly excluding physical paths, hostnames, and timestamps.
        """
        # Dataset logical config
        if isinstance(self.dataset.configuration, (CSVDatasetAdapterConfig, JSONLDatasetAdapterConfig)):
            ds_config_dict = dataclasses.asdict(self.dataset.configuration)
        else:
            ds_config_dict = {}

        dataset_logical = {
            "type": self.dataset.type,
            "configuration": ds_config_dict,
        }

        poisoning_dict = self.poisoning_config.model_dump(mode="json") if self.poisoning_config else None
        retraining_dict = self.retraining_config.model_dump(mode="json")
        calibration_dict = self.calibration_config.model_dump(mode="json")
        representation_dict = self.representation_config.model_dump(mode="json")

        meta = {
            "experiment_name": self.experiment_name,
            "dataset": dataset_logical,
            "poisoning_config": poisoning_dict,
            "methods": sorted(self.methods),
            "benchmark_mode": self.benchmark_mode,
            "filtering_budget": self.filtering_budget,
            "seeds": sorted(self.seeds),
            "retraining_config": retraining_dict,
            "calibration_config": calibration_dict,
            "representation_config": representation_dict,
        }

        canonical_json = json.dumps(meta, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class MethodComparisonRow(BaseModel):
    """
    Granular result row for a single method, attack configuration, and seed.
    """
    method: str
    dataset: str
    dataset_fingerprint: str
    attack: str
    poison_rate: float
    seed: int

    train_size: int
    validation_size: int
    test_size: int

    poisoned_train_count: int
    poisoned_validation_count: int
    poisoned_test_count: int

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
    detector_runtime_seconds: float | None = None
    purification_runtime_seconds: float | None = None
    retraining_runtime_seconds: float | None = None
    peak_memory_mb: float | None = None

    model_config = ConfigDict(frozen=True)


class BenchmarkComparisonReport(BaseModel):
    """
    Complete structured benchmark comparison report across all methods and seeds.
    """
    experiment_name: str
    experiment_fingerprint: str
    dataset_fingerprint: str
    configuration_fingerprint: str
    seeds: list[int]
    benchmark_mode: BenchmarkMode
    software_configuration: dict[str, str]
    timestamp: str
    rows: list[MethodComparisonRow]
    statistical_summaries: dict[str, dict[str, float]] | None = None

    model_config = ConfigDict(frozen=True)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, filepath: str | Path | None = None) -> str:
        data = self.model_dump(mode="json")
        serialized = json.dumps(data, indent=2, sort_keys=True)
        if filepath is not None:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(serialized, encoding="utf-8")
        return serialized

    def to_csv(self, filepath: str | Path | None = None) -> str:
        """
        Generate full benchmark comparison CSV containing all metrics.
        """
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

    def export_paper_artifacts(self, output_dir: str | Path) -> dict[str, str]:
        """
        Exports machine-readable CSV and Markdown tables formatted specifically for paper inclusion.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. Main JSON & Full CSV
        json_path = out_path / "benchmark_report.json"
        csv_path = out_path / "benchmark_results.csv"
        self.to_json(json_path)
        self.to_csv(csv_path)

        # 2. Detection Results Table (CSV & MD)
        det_fields = [
            "method", "attack", "poison_rate", "seed", "precision", "recall", "f1", "fpr", "fnr", "auroc", "auprc"
        ]
        det_csv = self._export_subset_csv(out_path / "detection_results.csv", det_fields)
        det_md = self._generate_markdown_table(det_fields, "Detection Performance Comparison")
        (out_path / "detection_results.md").write_text(det_md, encoding="utf-8")

        # 3. Purification & Downstream Retraining Table (CSV & MD)
        pur_fields = [
            "method", "attack", "poison_rate", "seed", "num_removed", "retention_rate",
            "downstream_clean_accuracy", "downstream_attack_success_rate"
        ]
        pur_csv = self._export_subset_csv(out_path / "purification_results.csv", pur_fields)
        pur_md = self._generate_markdown_table(pur_fields, "Purification and Downstream Retraining Evaluation")
        (out_path / "purification_results.md").write_text(pur_md, encoding="utf-8")

        # 4. Efficiency Results Table (CSV)
        eff_fields = [
            "method", "seed", "runtime_seconds", "detector_runtime_seconds",
            "purification_runtime_seconds", "retraining_runtime_seconds", "peak_memory_mb"
        ]
        eff_csv = self._export_subset_csv(out_path / "efficiency_results.csv", eff_fields)

        return {
            "json": str(json_path),
            "full_csv": str(csv_path),
            "detection_csv": str(out_path / "detection_results.csv"),
            "detection_md": str(out_path / "detection_results.md"),
            "purification_csv": str(out_path / "purification_results.csv"),
            "purification_md": str(out_path / "purification_results.md"),
            "efficiency_csv": str(out_path / "efficiency_results.csv"),
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
