import dataclasses
import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ml.data.csv_adapter import CSVDatasetAdapterConfig
from ml.data.jsonl_adapter import JSONLDatasetAdapterConfig
from ml.pipeline.schemas import DetectionPipelineConfig, DetectionPipelineResult


class CSVDatasetConfig(BaseModel):
    type: Literal["csv"] = "csv"
    path: str = Field(..., description="Path to the CSV file")
    configuration: CSVDatasetAdapterConfig


class JSONLDatasetConfig(BaseModel):
    type: Literal["jsonl"] = "jsonl"
    path: str = Field(..., description="Path to the JSONL file")
    configuration: JSONLDatasetAdapterConfig


DatasetConfig = Annotated[
    CSVDatasetConfig | JSONLDatasetConfig, Field(discriminator="type")
]


class ExperimentConfig(BaseModel):
    experiment_name: str = Field(..., min_length=1)
    dataset: DatasetConfig
    pipeline: DetectionPipelineConfig

    model_config = ConfigDict(frozen=True)

    def compute_fingerprint(self) -> str:
        """
        Computes a deterministic SHA-256 fingerprint of the experiment configuration.
        Excludes machine-specific fields such as the dataset 'path'.
        """
        # Serialize pipeline config
        pipeline_dict = self.pipeline.model_dump(mode="json")
        
        # Serialize dataset config logically (excluding 'path')
        if isinstance(self.dataset.configuration, CSVDatasetAdapterConfig):
            ds_config_dict = dataclasses.asdict(self.dataset.configuration)
        else:
            ds_config_dict = dataclasses.asdict(self.dataset.configuration)
            
        dataset_logical = {
            "type": self.dataset.type,
            "configuration": ds_config_dict,
        }

        metadata = {
            "experiment_name": self.experiment_name,
            "dataset": dataset_logical,
            "pipeline": pipeline_dict,
        }

        canonical_json = json.dumps(metadata, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class ExperimentResult(BaseModel):
    experiment_name: str
    experiment_fingerprint: str
    dataset_id: str
    dataset_version: str
    pipeline_result: DetectionPipelineResult

    model_config = ConfigDict(frozen=True)
