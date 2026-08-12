import os
from dataclasses import dataclass

import pandas as pd

from ml.data.label_handling import infer_label_mode
from ml.data.schemas import (
    DatasetImportResult,
    LabelStatus,
    Sample,
    Split,
)
from ml.interfaces import DatasetAdapter


@dataclass
class CSVDatasetAdapterConfig:
    dataset_id: str
    dataset_version: str
    text_column: str = "text"
    label_column: str = "label"
    id_column: str | None = None
    split_column: str | None = None

class CSVDatasetAdapter(DatasetAdapter):
    def __init__(self, config: CSVDatasetAdapterConfig):
        self.config = config

    def load(self, source: str) -> DatasetImportResult:
        if not os.path.exists(source):
            raise FileNotFoundError("Dataset file not found")

        df = pd.read_csv(source)
        if df.empty:
            raise ValueError("Empty dataset")

        if self.config.text_column not in df.columns:
            raise ValueError(f"Required text column '{self.config.text_column}' not found")

        if self.config.id_column and self.config.id_column not in df.columns:
            raise ValueError(f"Configured ID column '{self.config.id_column}' not found")

        samples = []

        seen_ids = set()

        for idx, row in df.iterrows():
            # Validate text
            text_val = row[self.config.text_column]
            if pd.isna(text_val) or not str(text_val).strip():
                raise ValueError(f"Empty text at row {idx}")
            
            text_str = str(text_val)

            # Generate or validate ID
            if self.config.id_column:
                sample_id = str(row[self.config.id_column])
                if not sample_id.strip() or pd.isna(row[self.config.id_column]):
                    raise ValueError(f"Empty sample ID at row {idx}")
            else:
                sample_id = f"{self.config.dataset_id}:{self.config.dataset_version}:{idx}"
            
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample ID found: {sample_id}")
            seen_ids.add(sample_id)

            # Handle label
            label = None
            label_status = LabelStatus.UNKNOWN
            if self.config.label_column in df.columns:
                label_val = row[self.config.label_column]
                if not pd.isna(label_val) and str(label_val).strip():
                    label = label_val
                    label_status = LabelStatus.KNOWN

            # Handle split
            split = Split.UNASSIGNED
            if self.config.split_column and self.config.split_column in df.columns:
                split_val = row[self.config.split_column]
                if not pd.isna(split_val) and str(split_val).strip():
                    split_str = str(split_val).strip().upper()
                    try:
                        split = Split(split_str)
                    except ValueError:
                        raise ValueError(f"Invalid split value '{split_str}' at row {idx}")
            
            sample = Sample(
                sample_id=sample_id,
                text=text_str,
                label=label,
                label_status=label_status,
                split=split,
                dataset_id=self.config.dataset_id,
                dataset_version=self.config.dataset_version,
                poison_ground_truth=None
            )
            samples.append(sample)
        
        label_mode = infer_label_mode(samples).label_mode

        return DatasetImportResult(
            samples=samples,
            label_mode=label_mode,
            total_samples=len(samples)
        )
