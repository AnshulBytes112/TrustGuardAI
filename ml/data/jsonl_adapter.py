import json
import os
from dataclasses import dataclass

from ml.data.schemas import (
    DatasetImportResult,
    DatasetLabelMode,
    LabelStatus,
    Sample,
    Split,
)
from ml.interfaces import DatasetAdapter


@dataclass
class JSONLDatasetAdapterConfig:
    dataset_id: str
    dataset_version: str
    text_field: str = "text"
    label_field: str = "label"
    id_field: str | None = None
    split_field: str | None = None


class JSONLDatasetAdapter(DatasetAdapter):
    def __init__(self, config: JSONLDatasetAdapterConfig):
        self.config = config

    def load(self, source: str) -> DatasetImportResult:
        if not os.path.exists(source):
            raise FileNotFoundError("Dataset file not found")

        # Check for empty file
        if os.path.getsize(source) == 0:
            raise ValueError("Empty dataset file")

        samples = []
        has_known_labels = False
        has_unknown_labels = False
        seen_ids = set()

        with open(source, "r", encoding="utf-8") as f:
            for line_number_0, line in enumerate(f):
                line_number = line_number_0 + 1
                line_stripped = line.strip()
                if not line_stripped:
                    raise ValueError(f"Line {line_number}: Empty line found")

                try:
                    obj = json.loads(line_stripped)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Line {line_number}: Malformed JSON - {e!s}")

                if not isinstance(obj, dict):
                    raise TypeError(f"Line {line_number}: Non-object JSON found")

                if self.config.text_field not in obj:
                    raise ValueError(f"Line {line_number}: Required text field '{self.config.text_field}' not found")

                text_val = obj[self.config.text_field]
                if text_val is None:
                    raise ValueError(f"Line {line_number}: Text cannot be null")
                
                text_str = str(text_val)
                if not text_str.strip():
                    raise ValueError(f"Line {line_number}: Text cannot be empty or whitespace only")

                # Handle ID
                if self.config.id_field:
                    if self.config.id_field not in obj:
                        # Missing configured ID field
                        raise ValueError(f"Line {line_number}: Configured ID field '{self.config.id_field}' not found")
                    sample_id_val = obj[self.config.id_field]
                    if sample_id_val is None:
                        raise ValueError(f"Line {line_number}: Empty sample ID")
                    sample_id = str(sample_id_val)
                    if not sample_id.strip():
                        raise ValueError(f"Line {line_number}: Empty sample ID")
                else:
                    sample_id = f"{self.config.dataset_id}:{self.config.dataset_version}:{line_number_0}"

                if sample_id in seen_ids:
                    raise ValueError(f"Line {line_number}: Duplicate sample ID found: {sample_id}")
                seen_ids.add(sample_id)

                # Handle label
                label = None
                label_status = LabelStatus.UNKNOWN
                if self.config.label_field in obj:
                    label_val = obj[self.config.label_field]
                    if label_val is not None and str(label_val).strip():
                        label = label_val
                        label_status = LabelStatus.KNOWN

                if label_status == LabelStatus.KNOWN:
                    has_known_labels = True
                else:
                    has_unknown_labels = True

                # Handle split
                split = Split.UNASSIGNED
                if self.config.split_field and self.config.split_field in obj:
                    split_val = obj[self.config.split_field]
                    if split_val is not None and str(split_val).strip():
                        split_str = str(split_val).strip().upper()
                        try:
                            split = Split(split_str)
                        except ValueError:
                            raise ValueError(f"Line {line_number}: Invalid split value '{split_val}'")

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

        # Replicate dataset inference logic
        if has_known_labels and has_unknown_labels:
            label_mode = DatasetLabelMode.PARTIALLY_LABELLED
        elif has_known_labels:
            label_mode = DatasetLabelMode.FULLY_LABELLED
        else:
            label_mode = DatasetLabelMode.UNLABELLED

        return DatasetImportResult(
            samples=samples,
            label_mode=label_mode,
            total_samples=len(samples)
        )
