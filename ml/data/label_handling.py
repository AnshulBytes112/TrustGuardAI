from dataclasses import dataclass

from ml.data.schemas import DatasetLabelMode, LabelStatus, Sample


@dataclass
class LabelAvailabilityStats:
    total_samples: int
    labelled_samples: int
    unlabelled_samples: int
    label_mode: DatasetLabelMode

    @property
    def labelled_percentage(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.labelled_samples / self.total_samples

    @property
    def unlabelled_percentage(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.unlabelled_samples / self.total_samples


def infer_label_mode(samples: list[Sample]) -> LabelAvailabilityStats:
    """
    Infers the label availability mode for a given dataset of samples.
    The source of truth is strictly `Sample.label_status`.
    """
    if not samples:
        raise ValueError("Cannot infer label mode for an empty dataset")

    total_samples = len(samples)
    labelled_samples = sum(1 for s in samples if s.label_status == LabelStatus.KNOWN)
    unlabelled_samples = sum(1 for s in samples if s.label_status == LabelStatus.UNKNOWN)

    # Sanity check against invalid states
    if labelled_samples + unlabelled_samples != total_samples:
        raise ValueError("Dataset contains samples with invalid label_status")

    if labelled_samples == total_samples:
        label_mode = DatasetLabelMode.FULLY_LABELLED
    elif unlabelled_samples == total_samples:
        label_mode = DatasetLabelMode.UNLABELLED
    else:
        label_mode = DatasetLabelMode.PARTIALLY_LABELLED

    return LabelAvailabilityStats(
        total_samples=total_samples,
        labelled_samples=labelled_samples,
        unlabelled_samples=unlabelled_samples,
        label_mode=label_mode,
    )
