from collections.abc import Sequence
from typing import Any
import numpy as np

from ml.data.schemas import LabelStatus, Sample
from ml.detectors.trustguard.schemas import SampleSignalResult
from ml.features.schemas import RepresentationResult


def get_layer_aggregated_representations(
    representations: RepresentationResult,
    layers: tuple[int, ...] | None = None,
) -> np.ndarray:
    """
    Extracts and aggregates representation matrices across the requested layers.
    If layer_representations is available and contains the requested layers, averages them.
    Otherwise falls back to the top-level representations array.
    """
    if (
        layers is not None
        and representations.layer_representations is not None
        and all(layer in representations.layer_representations for layer in layers)
    ):
        stacked = np.stack(
            [representations.layer_representations[layer] for layer in layers],
            axis=0,
        )
        return np.mean(stacked, axis=0)

    return representations.representations


def normalize_vectors(matrix: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """L2 normalizes rows of a 2D numpy array with numerical stability safeguards."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < eps, eps, norms)
    return matrix / norms


def extract_sample_labels_and_status(
    samples: Sequence[Sample] | None,
    explicit_labels: Sequence[Any] | None,
    num_samples: int,
) -> tuple[list[str | None], list[str]]:
    """
    Resolves labels and label status for a batch of samples.
    Returns (labels, statuses).
    """
    if explicit_labels is not None:
        if len(explicit_labels) != num_samples:
            raise ValueError(
                f"explicit_labels count ({len(explicit_labels)}) does not match sample count ({num_samples})"
            )
        resolved_labels: list[str | None] = [
            str(lbl) if lbl is not None else None for lbl in explicit_labels
        ]
        resolved_statuses = [
            "KNOWN" if lbl is not None else "UNKNOWN" for lbl in resolved_labels
        ]
        return resolved_labels, resolved_statuses

    if samples is not None:
        if len(samples) != num_samples:
            raise ValueError(
                f"samples count ({len(samples)}) does not match sample count ({num_samples})"
            )
        resolved_labels = [
            str(s.label) if (s.label is not None and s.label_status == LabelStatus.KNOWN) else None
            for s in samples
        ]
        resolved_statuses = [s.label_status.value for s in samples]
        return resolved_labels, resolved_statuses

    # Default to unlabelled
    return [None] * num_samples, ["UNKNOWN"] * num_samples


def validate_sample_ids_alignment(
    expected_sample_ids: list[str],
    signal_results_list: list[list[SampleSignalResult]],
) -> None:
    """
    Strictly verifies that all signal results have identical sample_ids in the exact same order.
    """
    for sig_items in signal_results_list:
        if len(sig_items) != len(expected_sample_ids):
            raise ValueError(
                f"Sample ID alignment error: expected {len(expected_sample_ids)} samples, "
                f"got {len(sig_items)} in signal items."
            )
        for expected_id, item in zip(expected_sample_ids, sig_items, strict=True):
            if item.sample_id != expected_id:
                raise ValueError(
                    f"Sample ID mismatch: expected '{expected_id}', got '{item.sample_id}'"
                )
