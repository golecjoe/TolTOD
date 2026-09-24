"""Convert an issue catalog into per-detector visualization data."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..detection.models import (
    DETECTOR_ISSUES,
    DetectorIssue,
    IssueCatalog,
    SampleIssue,
)
from .config import QCReportConfig


OTHER_GLITCH_TYPES = (
    SampleIssue.NONFINITE,
    SampleIssue.SATURATION,
    SampleIssue.DROPOUT,
    SampleIssue.FLATLINE,
)


@dataclass(frozen=True)
class DrasticThreshold:
    """Population mean-plus-sigma threshold for a sample-count column."""

    mean: float
    standard_deviation: float
    threshold: float
    eligible_detector_count: int


@dataclass(frozen=True)
class DetectorManifestRow:
    """One detector row used by CSV and HTML renderers."""

    network_position: int
    detector_index: int
    apt_uid: str
    spike_events: int
    spike_samples: int
    jump_events: int
    jump_samples: int
    other_events: int
    other_samples: int
    source_flag_samples: int
    correlated_samples: int
    affected_samples: int
    affected_fraction: float
    robust_noise: float | None
    quality_flags: tuple[str, ...]
    excluded_from_thresholds: bool
    spike_drastic: bool = False
    jump_drastic: bool = False
    other_drastic: bool = False
    affected_drastic: bool = False


@dataclass(frozen=True)
class DetectorManifest:
    """Per-detector rows and sample-impact thresholds for a network."""

    network_id: int
    sample_count: int
    rows: tuple[DetectorManifestRow, ...]
    thresholds: Mapping[str, DrasticThreshold]


def build_detector_manifest(
    catalog: IssueCatalog,
    *,
    detector_metadata: Mapping[str, np.ndarray] | None = None,
    config: QCReportConfig | None = None,
) -> DetectorManifest:
    """Aggregate detector events and sample masks for visualization."""

    settings = config or QCReportConfig()
    metadata = detector_metadata or {}
    apt_uid = metadata.get("apt_uid")
    if apt_uid is not None and len(apt_uid) != catalog.detector_count:
        raise ValueError("apt_uid metadata must contain one value per detector")

    event_counts = Counter(
        (event.network_position, event.issue_type)
        for event in catalog.events
        if event.network_position is not None
    )
    metrics_by_position = {
        metrics.network_position: metrics for metrics in catalog.detector_metrics
    }

    spike_mask = catalog.mask_for(SampleIssue.SPIKE)
    jump_mask = catalog.mask_for(SampleIssue.JUMP)
    other_mask = np.zeros_like(spike_mask)
    for issue in OTHER_GLITCH_TYPES:
        other_mask |= catalog.mask_for(issue)
    source_mask = catalog.mask_for(SampleIssue.SOURCE_FLAG)
    correlated_mask = catalog.mask_for(SampleIssue.CORRELATED)

    # Correlated is a classification of existing issue cells, so omit that bit
    # from the union to avoid counting the same sample twice.
    primary_union = catalog.sample_flags & ~np.uint16(SampleIssue.CORRELATED)
    affected_mask = primary_union != 0

    provisional: list[dict[str, Any]] = []
    for position in range(catalog.detector_count):
        metrics = metrics_by_position.get(position)
        quality = metrics.quality_flags if metrics is not None else DetectorIssue.NONE
        excluded = bool(quality & DetectorIssue.PERSISTENT_SOURCE_FLAG)
        other_events = sum(
            event_counts[(position, issue)] for issue in OTHER_GLITCH_TYPES
        )
        uid = _format_uid(apt_uid[position]) if apt_uid is not None else ""
        affected_samples = int(np.count_nonzero(affected_mask[:, position]))
        provisional.append(
            {
                "network_position": position,
                "detector_index": int(catalog.detector_indices[position]),
                "apt_uid": uid,
                "spike_events": event_counts[(position, SampleIssue.SPIKE)],
                "spike_samples": int(np.count_nonzero(spike_mask[:, position])),
                "jump_events": event_counts[(position, SampleIssue.JUMP)],
                "jump_samples": int(np.count_nonzero(jump_mask[:, position])),
                "other_events": other_events,
                "other_samples": int(np.count_nonzero(other_mask[:, position])),
                "source_flag_samples": int(
                    np.count_nonzero(source_mask[:, position])
                ),
                "correlated_samples": int(
                    np.count_nonzero(correlated_mask[:, position])
                ),
                "affected_samples": affected_samples,
                "affected_fraction": affected_samples / catalog.sample_count,
                "robust_noise": metrics.robust_noise if metrics is not None else None,
                "quality_flags": tuple(
                    issue.name.lower()
                    for issue in DETECTOR_ISSUES
                    if quality & issue
                ),
                "excluded_from_thresholds": (
                    excluded and settings.exclude_persistent_source_flags
                ),
            }
        )

    threshold_columns = (
        "spike_samples",
        "jump_samples",
        "other_samples",
        "affected_samples",
    )
    thresholds = {
        column: _calculate_threshold(provisional, column, settings.drastic_sigma)
        for column in threshold_columns
    }

    rows = tuple(
        DetectorManifestRow(
            **row,
            spike_drastic=_is_drastic(row, "spike_samples", thresholds),
            jump_drastic=_is_drastic(row, "jump_samples", thresholds),
            other_drastic=_is_drastic(row, "other_samples", thresholds),
            affected_drastic=_is_drastic(row, "affected_samples", thresholds),
        )
        for row in provisional
    )
    return DetectorManifest(
        network_id=catalog.network_id,
        sample_count=catalog.sample_count,
        rows=rows,
        thresholds=thresholds,
    )


def _calculate_threshold(
    rows: list[dict[str, Any]],
    column: str,
    sigma: float,
) -> DrasticThreshold:
    values = np.asarray(
        [row[column] for row in rows if not row["excluded_from_thresholds"]],
        dtype=np.float64,
    )
    if values.size == 0:
        return DrasticThreshold(0.0, 0.0, float("inf"), 0)
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=0))
    return DrasticThreshold(
        mean=mean,
        standard_deviation=standard_deviation,
        threshold=mean + sigma * standard_deviation,
        eligible_detector_count=int(values.size),
    )


def _is_drastic(
    row: Mapping[str, Any],
    column: str,
    thresholds: Mapping[str, DrasticThreshold],
) -> bool:
    return (
        not row["excluded_from_thresholds"]
        and row[column] > thresholds[column].threshold
    )


def _format_uid(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return ""
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"

