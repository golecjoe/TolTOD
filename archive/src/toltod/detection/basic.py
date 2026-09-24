"""Detection of direct validity failures, dropouts, and flatlines."""

from __future__ import annotations

import numpy as np

from .config import DetectionConfig
from .models import IssueCatalog, IssueEvent, SampleIssue
from .utilities import retain_runs, true_runs


def detect_basic_issues(
    signal: np.ndarray,
    catalog: IssueCatalog,
    config: DetectionConfig,
    *,
    source_bad_mask: np.ndarray | None,
) -> None:
    """Populate direct sample-level issues without modifying ``signal``."""

    finite = np.isfinite(signal)
    nonfinite = ~finite
    _record_mask(catalog, SampleIssue.NONFINITE, nonfinite)

    source_bad = np.zeros_like(finite, dtype=bool)
    if source_bad_mask is not None:
        source_bad = np.asarray(source_bad_mask, dtype=bool)
        _record_mask(catalog, SampleIssue.SOURCE_FLAG, source_bad)

    saturation = np.zeros_like(finite, dtype=bool)
    if config.saturation_lower is not None:
        saturation |= signal <= config.saturation_lower
    if config.saturation_upper is not None:
        saturation |= signal >= config.saturation_upper
    saturation &= finite & ~source_bad
    _record_mask(catalog, SampleIssue.SATURATION, saturation)

    minimum_dropout = config.samples_for(
        config.dropout_min_seconds,
        catalog.sample_interval,
    )
    dropout_candidates = (
        finite
        & ~source_bad
        & np.isclose(
            signal,
            config.dropout_value,
            atol=config.dropout_atol,
            rtol=0.0,
        )
    )
    dropout = retain_runs(dropout_candidates, minimum=minimum_dropout)
    _record_mask(catalog, SampleIssue.DROPOUT, dropout)

    minimum_flatline = config.samples_for(
        config.flatline_min_seconds,
        catalog.sample_interval,
        minimum=2,
    )
    excluded = nonfinite | source_bad | saturation | dropout
    flatline = _detect_flatlines(
        signal,
        excluded,
        minimum_samples=minimum_flatline,
        tolerance=config.flatline_atol,
    )
    _record_mask(catalog, SampleIssue.FLATLINE, flatline)


def _detect_flatlines(
    signal: np.ndarray,
    excluded: np.ndarray,
    *,
    minimum_samples: int,
    tolerance: float,
) -> np.ndarray:
    mask = np.zeros(signal.shape, dtype=bool)
    for detector in range(signal.shape[1]):
        valid_edges = ~excluded[:-1, detector] & ~excluded[1:, detector]
        equal_edges = (
            np.abs(np.diff(signal[:, detector])) <= tolerance
        ) & valid_edges
        for edge_start, edge_stop in true_runs(equal_edges):
            sample_start = edge_start
            sample_stop = edge_stop + 1
            if sample_stop - sample_start >= minimum_samples:
                mask[sample_start:sample_stop, detector] = True
    return mask


def _record_mask(
    catalog: IssueCatalog,
    issue: SampleIssue,
    mask: np.ndarray,
) -> None:
    if not np.any(mask):
        return
    catalog.add_sample_mask(issue, mask)
    events: list[IssueEvent] = []
    for position in range(mask.shape[1]):
        for start, stop in true_runs(mask[:, position]):
            events.append(
                IssueEvent(
                    issue_type=issue,
                    network_id=catalog.network_id,
                    detector_index=int(catalog.detector_indices[position]),
                    network_position=position,
                    start_sample=start,
                    stop_sample=stop,
                )
            )
    catalog.add_events(events)
