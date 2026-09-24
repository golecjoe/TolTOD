"""Detector-level diagnostic metrics calculated after sample detection."""

from __future__ import annotations

from collections import Counter

import numpy as np

from .config import DetectionConfig
from .models import (
    DetectorIssue,
    DetectorMetrics,
    IssueCatalog,
    SampleIssue,
)
from .utilities import robust_mad


def calculate_detector_metrics(
    signal: np.ndarray,
    catalog: IssueCatalog,
    config: DetectionConfig,
) -> None:
    """Populate metrics and diagnostic quality flags without rejecting detectors."""

    any_issue = catalog.any_sample_mask()
    source_mask = catalog.mask_for(SampleIssue.SOURCE_FLAG)
    event_counts = Counter(
        (event.network_position, event.issue_type)
        for event in catalog.events
        if event.network_position is not None
    )

    preliminary: list[dict[str, float | int | None]] = []
    for position in range(signal.shape[1]):
        good = ~any_issue[:, position] & np.isfinite(signal[:, position])
        valid_fraction = float(np.mean(good))
        issue_fraction = float(np.mean(any_issue[:, position]))
        source_fraction = float(np.mean(source_mask[:, position]))
        center: float | None = None
        noise: float | None = None
        if np.any(good):
            center = float(np.median(signal[good, position]))
        valid_pairs = good[:-1] & good[1:]
        if np.any(valid_pairs):
            estimate = robust_mad(np.diff(signal[:, position])[valid_pairs])
            if np.isfinite(estimate):
                noise = float(estimate / np.sqrt(2.0))
        preliminary.append(
            {
                "valid_fraction": valid_fraction,
                "issue_fraction": issue_fraction,
                "source_fraction": source_fraction,
                "center": center,
                "noise": noise,
            }
        )

    reference_values = [
        float(item["noise"])
        for item in preliminary
        if item["noise"] is not None
        and float(item["noise"]) >= config.minimum_scale
        and float(item["source_fraction"]) < config.permanent_source_flag_fraction
    ]
    reference_noise = (
        float(np.median(reference_values)) if reference_values else float("nan")
    )

    metrics: list[DetectorMetrics] = []
    for position, item in enumerate(preliminary):
        flags = DetectorIssue.NONE
        if float(item["issue_fraction"]) >= config.excess_issue_fraction:
            flags |= DetectorIssue.EXCESS_ISSUE_RATE
        if float(item["source_fraction"]) >= config.permanent_source_flag_fraction:
            flags |= DetectorIssue.PERSISTENT_SOURCE_FLAG
        noise = item["noise"]
        if (
            noise is not None
            and np.isfinite(reference_noise)
            and float(noise) > config.excess_noise_factor * reference_noise
        ):
            flags |= DetectorIssue.EXCESS_NOISE

        metrics.append(
            DetectorMetrics(
                network_id=catalog.network_id,
                network_position=position,
                detector_index=int(catalog.detector_indices[position]),
                valid_fraction=float(item["valid_fraction"]),
                issue_fraction=float(item["issue_fraction"]),
                source_flag_fraction=float(item["source_fraction"]),
                robust_center=(
                    float(item["center"]) if item["center"] is not None else None
                ),
                robust_noise=float(noise) if noise is not None else None,
                spike_event_count=event_counts[(position, SampleIssue.SPIKE)],
                jump_event_count=event_counts[(position, SampleIssue.JUMP)],
                quality_flags=flags,
            )
        )
    catalog.detector_metrics = metrics

