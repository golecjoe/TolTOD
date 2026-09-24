"""Network-coincidence detection from detector-local issue candidates."""

from __future__ import annotations

import numpy as np

from .config import DetectionConfig
from .models import IssueCatalog, IssueEvent, SampleIssue
from .utilities import retain_runs, true_runs


_CORRELATION_INPUT_BITS = (
    SampleIssue.NONFINITE
    | SampleIssue.SOURCE_FLAG
    | SampleIssue.SATURATION
    | SampleIssue.DROPOUT
    | SampleIssue.FLATLINE
    | SampleIssue.SPIKE
    | SampleIssue.JUMP
)


def detect_correlated_issues(
    catalog: IssueCatalog,
    config: DetectionConfig,
) -> None:
    """Identify times when an unusual fraction of active detectors has issues."""

    source_fraction = np.mean(catalog.mask_for(SampleIssue.SOURCE_FLAG), axis=0)
    eligible = source_fraction < config.permanent_source_flag_fraction
    eligible_count = int(np.count_nonzero(eligible))
    if eligible_count < config.correlated_min_detectors:
        return

    input_mask = (catalog.sample_flags & np.uint16(_CORRELATION_INPUT_BITS)) != 0
    eligible_issues = input_mask[:, eligible]
    affected_count = np.count_nonzero(eligible_issues, axis=1)
    affected_fraction = affected_count / eligible_count
    candidate_times = (
        affected_count >= config.correlated_min_detectors
    ) & (affected_fraction >= config.correlated_detector_fraction)
    minimum_samples = config.samples_for(
        config.correlated_min_seconds,
        catalog.sample_interval,
    )
    retained_times = retain_runs(
        candidate_times[:, np.newaxis],
        minimum=minimum_samples,
    )[:, 0]
    if not np.any(retained_times):
        return

    correlated_mask = input_mask & retained_times[:, np.newaxis]
    correlated_mask[:, ~eligible] = False
    catalog.add_sample_mask(SampleIssue.CORRELATED, correlated_mask)

    events: list[IssueEvent] = []
    for start, stop in true_runs(retained_times):
        detectors = np.any(correlated_mask[start:stop], axis=0)
        events.append(
            IssueEvent(
                issue_type=SampleIssue.CORRELATED,
                network_id=catalog.network_id,
                detector_index=None,
                network_position=None,
                start_sample=start,
                stop_sample=stop,
                severity=float(np.max(affected_fraction[start:stop])),
                affected_detector_count=int(np.count_nonzero(detectors)),
                details={
                    "peak_detector_fraction": float(
                        np.max(affected_fraction[start:stop])
                    ),
                    "eligible_detector_count": eligible_count,
                },
            )
        )
    catalog.add_events(events)

