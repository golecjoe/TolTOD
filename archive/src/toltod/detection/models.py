"""Issue catalogs and detector-level diagnostic models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntFlag
from typing import Any, Mapping, Sequence

import numpy as np


class SampleIssue(IntFlag):
    """Bit assignments for sample-level issue masks."""

    NONE = 0
    NONFINITE = 1 << 0
    SOURCE_FLAG = 1 << 1
    SATURATION = 1 << 2
    DROPOUT = 1 << 3
    FLATLINE = 1 << 4
    SPIKE = 1 << 5
    JUMP = 1 << 6
    CORRELATED = 1 << 7


class DetectorIssue(IntFlag):
    """Detector-level diagnostic states; none imply automatic rejection."""

    NONE = 0
    EXCESS_ISSUE_RATE = 1 << 0
    EXCESS_NOISE = 1 << 1
    PERSISTENT_SOURCE_FLAG = 1 << 2


SAMPLE_ISSUES = tuple(issue for issue in SampleIssue if issue is not SampleIssue.NONE)
DETECTOR_ISSUES = tuple(
    issue for issue in DetectorIssue if issue is not DetectorIssue.NONE
)


@dataclass(frozen=True)
class IssueEvent:
    """One contiguous detector-local or network-level issue."""

    issue_type: SampleIssue
    network_id: int
    start_sample: int
    stop_sample: int
    detector_index: int | None = None
    network_position: int | None = None
    severity: float | None = None
    affected_detector_count: int = 1
    details: Mapping[str, float | int | str] = field(default_factory=dict)

    @property
    def duration_samples(self) -> int:
        return self.stop_sample - self.start_sample

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["issue_type"] = self.issue_type.name.lower()
        result["details"] = dict(self.details)
        return result


@dataclass(frozen=True)
class DetectorMetrics:
    """Summary statistics for one detector after issue detection."""

    network_id: int
    network_position: int
    detector_index: int
    valid_fraction: float
    issue_fraction: float
    source_flag_fraction: float
    robust_center: float | None
    robust_noise: float | None
    spike_event_count: int
    jump_event_count: int
    quality_flags: DetectorIssue = DetectorIssue.NONE

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["quality_flags"] = [
            issue.name.lower()
            for issue in DETECTOR_ISSUES
            if self.quality_flags & issue
        ]
        return result


@dataclass
class IssueCatalog:
    """All sample flags, events, and detector diagnostics for one network."""

    network_id: int
    sample_count: int
    detector_indices: np.ndarray
    sample_interval: float
    sample_flags: np.ndarray
    events: list[IssueEvent] = field(default_factory=list)
    detector_metrics: list[DetectorMetrics] = field(default_factory=list)
    source_path: str | None = None
    elapsed_seconds: float | None = None
    detection_settings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(
        cls,
        *,
        network_id: int,
        sample_count: int,
        detector_indices: Sequence[int] | np.ndarray,
        sample_interval: float,
        source_path: str | None = None,
        detection_settings: Mapping[str, Any] | None = None,
    ) -> "IssueCatalog":
        indices = np.asarray(detector_indices, dtype=np.int64)
        return cls(
            network_id=network_id,
            sample_count=sample_count,
            detector_indices=indices,
            sample_interval=sample_interval,
            sample_flags=np.zeros((sample_count, indices.size), dtype=np.uint16),
            source_path=source_path,
            detection_settings=dict(detection_settings or {}),
        )

    @property
    def detector_count(self) -> int:
        return int(self.detector_indices.size)

    def add_sample_mask(self, issue: SampleIssue, mask: np.ndarray) -> None:
        if issue is SampleIssue.NONE or int(issue) & (int(issue) - 1):
            raise ValueError("Exactly one nonzero SampleIssue must be supplied")
        boolean_mask = np.asarray(mask, dtype=bool)
        if boolean_mask.shape != self.sample_flags.shape:
            raise ValueError(
                f"Issue mask shape {boolean_mask.shape} does not match "
                f"catalog shape {self.sample_flags.shape}"
            )
        self.sample_flags[boolean_mask] |= np.uint16(issue)

    def mask_for(self, issue: SampleIssue) -> np.ndarray:
        return (self.sample_flags & np.uint16(issue)) != 0

    def any_sample_mask(self) -> np.ndarray:
        return self.sample_flags != 0

    def add_events(self, events: Sequence[IssueEvent]) -> None:
        self.events.extend(events)

    def summary(self) -> dict[str, Any]:
        sample_counts = {
            issue.name.lower(): int(np.count_nonzero(self.mask_for(issue)))
            for issue in SAMPLE_ISSUES
        }
        event_counts = {
            issue.name.lower(): sum(
                event.issue_type == issue for event in self.events
            )
            for issue in SAMPLE_ISSUES
        }
        detector_counts = {
            issue.name.lower(): sum(
                bool(metrics.quality_flags & issue)
                for metrics in self.detector_metrics
            )
            for issue in DETECTOR_ISSUES
        }
        return {
            "source_path": self.source_path,
            "network_id": self.network_id,
            "sample_count": self.sample_count,
            "detector_count": self.detector_count,
            "sample_interval": self.sample_interval,
            "sample_detector_issue_counts": sample_counts,
            "event_counts": event_counts,
            "detector_issue_counts": detector_counts,
            "elapsed_seconds": self.elapsed_seconds,
        }
