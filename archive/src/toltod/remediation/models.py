"""Immutable repair and detector-disposition models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np

from ..detection.models import SampleIssue


class RepairAction(str, Enum):
    """Signal operation that a later remediation executor must perform."""

    INTERPOLATE_WITH_NOISE = "interpolate_with_noise"
    DEJUMP_AND_INTERPOLATE_WITH_NOISE = (
        "dejump_and_interpolate_with_noise"
    )


class DetectorDisposition(str, Enum):
    """Whether a detector remains in the cleaned output."""

    KEEP = "keep"
    REMOVE = "remove"


class RemovalReason(str, Enum):
    """Policy condition that requires removal of an entire detector."""

    BAD_SAMPLE_FRACTION = "bad_sample_fraction"
    NO_BRACKETING_GOOD_SAMPLES = "no_bracketing_good_samples"
    INSUFFICIENT_NOISE_SAMPLES = "insufficient_noise_samples"
    INSUFFICIENT_JUMP_LEVEL_SAMPLES = "insufficient_jump_level_samples"


@dataclass(frozen=True)
class RepairSpan:
    """One half-open sample interval that will be replaced during cleaning."""

    network_id: int
    network_position: int
    detector_index: int
    start_sample: int
    stop_sample: int
    action: RepairAction
    issue_types: tuple[SampleIssue, ...]
    left_anchor_sample: int
    right_anchor_sample: int
    noise_context_start: int
    noise_context_stop: int
    available_noise_samples: int
    jump_samples: tuple[int, ...] = ()

    @property
    def duration_samples(self) -> int:
        return self.stop_sample - self.start_sample

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["action"] = self.action.value
        result["issue_types"] = [issue.name.lower() for issue in self.issue_types]
        result["jump_samples"] = list(self.jump_samples)
        return result


@dataclass(frozen=True)
class DetectorRemediation:
    """Policy decision and planned repairs for one detector timestream."""

    network_id: int
    network_position: int
    detector_index: int
    disposition: DetectorDisposition
    bad_sample_count: int
    bad_sample_fraction: float
    planned_repair_sample_count: int
    repair_spans: tuple[RepairSpan, ...] = ()
    removal_reasons: tuple[RemovalReason, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "network_id": self.network_id,
            "network_position": self.network_position,
            "detector_index": self.detector_index,
            "disposition": self.disposition.value,
            "bad_sample_count": self.bad_sample_count,
            "bad_sample_fraction": self.bad_sample_fraction,
            "planned_repair_sample_count": self.planned_repair_sample_count,
            "repair_spans": [span.to_dict() for span in self.repair_spans],
            "removal_reasons": [reason.value for reason in self.removal_reasons],
        }


@dataclass(frozen=True)
class RemediationPlan:
    """All repair and rejection decisions for one detected network."""

    network_id: int
    sample_count: int
    sample_interval: float
    detectors: tuple[DetectorRemediation, ...]
    policy_settings: Mapping[str, Any] = field(default_factory=dict)
    source_path: str | None = None

    @property
    def repair_spans(self) -> tuple[RepairSpan, ...]:
        return tuple(
            span
            for detector in self.detectors
            for span in detector.repair_spans
        )

    @property
    def removed_detector_indices(self) -> tuple[int, ...]:
        return tuple(
            detector.detector_index
            for detector in self.detectors
            if detector.disposition is DetectorDisposition.REMOVE
        )

    @property
    def kept_detector_indices(self) -> tuple[int, ...]:
        return tuple(
            detector.detector_index
            for detector in self.detectors
            if detector.disposition is DetectorDisposition.KEEP
        )

    def summary(self) -> dict[str, Any]:
        removed = [
            detector
            for detector in self.detectors
            if detector.disposition is DetectorDisposition.REMOVE
        ]
        reasons = {
            reason.value: sum(
                reason in detector.removal_reasons for detector in removed
            )
            for reason in RemovalReason
        }
        actions = {
            action.value: sum(span.action is action for span in self.repair_spans)
            for action in RepairAction
        }
        return {
            "source_path": self.source_path,
            "network_id": self.network_id,
            "sample_count": self.sample_count,
            "detector_count": len(self.detectors),
            "kept_detector_count": len(self.detectors) - len(removed),
            "removed_detector_count": len(removed),
            "repair_span_count": len(self.repair_spans),
            "repair_action_counts": actions,
            "removal_reason_counts": reasons,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "sample_interval": self.sample_interval,
            "policy_settings": dict(self.policy_settings),
            "detectors": [detector.to_dict() for detector in self.detectors],
        }


@dataclass(frozen=True)
class JumpCorrection:
    """One level-offset estimate applied from a jump sample onward."""

    jump_sample: int
    estimated_offset: float
    left_level: float
    right_level: float
    left_sample_count: int
    right_sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AppliedRepair:
    """Measured values recorded while executing one planned repair span."""

    network_id: int
    network_position: int
    detector_index: int
    start_sample: int
    stop_sample: int
    action: RepairAction
    issue_types: tuple[SampleIssue, ...]
    noise_scale: float
    jump_corrections: tuple[JumpCorrection, ...] = ()

    @property
    def duration_samples(self) -> int:
        return self.stop_sample - self.start_sample

    def to_dict(self) -> dict[str, Any]:
        return {
            "network_id": self.network_id,
            "network_position": self.network_position,
            "detector_index": self.detector_index,
            "start_sample": self.start_sample,
            "stop_sample": self.stop_sample,
            "action": self.action.value,
            "issue_types": [
                issue.name.lower() for issue in self.issue_types
            ],
            "noise_scale": self.noise_scale,
            "jump_corrections": [
                correction.to_dict() for correction in self.jump_corrections
            ],
        }


@dataclass(frozen=True)
class RemediationResult:
    """Cleaned retained-detector signal and the complete remediation audit."""

    cleaned_signal: np.ndarray
    detector_indices: np.ndarray
    source_network_positions: np.ndarray
    plan: RemediationPlan
    applied_repairs: tuple[AppliedRepair, ...] = ()

    @property
    def sample_count(self) -> int:
        return int(self.cleaned_signal.shape[0])

    @property
    def detector_count(self) -> int:
        return int(self.cleaned_signal.shape[1])

    def assert_mapmaker_ready(self) -> None:
        """Raise when retained output still contains unusable samples."""

        if self.cleaned_signal.ndim != 2:
            raise ValueError("cleaned_signal must be two-dimensional")
        if self.detector_indices.shape != (self.detector_count,):
            raise ValueError("detector_indices do not match cleaned_signal")
        if self.source_network_positions.shape != (self.detector_count,):
            raise ValueError(
                "source_network_positions do not match cleaned_signal"
            )
        if not np.isfinite(self.cleaned_signal).all():
            raise ValueError("cleaned_signal contains non-finite samples")

    def summary(self) -> dict[str, Any]:
        noise_scales = [repair.noise_scale for repair in self.applied_repairs]
        jump_count = sum(
            len(repair.jump_corrections) for repair in self.applied_repairs
        )
        return {
            **self.plan.summary(),
            "output_detector_count": self.detector_count,
            "applied_repair_count": len(self.applied_repairs),
            "applied_jump_correction_count": jump_count,
            "median_repair_noise_scale": (
                float(np.median(noise_scales)) if noise_scales else None
            ),
            "mapmaker_ready": bool(np.isfinite(self.cleaned_signal).all()),
        }

    def audit_dict(self) -> dict[str, Any]:
        """Return JSON-compatible metadata without embedding signal samples."""

        return {
            **self.summary(),
            "detector_indices": self.detector_indices.tolist(),
            "source_network_positions": self.source_network_positions.tolist(),
            "removed_detector_indices": list(
                self.plan.removed_detector_indices
            ),
            "applied_repairs": [
                repair.to_dict() for repair in self.applied_repairs
            ],
        }
