"""Convert a read-only issue catalog into an immutable remediation plan."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from ..detection.models import IssueCatalog, SAMPLE_ISSUES, SampleIssue
from ..detection.utilities import true_runs
from .config import RemediationConfig
from .models import (
    DetectorDisposition,
    DetectorRemediation,
    RemediationPlan,
    RemovalReason,
    RepairAction,
    RepairSpan,
)


ACTIONABLE_ISSUES = tuple(
    issue for issue in SAMPLE_ISSUES if issue is not SampleIssue.CORRELATED
)
ACTIONABLE_BITS = np.uint16(
    sum(int(issue) for issue in ACTIONABLE_ISSUES)
)


def build_remediation_plan(
    catalog: IssueCatalog,
    *,
    config: RemediationConfig | None = None,
) -> RemediationPlan:
    """Plan interpolation, dejumping, and detector removal without mutation.

    The ``CORRELATED`` bit is a classification of another local issue and does
    not independently increase a detector's bad-sample fraction.
    """

    settings = config or RemediationConfig()
    _validate_catalog(catalog)
    spike_padding = settings.samples_for(
        settings.spike_padding_seconds,
        catalog.sample_interval,
    )
    jump_padding = settings.samples_for(
        settings.jump_padding_seconds,
        catalog.sample_interval,
    )
    other_padding = settings.samples_for(
        settings.other_padding_seconds,
        catalog.sample_interval,
    )
    noise_context = settings.samples_for(
        settings.noise_context_seconds,
        catalog.sample_interval,
        minimum=1,
    )
    jump_level_context = settings.samples_for(
        settings.jump_level_context_seconds,
        catalog.sample_interval,
        minimum=1,
    )

    actionable = (catalog.sample_flags & ACTIONABLE_BITS) != 0
    detectors: list[DetectorRemediation] = []
    for position, detector_index in enumerate(catalog.detector_indices):
        bad = actionable[:, position]
        bad_count = int(np.count_nonzero(bad))
        bad_fraction = bad_count / catalog.sample_count
        if bad_count > 0 and bad_fraction >= settings.detector_bad_fraction:
            detectors.append(
                _removed_detector(
                    catalog,
                    position,
                    int(detector_index),
                    bad_count,
                    bad_fraction,
                    (RemovalReason.BAD_SAMPLE_FRACTION,),
                )
            )
            continue

        expanded_bits = _expanded_issue_bits(
            catalog,
            position,
            spike_padding=spike_padding,
            jump_padding=jump_padding,
            other_padding=other_padding,
        )
        planned = expanded_bits != 0
        planned_count = int(np.count_nonzero(planned))
        repairs, removal_reasons = _build_detector_repairs(
            catalog,
            position,
            int(detector_index),
            bad,
            planned,
            expanded_bits,
            noise_context=noise_context,
            minimum_noise_samples=settings.minimum_noise_samples,
            jump_level_context=jump_level_context,
            minimum_jump_level_samples=settings.minimum_jump_level_samples,
        )
        if removal_reasons:
            detectors.append(
                _removed_detector(
                    catalog,
                    position,
                    int(detector_index),
                    bad_count,
                    bad_fraction,
                    tuple(removal_reasons),
                    planned_repair_sample_count=planned_count,
                )
            )
            continue
        detectors.append(
            DetectorRemediation(
                network_id=catalog.network_id,
                network_position=position,
                detector_index=int(detector_index),
                disposition=DetectorDisposition.KEEP,
                bad_sample_count=bad_count,
                bad_sample_fraction=bad_fraction,
                planned_repair_sample_count=planned_count,
                repair_spans=tuple(repairs),
            )
        )

    return RemediationPlan(
        network_id=catalog.network_id,
        sample_count=catalog.sample_count,
        sample_interval=catalog.sample_interval,
        detectors=tuple(detectors),
        policy_settings=asdict(settings),
        source_path=catalog.source_path,
    )


def _expanded_issue_bits(
    catalog: IssueCatalog,
    position: int,
    *,
    spike_padding: int,
    jump_padding: int,
    other_padding: int,
) -> np.ndarray:
    expanded = np.zeros(catalog.sample_count, dtype=np.uint16)
    for issue in ACTIONABLE_ISSUES:
        issue_mask = catalog.mask_for(issue)[:, position]
        if not np.any(issue_mask):
            continue
        if issue is SampleIssue.SPIKE:
            padding = spike_padding
        elif issue is SampleIssue.JUMP:
            padding = jump_padding
        else:
            padding = other_padding
        for start, stop in true_runs(issue_mask):
            repair_start = max(0, start - padding)
            repair_stop = min(catalog.sample_count, stop + padding)
            expanded[repair_start:repair_stop] |= np.uint16(issue)
    return expanded


def _build_detector_repairs(
    catalog: IssueCatalog,
    position: int,
    detector_index: int,
    bad: np.ndarray,
    planned: np.ndarray,
    expanded_bits: np.ndarray,
    *,
    noise_context: int,
    minimum_noise_samples: int,
    jump_level_context: int,
    minimum_jump_level_samples: int,
) -> tuple[list[RepairSpan], list[RemovalReason]]:
    repairs: list[RepairSpan] = []
    removal_reasons: list[RemovalReason] = []
    jump_mask = catalog.mask_for(SampleIssue.JUMP)[:, position]
    noise_eligible = ~bad & ~planned

    for start, stop in true_runs(planned):
        jump_samples = tuple(
            int(sample)
            for sample in np.flatnonzero(jump_mask[start:stop]) + start
        )
        if start == 0 or stop == catalog.sample_count:
            _append_unique(
                removal_reasons,
                RemovalReason.NO_BRACKETING_GOOD_SAMPLES,
            )
            continue
        left_anchor = start - 1
        right_anchor = stop
        if bad[left_anchor] or bad[right_anchor]:
            _append_unique(
                removal_reasons,
                RemovalReason.NO_BRACKETING_GOOD_SAMPLES,
            )
            continue

        context_start = max(0, start - noise_context)
        context_stop = min(catalog.sample_count, stop + noise_context)
        available_noise = int(
            np.count_nonzero(noise_eligible[context_start:context_stop])
        )
        if available_noise < minimum_noise_samples:
            _append_unique(
                removal_reasons,
                RemovalReason.INSUFFICIENT_NOISE_SAMPLES,
            )
            continue

        jump_context_is_valid = True
        for jump_sample in jump_samples:
            jump_context_start = max(0, jump_sample - jump_level_context)
            jump_context_stop = min(
                catalog.sample_count,
                jump_sample + jump_level_context + 1,
            )
            left_count = int(
                np.count_nonzero(
                    noise_eligible[jump_context_start:jump_sample]
                )
            )
            right_count = int(
                np.count_nonzero(
                    noise_eligible[jump_sample:jump_context_stop]
                )
            )
            if (
                left_count < minimum_jump_level_samples
                or right_count < minimum_jump_level_samples
            ):
                jump_context_is_valid = False
                _append_unique(
                    removal_reasons,
                    RemovalReason.INSUFFICIENT_JUMP_LEVEL_SAMPLES,
                )
        if not jump_context_is_valid:
            continue

        issue_value = int(np.bitwise_or.reduce(expanded_bits[start:stop]))
        issue_types = tuple(
            issue for issue in ACTIONABLE_ISSUES if issue_value & int(issue)
        )
        action = (
            RepairAction.DEJUMP_AND_INTERPOLATE_WITH_NOISE
            if jump_samples
            else RepairAction.INTERPOLATE_WITH_NOISE
        )
        repairs.append(
            RepairSpan(
                network_id=catalog.network_id,
                network_position=position,
                detector_index=detector_index,
                start_sample=start,
                stop_sample=stop,
                action=action,
                issue_types=issue_types,
                left_anchor_sample=left_anchor,
                right_anchor_sample=right_anchor,
                noise_context_start=context_start,
                noise_context_stop=context_stop,
                available_noise_samples=available_noise,
                jump_samples=jump_samples,
            )
        )
    return repairs, removal_reasons


def _removed_detector(
    catalog: IssueCatalog,
    position: int,
    detector_index: int,
    bad_count: int,
    bad_fraction: float,
    reasons: tuple[RemovalReason, ...],
    *,
    planned_repair_sample_count: int = 0,
) -> DetectorRemediation:
    return DetectorRemediation(
        network_id=catalog.network_id,
        network_position=position,
        detector_index=detector_index,
        disposition=DetectorDisposition.REMOVE,
        bad_sample_count=bad_count,
        bad_sample_fraction=bad_fraction,
        planned_repair_sample_count=planned_repair_sample_count,
        removal_reasons=reasons,
    )


def _append_unique(
    values: list[RemovalReason],
    value: RemovalReason,
) -> None:
    if value not in values:
        values.append(value)


def _validate_catalog(catalog: IssueCatalog) -> None:
    expected = (catalog.sample_count, catalog.detector_count)
    if catalog.sample_count < 1:
        raise ValueError("catalog must contain at least one sample")
    if catalog.sample_interval <= 0:
        raise ValueError("catalog sample_interval must be positive")
    if catalog.sample_flags.shape != expected:
        raise ValueError(
            f"catalog sample_flags shape {catalog.sample_flags.shape} does not "
            f"match {expected}"
        )
