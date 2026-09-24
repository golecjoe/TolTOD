"""Apply a remediation plan to an in-memory sample-by-detector signal."""

from __future__ import annotations

import numpy as np

from ..detection.models import IssueCatalog
from ..detection.utilities import robust_mad
from ..errors import RemediationError
from .config import RemediationConfig
from .models import (
    AppliedRepair,
    DetectorDisposition,
    JumpCorrection,
    RemediationPlan,
    RemediationResult,
    RepairSpan,
)
from .policy import ACTIONABLE_BITS, build_remediation_plan


def remediate_array(
    signal: np.ndarray,
    catalog: IssueCatalog,
    *,
    config: RemediationConfig | None = None,
) -> RemediationResult:
    """Build and execute a remediation plan without modifying ``signal``."""

    settings = config or RemediationConfig()
    plan = build_remediation_plan(catalog, config=settings)
    return apply_remediation_plan(signal, catalog, plan)


def apply_remediation_plan(
    signal: np.ndarray,
    catalog: IssueCatalog,
    plan: RemediationPlan,
) -> RemediationResult:
    """Execute a previously built plan on a copy of the supplied signal."""

    values = _as_float_array(signal)
    _validate_inputs(values, catalog, plan)
    try:
        settings = RemediationConfig.from_mapping(plan.policy_settings)
    except (TypeError, ValueError) as error:
        raise RemediationError(
            "plan contains invalid remediation policy settings"
        ) from error

    kept_plans = [
        detector
        for detector in plan.detectors
        if detector.disposition is DetectorDisposition.KEEP
    ]
    cleaned = np.empty(
        (catalog.sample_count, len(kept_plans)),
        dtype=np.float64,
    )
    detector_indices = np.empty(len(kept_plans), dtype=np.int64)
    source_positions = np.empty(len(kept_plans), dtype=np.int64)
    audit: list[AppliedRepair] = []

    for output_position, detector_plan in enumerate(kept_plans):
        source_position = detector_plan.network_position
        detector = values[:, source_position].copy()
        bad = (
            catalog.sample_flags[:, source_position] & ACTIONABLE_BITS
        ) != 0
        planned = np.zeros(catalog.sample_count, dtype=bool)
        for span in detector_plan.repair_spans:
            planned[span.start_sample:span.stop_sample] = True
        if np.any(bad & ~planned):
            raise RemediationError(
                f"detector {detector_plan.detector_index} has bad samples "
                "outside its repair spans"
            )

        context_good = ~bad & ~planned & np.isfinite(detector)
        jump_audit = _apply_jump_corrections(
            detector,
            detector_plan.repair_spans,
            context_good,
            catalog.sample_interval,
            settings,
        )
        rng = _detector_rng(
            settings.noise_seed,
            catalog.network_id,
            detector_plan.detector_index,
        )
        for span in detector_plan.repair_spans:
            noise_scale = _apply_interpolation(
                detector,
                span,
                context_good,
                rng,
            )
            audit.append(
                AppliedRepair(
                    network_id=catalog.network_id,
                    network_position=source_position,
                    detector_index=detector_plan.detector_index,
                    start_sample=span.start_sample,
                    stop_sample=span.stop_sample,
                    action=span.action,
                    issue_types=span.issue_types,
                    noise_scale=noise_scale,
                    jump_corrections=tuple(
                        jump_audit[sample]
                        for sample in span.jump_samples
                    ),
                )
            )

        if not np.isfinite(detector).all():
            raise RemediationError(
                f"detector {detector_plan.detector_index} remains non-finite "
                "after remediation"
            )
        cleaned[:, output_position] = detector
        detector_indices[output_position] = detector_plan.detector_index
        source_positions[output_position] = source_position

    result = RemediationResult(
        cleaned_signal=cleaned,
        detector_indices=detector_indices,
        source_network_positions=source_positions,
        plan=plan,
        applied_repairs=tuple(audit),
    )
    result.assert_mapmaker_ready()
    return result


def _apply_jump_corrections(
    detector: np.ndarray,
    spans: tuple[RepairSpan, ...],
    context_good: np.ndarray,
    sample_interval: float,
    settings: RemediationConfig,
) -> dict[int, JumpCorrection]:
    jump_context = settings.samples_for(
        settings.jump_level_context_seconds,
        sample_interval,
        minimum=1,
    )
    jump_spans = sorted(
        (
            (jump_sample, span)
            for span in spans
            for jump_sample in span.jump_samples
        ),
        key=lambda item: item[0],
    )
    corrections: dict[int, JumpCorrection] = {}
    for jump_sample, _ in jump_spans:
        context_start = max(0, jump_sample - jump_context)
        context_stop = min(
            detector.size,
            jump_sample + jump_context + 1,
        )
        left_indices = np.flatnonzero(
            context_good[context_start:jump_sample]
        ) + context_start
        right_indices = np.flatnonzero(
            context_good[jump_sample:context_stop]
        ) + jump_sample
        if (
            left_indices.size < settings.minimum_jump_level_samples
            or right_indices.size < settings.minimum_jump_level_samples
        ):
            raise RemediationError(
                f"jump at sample {jump_sample} lacks enough good level samples"
            )
        left_level = float(np.median(detector[left_indices]))
        right_level = float(np.median(detector[right_indices]))
        offset = right_level - left_level
        if not np.isfinite(offset):
            raise RemediationError(
                f"jump at sample {jump_sample} produced a non-finite offset"
            )
        detector[jump_sample:] -= offset
        corrections[jump_sample] = JumpCorrection(
            jump_sample=jump_sample,
            estimated_offset=float(offset),
            left_level=left_level,
            right_level=right_level,
            left_sample_count=int(left_indices.size),
            right_sample_count=int(right_indices.size),
        )
    return corrections


def _apply_interpolation(
    detector: np.ndarray,
    span: RepairSpan,
    context_good: np.ndarray,
    rng: np.random.Generator,
) -> float:
    left_value = float(detector[span.left_anchor_sample])
    right_value = float(detector[span.right_anchor_sample])
    if not np.isfinite(left_value) or not np.isfinite(right_value):
        raise RemediationError(
            f"repair {span.start_sample}:{span.stop_sample} has a non-finite "
            "interpolation anchor"
        )

    context_indices = np.flatnonzero(
        context_good[span.noise_context_start:span.noise_context_stop]
    ) + span.noise_context_start
    if context_indices.size < span.available_noise_samples:
        raise RemediationError(
            f"repair {span.start_sample}:{span.stop_sample} has fewer good "
            "noise samples than its plan"
        )
    context_values = detector[context_indices]
    noise_scale = _detrended_noise_scale(context_indices, context_values)
    interpolation = np.linspace(
        left_value,
        right_value,
        span.duration_samples + 2,
        dtype=np.float64,
    )[1:-1]
    noise = rng.normal(0.0, noise_scale, size=span.duration_samples)
    detector[span.start_sample:span.stop_sample] = interpolation + noise
    return noise_scale


def _detrended_noise_scale(
    sample_indices: np.ndarray,
    values: np.ndarray,
) -> float:
    x = np.asarray(sample_indices, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    centered_x = x - np.mean(x)
    design = np.column_stack((centered_x, np.ones_like(centered_x)))
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - (slope * centered_x + intercept)
    scale = robust_mad(residual)
    if not np.isfinite(scale) or scale < 0:
        raise RemediationError("local noise estimation produced an invalid scale")
    return float(scale)


def _detector_rng(
    seed: int,
    network_id: int,
    detector_index: int,
) -> np.random.Generator:
    sequence = np.random.SeedSequence(
        [
            int(seed),
            int(network_id) & 0xFFFFFFFF,
            int(detector_index) & 0xFFFFFFFF,
        ]
    )
    return np.random.default_rng(sequence)


def _validate_inputs(
    signal: np.ndarray,
    catalog: IssueCatalog,
    plan: RemediationPlan,
) -> None:
    expected_shape = (catalog.sample_count, catalog.detector_count)
    if signal.ndim != 2 or signal.shape != expected_shape:
        raise RemediationError(
            f"signal shape {signal.shape} does not match catalog shape "
            f"{expected_shape}"
        )
    if plan.network_id != catalog.network_id:
        raise RemediationError("plan and catalog network IDs do not match")
    if plan.sample_count != catalog.sample_count:
        raise RemediationError("plan and catalog sample counts do not match")
    if not np.isclose(plan.sample_interval, catalog.sample_interval):
        raise RemediationError("plan and catalog sample intervals do not match")
    if len(plan.detectors) != catalog.detector_count:
        raise RemediationError("plan and catalog detector counts do not match")
    for position, detector_plan in enumerate(plan.detectors):
        if detector_plan.network_position != position:
            raise RemediationError("plan detector positions are not ordered")
        if detector_plan.detector_index != int(catalog.detector_indices[position]):
            raise RemediationError("plan and catalog detector indices do not match")
        for span in detector_plan.repair_spans:
            if not 0 <= span.start_sample < span.stop_sample <= plan.sample_count:
                raise RemediationError("plan contains an invalid repair interval")
            if span.network_position != position:
                raise RemediationError("repair span detector position is invalid")


def _as_float_array(values: np.ndarray) -> np.ndarray:
    array = np.ma.asarray(values)
    if np.ma.getmaskarray(array).any():
        return np.asarray(np.ma.filled(array, np.nan), dtype=np.float64)
    return np.asarray(array.data, dtype=np.float64)
