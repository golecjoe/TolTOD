from __future__ import annotations

import json

import numpy as np

from toltod import (
    DetectorDisposition,
    IssueCatalog,
    RemediationConfig,
    RemovalReason,
    RepairAction,
    SampleIssue,
    apply_remediation_plan,
    build_remediation_plan,
    detect_array_issues,
    remediate_array,
)
from toltod.detection.synthetic import make_synthetic_tod


def _catalog(*, sample_count: int = 100, detector_count: int = 3) -> IssueCatalog:
    return IssueCatalog.empty(
        network_id=4,
        sample_count=sample_count,
        detector_indices=np.arange(100, 100 + detector_count),
        sample_interval=0.04,
        source_path="synthetic.nc",
    )


def _mask(catalog: IssueCatalog, start: int, stop: int, detector: int) -> np.ndarray:
    mask = np.zeros_like(catalog.sample_flags, dtype=bool)
    mask[start:stop, detector] = True
    return mask


def test_plans_padded_spike_interpolation_with_noise_context() -> None:
    catalog = _catalog()
    catalog.add_sample_mask(SampleIssue.SPIKE, _mask(catalog, 50, 51, 0))

    plan = build_remediation_plan(catalog)
    detector = plan.detectors[0]
    span = detector.repair_spans[0]

    assert detector.disposition is DetectorDisposition.KEEP
    assert detector.bad_sample_count == 1
    assert detector.planned_repair_sample_count == 5
    assert span.action is RepairAction.INTERPOLATE_WITH_NOISE
    assert (span.start_sample, span.stop_sample) == (48, 53)
    assert (span.left_anchor_sample, span.right_anchor_sample) == (47, 53)
    assert span.issue_types == (SampleIssue.SPIKE,)
    assert span.available_noise_samples == 95


def test_plans_dejump_before_interpolating_around_jump() -> None:
    catalog = _catalog()
    catalog.add_sample_mask(SampleIssue.JUMP, _mask(catalog, 40, 41, 1))

    plan = build_remediation_plan(catalog)
    span = plan.detectors[1].repair_spans[0]

    assert span.action is RepairAction.DEJUMP_AND_INTERPOLATE_WITH_NOISE
    assert (span.start_sample, span.stop_sample) == (38, 43)
    assert span.jump_samples == (40,)
    assert span.issue_types == (SampleIssue.JUMP,)


def test_merges_overlapping_spike_and_jump_windows() -> None:
    catalog = _catalog()
    catalog.add_sample_mask(SampleIssue.SPIKE, _mask(catalog, 50, 51, 0))
    catalog.add_sample_mask(SampleIssue.JUMP, _mask(catalog, 52, 53, 0))

    plan = build_remediation_plan(catalog)
    detector = plan.detectors[0]

    assert len(detector.repair_spans) == 1
    span = detector.repair_spans[0]
    assert span.action is RepairAction.DEJUMP_AND_INTERPOLATE_WITH_NOISE
    assert span.issue_types == (SampleIssue.SPIKE, SampleIssue.JUMP)
    assert span.jump_samples == (52,)


def test_removes_detector_at_configured_bad_sample_fraction() -> None:
    catalog = _catalog()
    catalog.add_sample_mask(SampleIssue.SOURCE_FLAG, _mask(catalog, 0, 10, 2))

    plan = build_remediation_plan(
        catalog,
        config=RemediationConfig(detector_bad_fraction=0.10),
    )
    detector = plan.detectors[2]

    assert detector.disposition is DetectorDisposition.REMOVE
    assert detector.bad_sample_fraction == 0.10
    assert detector.removal_reasons == (RemovalReason.BAD_SAMPLE_FRACTION,)
    assert detector.repair_spans == ()


def test_zero_bad_fraction_threshold_keeps_clean_detectors() -> None:
    catalog = _catalog()

    plan = build_remediation_plan(
        catalog,
        config=RemediationConfig(detector_bad_fraction=0.0),
    )

    assert all(
        detector.disposition is DetectorDisposition.KEEP
        for detector in plan.detectors
    )


def test_correlated_classification_does_not_create_repair_or_rejection() -> None:
    catalog = _catalog()
    catalog.add_sample_mask(SampleIssue.CORRELATED, _mask(catalog, 0, 100, 0))

    plan = build_remediation_plan(catalog)
    detector = plan.detectors[0]

    assert detector.disposition is DetectorDisposition.KEEP
    assert detector.bad_sample_count == 0
    assert detector.repair_spans == ()


def test_removes_detector_when_interpolation_has_no_good_bracket() -> None:
    catalog = _catalog()
    catalog.add_sample_mask(SampleIssue.SPIKE, _mask(catalog, 0, 1, 0))

    plan = build_remediation_plan(catalog)
    detector = plan.detectors[0]

    assert detector.disposition is DetectorDisposition.REMOVE
    assert detector.removal_reasons == (
        RemovalReason.NO_BRACKETING_GOOD_SAMPLES,
    )


def test_removes_detector_when_noise_context_is_too_sparse() -> None:
    catalog = _catalog(sample_count=30)
    catalog.add_sample_mask(SampleIssue.SPIKE, _mask(catalog, 15, 16, 0))
    config = RemediationConfig(minimum_noise_samples=28)

    plan = build_remediation_plan(catalog, config=config)
    detector = plan.detectors[0]

    assert detector.disposition is DetectorDisposition.REMOVE
    assert detector.removal_reasons == (
        RemovalReason.INSUFFICIENT_NOISE_SAMPLES,
    )


def test_removes_detector_when_jump_level_context_is_too_sparse() -> None:
    catalog = _catalog(sample_count=40)
    catalog.add_sample_mask(SampleIssue.JUMP, _mask(catalog, 10, 11, 0))
    config = RemediationConfig(minimum_jump_level_samples=9)

    plan = build_remediation_plan(catalog, config=config)
    detector = plan.detectors[0]

    assert detector.disposition is DetectorDisposition.REMOVE
    assert detector.removal_reasons == (
        RemovalReason.INSUFFICIENT_JUMP_LEVEL_SAMPLES,
    )


def test_plan_summary_and_details_are_json_serializable() -> None:
    catalog = _catalog()
    catalog.add_sample_mask(SampleIssue.SPIKE, _mask(catalog, 50, 51, 0))

    plan = build_remediation_plan(catalog)

    assert plan.summary()["repair_span_count"] == 1
    assert plan.removed_detector_indices == ()
    assert plan.kept_detector_indices == (100, 101, 102)
    json.dumps(plan.to_dict())


def test_executes_spike_interpolation_without_modifying_input() -> None:
    catalog = _catalog(sample_count=200, detector_count=2)
    rng = np.random.default_rng(12)
    signal = rng.normal(0.0, 1.0, size=(200, 2))
    signal[100, 0] += 100.0
    original = signal.copy()
    catalog.add_sample_mask(SampleIssue.SPIKE, _mask(catalog, 100, 101, 0))

    result = remediate_array(
        signal,
        catalog,
        config=RemediationConfig(detector_bad_fraction=0.50, noise_seed=8),
    )
    span = result.plan.detectors[0].repair_spans[0]

    np.testing.assert_array_equal(signal, original)
    np.testing.assert_array_equal(
        result.cleaned_signal[:span.start_sample, 0],
        original[:span.start_sample, 0],
    )
    np.testing.assert_array_equal(
        result.cleaned_signal[span.stop_sample:, 0],
        original[span.stop_sample:, 0],
    )
    assert np.max(np.abs(result.cleaned_signal[span.start_sample:span.stop_sample, 0])) < 10
    assert result.applied_repairs[0].noise_scale > 0
    result.assert_mapmaker_ready()


def test_executes_jump_correction_before_interpolation() -> None:
    catalog = _catalog(sample_count=300, detector_count=1)
    rng = np.random.default_rng(4)
    signal = rng.normal(0.0, 0.2, size=(300, 1))
    signal[150:, 0] += 10.0
    original = signal.copy()
    catalog.add_sample_mask(SampleIssue.JUMP, _mask(catalog, 150, 151, 0))

    result = remediate_array(
        signal,
        catalog,
        config=RemediationConfig(detector_bad_fraction=0.50, noise_seed=3),
    )
    repair = result.applied_repairs[0]
    correction = repair.jump_corrections[0]
    cleaned = result.cleaned_signal[:, 0]

    np.testing.assert_array_equal(signal, original)
    assert abs(correction.estimated_offset - 10.0) < 0.5
    assert abs(np.median(cleaned[130:145]) - np.median(cleaned[155:170])) < 0.3
    assert repair.action is RepairAction.DEJUMP_AND_INTERPOLATE_WITH_NOISE
    result.assert_mapmaker_ready()


def test_repairs_nonfinite_span_and_removes_rejected_detector_column() -> None:
    catalog = _catalog(sample_count=100, detector_count=2)
    rng = np.random.default_rng(9)
    signal = rng.normal(size=(100, 2))
    signal[40:43, 0] = np.nan
    catalog.add_sample_mask(
        SampleIssue.NONFINITE,
        _mask(catalog, 40, 43, 0),
    )
    catalog.add_sample_mask(
        SampleIssue.SOURCE_FLAG,
        _mask(catalog, 0, 10, 1),
    )

    result = remediate_array(
        signal,
        catalog,
        config=RemediationConfig(detector_bad_fraction=0.10),
    )

    assert result.cleaned_signal.shape == (100, 1)
    np.testing.assert_array_equal(result.detector_indices, np.array([100]))
    np.testing.assert_array_equal(
        result.source_network_positions,
        np.array([0]),
    )
    assert result.plan.removed_detector_indices == (101,)
    assert np.isfinite(result.cleaned_signal).all()


def test_remediation_noise_is_reproducible_and_seeded() -> None:
    catalog = _catalog(sample_count=200, detector_count=1)
    rng = np.random.default_rng(14)
    signal = rng.normal(size=(200, 1))
    signal[100, 0] += 50.0
    catalog.add_sample_mask(SampleIssue.SPIKE, _mask(catalog, 100, 101, 0))

    first = remediate_array(
        signal,
        catalog,
        config=RemediationConfig(noise_seed=5),
    )
    second = remediate_array(
        signal,
        catalog,
        config=RemediationConfig(noise_seed=5),
    )
    different = remediate_array(
        signal,
        catalog,
        config=RemediationConfig(noise_seed=6),
    )

    np.testing.assert_array_equal(first.cleaned_signal, second.cleaned_signal)
    assert not np.array_equal(first.cleaned_signal, different.cleaned_signal)


def test_applies_prebuilt_plan_and_emits_json_serializable_audit() -> None:
    catalog = _catalog(sample_count=200, detector_count=1)
    signal = np.random.default_rng(18).normal(size=(200, 1))
    signal[80, 0] += 40.0
    catalog.add_sample_mask(SampleIssue.SPIKE, _mask(catalog, 80, 81, 0))
    plan = build_remediation_plan(catalog)

    result = apply_remediation_plan(signal, catalog, plan)

    assert result.summary()["mapmaker_ready"] is True
    assert result.summary()["applied_repair_count"] == 1
    json.dumps(result.audit_dict())


def test_detection_to_remediation_end_to_end() -> None:
    tod = make_synthetic_tod(
        sample_count=2000,
        detector_count=3,
        seed=23,
    )
    tod.inject_spike(0, 500, 30.0)
    tod.inject_jump(1, 1000, 20.0)
    original = tod.signal.copy()
    catalog = detect_array_issues(
        tod.signal,
        tod.time,
        detector_indices=tod.detector_indices,
        source_flags=tod.source_flags,
    )

    result = remediate_array(tod.signal, catalog)

    np.testing.assert_array_equal(tod.signal, original)
    assert result.detector_count == 3
    assert np.isfinite(result.cleaned_signal).all()
    assert abs(result.cleaned_signal[500, 0]) < 10.0
    assert (
        abs(
            np.median(result.cleaned_signal[950:990, 1])
            - np.median(result.cleaned_signal[1010:1050, 1])
        )
        < 1.0
    )
