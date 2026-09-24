from __future__ import annotations

import json

import numpy as np

from toltod import (
    DetectionConfig,
    DetectorIssue,
    SampleIssue,
    detect_array_issues,
)
from toltod.detection.jumps import _gaussian_step_filter
from toltod.detection.spikes import _difference_of_gaussians_template
from toltod.detection.synthetic import make_synthetic_tod


def test_detects_direct_sample_issues() -> None:
    tod = make_synthetic_tod(sample_count=800, detector_count=5, seed=1)
    tod.inject_dropout(0, 100, 110)
    tod.inject_flatline(1, 200, 220)
    tod.inject_nonfinite(2, 300, 303)
    tod.inject_source_flag(3, 400, 405)

    catalog = detect_array_issues(
        tod.signal,
        tod.time,
        detector_indices=tod.detector_indices,
        source_flags=tod.source_flags,
    )

    assert catalog.mask_for(SampleIssue.DROPOUT)[100:110, 0].all()
    assert catalog.mask_for(SampleIssue.FLATLINE)[200:220, 1].all()
    assert catalog.mask_for(SampleIssue.NONFINITE)[300:303, 2].all()
    assert catalog.mask_for(SampleIssue.SOURCE_FLAG)[400:405, 3].all()


def test_detects_configured_saturation_without_modifying_input() -> None:
    tod = make_synthetic_tod(sample_count=300, detector_count=4, seed=6)
    tod.signal[100, 0] = 100.0
    original_signal = tod.signal.copy()
    original_flags = tod.source_flags.copy()

    catalog = detect_array_issues(
        tod.signal,
        tod.time,
        detector_indices=tod.detector_indices,
        source_flags=tod.source_flags,
        config=DetectionConfig(saturation_upper=50.0),
    )

    assert catalog.mask_for(SampleIssue.SATURATION)[100, 0]
    np.testing.assert_array_equal(tod.signal, original_signal)
    np.testing.assert_array_equal(tod.source_flags, original_flags)


def test_detects_spike_and_persistent_jump() -> None:
    tod = make_synthetic_tod(sample_count=1000, detector_count=6, seed=2)
    tod.inject_spike(0, 250, 30.0)
    tod.inject_jump(1, 500, 20.0)

    catalog = detect_array_issues(
        tod.signal,
        tod.time,
        detector_indices=tod.detector_indices,
        source_flags=tod.source_flags,
    )

    assert catalog.mask_for(SampleIssue.SPIKE)[250, 0]
    assert catalog.mask_for(SampleIssue.JUMP)[500, 1]
    assert not catalog.mask_for(SampleIssue.JUMP)[:, 0].any()


def test_jump_filter_is_normalized_step_contrast() -> None:
    step_filter = _gaussian_step_filter(1000, 10)

    assert np.isclose(step_filter.sum(), 0.0, atol=1.0e-12)
    assert np.isclose(step_filter[step_filter > 0].sum(), 1.0)
    assert np.isclose(step_filter[step_filter < 0].sum(), -1.0)


def test_fourier_jump_filter_recovers_transition_sample() -> None:
    tod = make_synthetic_tod(
        sample_count=2000,
        detector_count=1,
        seed=14,
    )
    tod.inject_jump(0, 1000, 30.0)

    catalog = detect_array_issues(
        tod.signal,
        tod.time,
        detector_indices=tod.detector_indices,
        source_flags=tod.source_flags,
    )
    events = [
        event
        for event in catalog.events
        if event.issue_type == SampleIssue.JUMP
    ]

    assert len(events) == 1
    assert events[0].start_sample == 1000
    assert events[0].details["filter_width_samples"] == 10
    assert events[0].details["opposite_response_ratio"] <= 0.5


def test_spike_template_is_zero_dc_and_unit_energy() -> None:
    config = DetectionConfig()
    template, negative_amplitude = _difference_of_gaussians_template(
        1.0 / 25.0,
        narrow_sigma_seconds=config.spike_narrow_sigma_seconds,
        wide_sigma_seconds=config.spike_wide_sigma_seconds,
        truncate=config.spike_template_truncate,
    )

    assert template.size == 7
    assert np.isclose(template.sum(), 0.0, atol=1.0e-12)
    assert np.isclose(np.linalg.norm(template), 1.0)
    assert template[template.size // 2] > 0
    assert np.count_nonzero(template > 0) == 1
    assert template.min() < 0
    assert negative_amplitude < 0


def test_fourier_spike_filter_prefers_template_over_broad_structure() -> None:
    config = DetectionConfig()
    sample_rate = 25.0
    template, _ = _difference_of_gaussians_template(
        1.0 / sample_rate,
        narrow_sigma_seconds=config.spike_narrow_sigma_seconds,
        wide_sigma_seconds=config.spike_wide_sigma_seconds,
        truncate=config.spike_template_truncate,
    )
    tod = make_synthetic_tod(
        sample_count=2000,
        detector_count=2,
        sample_rate=sample_rate,
        seed=11,
    )
    center = 1000
    radius = template.size // 2
    tod.signal[center - radius : center + radius + 1, 0] += 20.0 * template
    relative_time = tod.time - tod.time[center]
    tod.signal[:, 1] += 30.0 * np.exp(
        -0.5 * (relative_time / 0.40) ** 2
    )

    catalog = detect_array_issues(
        tod.signal,
        tod.time,
        detector_indices=tod.detector_indices,
        source_flags=tod.source_flags,
        config=config,
    )

    assert catalog.mask_for(SampleIssue.SPIKE)[center, 0]
    assert not catalog.mask_for(SampleIssue.SPIKE)[:, 1].any()


def test_persistent_jump_is_excluded_from_spike_filter() -> None:
    tod = make_synthetic_tod(
        sample_count=2000,
        detector_count=1,
        seed=13,
    )
    tod.inject_jump(0, 1000, 50.0)

    catalog = detect_array_issues(
        tod.signal,
        tod.time,
        detector_indices=tod.detector_indices,
        source_flags=tod.source_flags,
    )

    assert catalog.mask_for(SampleIssue.JUMP)[1000, 0]
    assert not catalog.mask_for(SampleIssue.SPIKE)[:, 0].any()


def test_detects_network_correlated_candidates() -> None:
    tod = make_synthetic_tod(sample_count=600, detector_count=10, seed=3)
    tod.inject_correlated_spike([0, 1, 2, 3], 300, 40.0)
    config = DetectionConfig(
        correlated_detector_fraction=0.30,
        correlated_min_detectors=3,
    )

    catalog = detect_array_issues(
        tod.signal,
        tod.time,
        detector_indices=tod.detector_indices,
        source_flags=tod.source_flags,
        config=config,
    )

    assert catalog.mask_for(SampleIssue.CORRELATED)[300, :4].all()
    events = [
        event
        for event in catalog.events
        if event.issue_type == SampleIssue.CORRELATED
    ]
    assert any(event.affected_detector_count >= 4 for event in events)


def test_persistently_source_flagged_detectors_do_not_create_correlation() -> None:
    tod = make_synthetic_tod(sample_count=300, detector_count=10, seed=7)
    tod.source_flags[:, :5] = 1
    config = DetectionConfig(
        correlated_detector_fraction=0.50,
        correlated_min_detectors=3,
    )

    catalog = detect_array_issues(
        tod.signal,
        tod.time,
        detector_indices=tod.detector_indices,
        source_flags=tod.source_flags,
        config=config,
    )

    assert not catalog.mask_for(SampleIssue.CORRELATED).any()


def test_calculates_detector_quality_diagnostics() -> None:
    tod = make_synthetic_tod(sample_count=1000, detector_count=10, seed=4)
    tod.signal[:, 1] *= 20.0
    tod.inject_source_flag(2, 0, tod.signal.shape[0])

    catalog = detect_array_issues(
        tod.signal,
        tod.time,
        detector_indices=tod.detector_indices,
        source_flags=tod.source_flags,
    )

    assert catalog.detector_metrics[1].quality_flags & DetectorIssue.EXCESS_NOISE
    assert (
        catalog.detector_metrics[2].quality_flags
        & DetectorIssue.PERSISTENT_SOURCE_FLAG
    )
    assert (
        catalog.detector_metrics[2].quality_flags
        & DetectorIssue.EXCESS_ISSUE_RATE
    )


def test_catalog_masks_are_compact_and_summary_is_serializable() -> None:
    tod = make_synthetic_tod(sample_count=100, detector_count=3, seed=5)
    catalog = detect_array_issues(
        tod.signal,
        tod.time,
        detector_indices=np.array([10, 11, 12]),
        source_flags=tod.source_flags,
    )

    assert catalog.sample_flags.dtype == np.uint16
    summary = catalog.summary()
    assert summary["network_id"] == 0
    assert summary["detector_count"] == 3
    assert "sample_detector_issue_counts" in summary
    json.dumps(summary)
