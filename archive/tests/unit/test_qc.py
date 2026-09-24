from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from toltod import (
    DetectionConfig,
    QCReportConfig,
    RemediationQCConfig,
    build_detector_manifest,
    detect_array_issues,
    generate_network_qc_report,
    generate_remediation_qc_report,
    remediate_array,
)
from toltod.detection.synthetic import make_synthetic_tod


def _catalog_with_sample_impact_outlier():
    tod = make_synthetic_tod(
        sample_count=900,
        detector_count=20,
        seed=12,
    )
    for sample in range(100, 700, 40):
        tod.inject_spike(0, sample, 50.0)
    catalog = detect_array_issues(
        tod.signal,
        tod.time,
        detector_indices=np.arange(100, 120),
        source_flags=tod.source_flags,
        config=DetectionConfig(correlated_detector_fraction=1.0),
    )
    return catalog


def test_manifest_colors_issue_cells_from_sample_impact() -> None:
    catalog = _catalog_with_sample_impact_outlier()
    manifest = build_detector_manifest(
        catalog,
        detector_metadata={"apt_uid": np.arange(200, 220)},
    )

    first = manifest.rows[0]
    assert first.spike_events >= 10
    assert first.spike_samples > manifest.thresholds["spike_samples"].threshold
    assert first.spike_drastic
    assert not manifest.rows[1].spike_drastic
    assert first.detector_index == 100
    assert first.apt_uid == "200"


def test_network_report_writes_plots_sortable_manifest_and_csv(
    tmp_path: Path,
) -> None:
    catalog = _catalog_with_sample_impact_outlier()
    paths = generate_network_qc_report(
        catalog,
        tmp_path,
        detector_metadata={"apt_uid": np.arange(200, 220)},
        config=QCReportConfig(figure_dpi=100),
    )

    for path in vars(paths).values():
        assert path.exists()
        assert path.stat().st_size > 100
    assert paths.overview_plot.read_bytes().startswith(b"\x89PNG")
    assert paths.timeline_plot.read_bytes().startswith(b"\x89PNG")

    report = paths.report_html.read_text(encoding="utf-8")
    manifest = paths.manifest_html.read_text(encoding="utf-8")
    assert "Detector manifest" in report
    assert 'class="issue-cell drastic"' in report
    assert " + 3σ" in report
    assert "addEventListener('click'" in manifest

    with paths.manifest_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 20
    assert rows[0]["detector_index"] == "100"
    assert rows[0]["spike_drastic"] == "True"


def test_remediation_report_writes_two_column_raw_cleaned_snippets(
    tmp_path: Path,
) -> None:
    tod = make_synthetic_tod(
        sample_count=1200,
        detector_count=4,
        seed=24,
    )
    tod.inject_spike(0, 250, 30.0)
    tod.inject_jump(1, 500, 20.0)
    tod.inject_dropout(2, 800, 808)
    catalog = detect_array_issues(
        tod.signal,
        tod.time,
        detector_indices=np.arange(100, 104),
        source_flags=tod.source_flags,
    )
    result = remediate_array(tod.signal, catalog)

    paths = generate_remediation_qc_report(
        tod.signal,
        catalog,
        result,
        tmp_path,
        detector_metadata={"apt_uid": np.arange(200, 204)},
        signal_units="mJy/beam",
        config=RemediationQCConfig(
            max_spike_examples=1,
            max_jump_examples=1,
            max_other_examples=1,
            figure_dpi=90,
        ),
    )

    assert paths.report_html.exists()
    assert paths.snippets_directory.is_dir()
    assert len(paths.raw_plots) == 3
    assert len(paths.cleaned_plots) == 3
    for path in paths.raw_plots + paths.cleaned_plots:
        assert path.read_bytes().startswith(b"\x89PNG")
    report = paths.report_html.read_text(encoding="utf-8")
    assert report.count('class="comparison"') == 3
    assert "Raw" in report
    assert "Cleaned" in report
    assert "apt_uid 200" in report
    assert "10 seconds of context" in report


def test_remediation_report_handles_no_applied_repairs(
    tmp_path: Path,
) -> None:
    tod = make_synthetic_tod(
        sample_count=200,
        detector_count=2,
        seed=25,
    )
    catalog = detect_array_issues(
        tod.signal,
        tod.time,
        detector_indices=tod.detector_indices,
        source_flags=tod.source_flags,
    )
    result = remediate_array(tod.signal, catalog)

    paths = generate_remediation_qc_report(
        tod.signal,
        catalog,
        result,
        tmp_path,
        config=RemediationQCConfig(figure_dpi=90),
    )

    assert paths.raw_plots == ()
    assert paths.cleaned_plots == ()
    assert "No applied repairs" in paths.report_html.read_text(encoding="utf-8")
