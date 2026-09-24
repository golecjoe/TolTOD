"""Static per-network QC plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..detection.models import DetectorIssue, IssueCatalog, SampleIssue
from .aggregate import DetectorManifest
from .config import QCReportConfig


_TIMELINE_INPUT_BITS = (
    SampleIssue.NONFINITE
    | SampleIssue.SOURCE_FLAG
    | SampleIssue.SATURATION
    | SampleIssue.DROPOUT
    | SampleIssue.FLATLINE
    | SampleIssue.SPIKE
    | SampleIssue.JUMP
)


def plot_issue_histograms(
    manifest: DetectorManifest,
    path: str | Path,
    *,
    config: QCReportConfig | None = None,
) -> Path:
    """Plot per-detector event-count distributions for one network."""

    settings = config or QCReportConfig()
    destination = Path(path)
    rows = [row for row in manifest.rows if not row.excluded_from_thresholds]
    excluded_count = len(manifest.rows) - len(rows)
    series = (
        ("Spikes", np.asarray([row.spike_events for row in rows]), "#0072B2"),
        ("Jumps", np.asarray([row.jump_events for row in rows]), "#D55E00"),
        (
            "Other glitches",
            np.asarray([row.other_events for row in rows]),
            "#009E73",
        ),
    )

    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(13.5, 4.2),
            constrained_layout=True,
            sharey=False,
        )
        for axis, (title, values, color) in zip(axes, series):
            display_values, bins, overflow = _histogram_data(
                values, settings.histogram_max_bins
            )
            if values.size and np.any(values):
                axis.hist(
                    display_values,
                    bins=bins,
                    color=color,
                    edgecolor="white",
                    linewidth=0.7,
                )
            else:
                axis.text(
                    0.5,
                    0.5,
                    "No events detected",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="#475569",
                )
                axis.set_xlim(-0.5, 0.5)
            axis.set_title(title, fontweight="semibold")
            xlabel = "Events per detector"
            if overflow is not None:
                xlabel += f" (last bin includes ≥{overflow})"
            axis.set_xlabel(xlabel)
            axis.set_ylabel("Detector count")
            if values.size:
                summary = (
                    f"mean {np.mean(values):.2f}   median {np.median(values):.1f}"
                    f"   max {np.max(values):d}"
                )
            else:
                summary = "No eligible detectors"
            axis.text(
                0.02,
                0.96,
                summary,
                transform=axis.transAxes,
                va="top",
                fontsize=9,
            )
        subtitle = f"{len(rows)} detectors included"
        if excluded_count:
            subtitle += f"; {excluded_count} persistently source-flagged excluded"
        figure.suptitle(
            f"Network {manifest.network_id}: issue events per detector\n{subtitle}",
            fontsize=14,
            fontweight="semibold",
        )
        figure.savefig(destination, dpi=settings.figure_dpi, bbox_inches="tight")
        plt.close(figure)
    return destination


def plot_network_timeline(
    catalog: IssueCatalog,
    path: str | Path,
    *,
    config: QCReportConfig | None = None,
) -> Path:
    """Plot affected-detector fraction and correlated-event intervals."""

    settings = config or QCReportConfig()
    destination = Path(path)
    eligible = np.ones(catalog.detector_count, dtype=bool)
    if settings.exclude_persistent_source_flags and catalog.detector_metrics:
        for metrics in catalog.detector_metrics:
            if metrics.quality_flags & DetectorIssue.PERSISTENT_SOURCE_FLAG:
                eligible[metrics.network_position] = False
    eligible_count = int(np.count_nonzero(eligible))

    local_issues = (catalog.sample_flags & np.uint16(_TIMELINE_INPUT_BITS)) != 0
    if eligible_count:
        affected_count = np.count_nonzero(local_issues[:, eligible], axis=1)
        affected_percent = affected_count / eligible_count * 100.0
    else:
        affected_count = np.zeros(catalog.sample_count, dtype=int)
        affected_percent = np.zeros(catalog.sample_count, dtype=float)
    time_seconds = np.arange(catalog.sample_count) * catalog.sample_interval
    correlated_events = [
        event
        for event in catalog.events
        if event.issue_type == SampleIssue.CORRELATED
    ]
    correlation_threshold = float(
        catalog.detection_settings.get("correlated_detector_fraction", 0.10)
    ) * 100.0

    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, axis = plt.subplots(
            figsize=(13.5, 4.5),
            constrained_layout=True,
        )
        axis.plot(
            time_seconds,
            affected_percent,
            color="#0072B2",
            linewidth=0.8,
            label="Detectors with local issue candidates",
        )
        axis.fill_between(
            time_seconds,
            0,
            affected_percent,
            color="#0072B2",
            alpha=0.16,
            linewidth=0,
        )
        for index, event in enumerate(correlated_events):
            center = (event.start_sample + event.stop_sample) / 2
            center_seconds = center * catalog.sample_interval
            axis.axvspan(
                event.start_sample * catalog.sample_interval,
                event.stop_sample * catalog.sample_interval,
                color="#D62728",
                alpha=0.18,
                linewidth=0,
            )
            axis.axvline(
                center_seconds,
                color="#D62728",
                alpha=0.62,
                linewidth=0.9,
                label="Reported network-wide interval" if index == 0 else None,
            )
        if correlated_events:
            centers = np.asarray(
                [
                    (event.start_sample + event.stop_sample)
                    / 2
                    * catalog.sample_interval
                    for event in correlated_events
                ]
            )
            peaks = np.asarray(
                [
                    (event.severity or 0.0) * 100.0
                    for event in correlated_events
                ]
            )
            axis.scatter(
                centers,
                peaks,
                marker="v",
                s=28,
                color="#D62728",
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )
        axis.axhline(
            correlation_threshold,
            color="#6B7280",
            linestyle="--",
            linewidth=1.1,
            label=f"Correlation threshold ({correlation_threshold:g}%)",
        )
        axis.set_xlim(0, time_seconds[-1] if time_seconds.size else 1)
        axis.set_ylim(bottom=0)
        axis.set_xlabel("Detector time since observation start (s)")
        axis.set_ylabel("Affected active detectors (%)")
        axis.set_title(
            f"Network {catalog.network_id}: network-wide issue timeline\n"
            f"{len(correlated_events)} reported intervals; "
            f"{eligible_count} active detectors",
            fontweight="semibold",
        )
        if eligible_count:
            secondary = axis.secondary_yaxis(
                "right",
                functions=(
                    lambda percent: percent * eligible_count / 100.0,
                    lambda count: count / eligible_count * 100.0,
                ),
            )
            secondary.set_ylabel("Affected detector count")
        axis.legend(loc="upper right", frameon=True, fontsize=9)
        figure.savefig(destination, dpi=settings.figure_dpi, bbox_inches="tight")
        plt.close(figure)
    return destination


def _histogram_data(
    values: np.ndarray,
    maximum_bins: int,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    if values.size == 0:
        return values, np.asarray([-0.5, 0.5]), None
    maximum = int(np.max(values))
    if maximum + 1 <= maximum_bins:
        return values, np.arange(-0.5, maximum + 1.5, 1.0), None
    upper = max(10, int(np.ceil(np.percentile(values, 99))))
    upper = min(upper, maximum)
    displayed = np.minimum(values, upper)
    if upper + 1 <= maximum_bins:
        bins = np.arange(-0.5, upper + 1.5, 1.0)
    else:
        bins = np.linspace(-0.5, upper + 0.5, maximum_bins + 1)
    overflow = upper if maximum > upper else None
    return displayed, bins, overflow
