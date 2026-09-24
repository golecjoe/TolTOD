"""Memory-bounded raw-versus-cleaned remediation QC report."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Mapping

import numpy as np

from ..detection.models import IssueCatalog, IssueEvent, SampleIssue
from ..remediation.models import AppliedRepair, RemediationResult
from .config import RemediationQCConfig


@dataclass(frozen=True)
class RemediationQCReportPaths:
    """Files produced for one network's remediation snippet report."""

    report_html: Path
    snippets_directory: Path
    raw_plots: tuple[Path, ...]
    cleaned_plots: tuple[Path, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "report_html": str(self.report_html),
            "snippets_directory": str(self.snippets_directory),
            "raw_plots": [str(path) for path in self.raw_plots],
            "cleaned_plots": [str(path) for path in self.cleaned_plots],
        }


@dataclass(frozen=True)
class _SnippetCandidate:
    category: str
    repair: AppliedRepair
    event_sample: int
    severity: float


def generate_remediation_qc_report(
    raw_signal: np.ndarray,
    catalog: IssueCatalog,
    result: RemediationResult,
    output_directory: str | Path,
    *,
    detector_metadata: Mapping[str, np.ndarray] | None = None,
    signal_units: str | None = None,
    config: RemediationQCConfig | None = None,
) -> RemediationQCReportPaths:
    """Write a two-column page of bounded raw and cleaned event snippets."""

    settings = config or RemediationQCConfig()
    raw = _as_float_array(raw_signal)
    _validate_inputs(raw, catalog, result, detector_metadata)

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"network_{catalog.network_id:02d}_remediation"
    snippets = output / f"{stem}_snippets"
    snippets.mkdir(parents=True, exist_ok=True)
    report_path = output / f"{stem}_report.html"

    candidates = _select_candidates(catalog, result, settings)
    output_column_by_source = {
        int(source_position): output_position
        for output_position, source_position in enumerate(
            result.source_network_positions
        )
    }
    apt_uid = None
    if detector_metadata is not None:
        apt_uid = detector_metadata.get("apt_uid")

    raw_paths: list[Path] = []
    cleaned_paths: list[Path] = []
    rows: list[str] = []
    for index, candidate in enumerate(candidates, start=1):
        repair = candidate.repair
        output_position = output_column_by_source[repair.network_position]
        context_start, context_stop = _context_bounds(
            candidate.event_sample,
            catalog.sample_count,
            catalog.sample_interval,
            settings.context_seconds,
        )
        raw_values = raw[
            context_start:context_stop,
            repair.network_position,
        ]
        cleaned_values = result.cleaned_signal[
            context_start:context_stop,
            output_position,
        ]
        local_repairs = tuple(
            item
            for item in result.applied_repairs
            if item.network_position == repair.network_position
            and item.start_sample < context_stop
            and item.stop_sample > context_start
        )
        y_limits = _shared_y_limits(raw_values, cleaned_values)
        base = (
            f"{index:03d}_{candidate.category}_detector_"
            f"{repair.detector_index}_sample_{candidate.event_sample}"
        )
        raw_path = snippets / f"{base}_raw.png"
        cleaned_path = snippets / f"{base}_cleaned.png"
        _plot_snippet(
            raw_values,
            context_start=context_start,
            center_sample=candidate.event_sample,
            sample_interval=catalog.sample_interval,
            repair=repair,
            local_repairs=local_repairs,
            destination=raw_path,
            title="Raw",
            color="#0072B2",
            signal_units=signal_units,
            y_limits=y_limits,
            dpi=settings.figure_dpi,
        )
        _plot_snippet(
            cleaned_values,
            context_start=context_start,
            center_sample=candidate.event_sample,
            sample_interval=catalog.sample_interval,
            repair=repair,
            local_repairs=local_repairs,
            destination=cleaned_path,
            title="Cleaned",
            color="#009E73",
            signal_units=signal_units,
            y_limits=y_limits,
            dpi=settings.figure_dpi,
        )
        raw_paths.append(raw_path)
        cleaned_paths.append(cleaned_path)
        uid = ""
        if apt_uid is not None:
            uid = _format_metadata_value(apt_uid[repair.network_position])
        rows.append(
            _snippet_row_html(
                candidate,
                raw_path.relative_to(output),
                cleaned_path.relative_to(output),
                uid=uid,
                sample_interval=catalog.sample_interval,
            )
        )

    content = _report_intro(catalog, result, candidates, settings)
    if rows:
        content += '<section class="snippet-list">' + "".join(rows) + "</section>"
    else:
        content += (
            '<p class="empty">No applied repairs were available for snippets. '
            "Detectors removed in full are reported in the summary above.</p>"
        )
    report_path.write_text(
        _html_document(
            title=f"TolTOD network {catalog.network_id} remediation QC",
            content=content,
        ),
        encoding="utf-8",
    )
    return RemediationQCReportPaths(
        report_html=report_path,
        snippets_directory=snippets,
        raw_plots=tuple(raw_paths),
        cleaned_plots=tuple(cleaned_paths),
    )


def _select_candidates(
    catalog: IssueCatalog,
    result: RemediationResult,
    config: RemediationQCConfig,
) -> list[_SnippetCandidate]:
    grouped: dict[str, list[_SnippetCandidate]] = {
        "spike": [],
        "jump": [],
        "other": [],
    }
    events_by_position: dict[int, list[IssueEvent]] = {}
    for event in catalog.events:
        if event.network_position is not None:
            events_by_position.setdefault(event.network_position, []).append(event)

    for repair in result.applied_repairs:
        if SampleIssue.JUMP in repair.issue_types:
            category = "jump"
        elif SampleIssue.SPIKE in repair.issue_types:
            category = "spike"
        else:
            category = "other"
        matching = [
            event
            for event in events_by_position.get(repair.network_position, [])
            if event.issue_type in repair.issue_types
            and event.start_sample < repair.stop_sample
            and event.stop_sample > repair.start_sample
        ]
        best_event = max(
            matching,
            key=lambda event: (
                float(event.severity or 0.0),
                event.duration_samples,
            ),
            default=None,
        )
        if best_event is not None:
            event_sample = int(
                best_event.details.get(
                    "peak_sample",
                    (best_event.start_sample + best_event.stop_sample - 1) // 2,
                )
            )
            severity = float(best_event.severity or 0.0)
        elif repair.jump_corrections:
            event_sample = repair.jump_corrections[0].jump_sample
            severity = abs(repair.jump_corrections[0].estimated_offset)
        else:
            event_sample = (repair.start_sample + repair.stop_sample - 1) // 2
            severity = float(repair.duration_samples)
        grouped[category].append(
            _SnippetCandidate(
                category=category,
                repair=repair,
                event_sample=event_sample,
                severity=severity,
            )
        )

    limits = {
        "spike": config.max_spike_examples,
        "jump": config.max_jump_examples,
        "other": config.max_other_examples,
    }
    selected: list[_SnippetCandidate] = []
    for category in ("spike", "jump", "other"):
        selected.extend(_select_diverse(grouped[category], limits[category]))
    return selected


def _select_diverse(
    candidates: list[_SnippetCandidate],
    limit: int,
) -> list[_SnippetCandidate]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            -item.severity,
            -item.repair.duration_samples,
            item.repair.detector_index,
            item.event_sample,
        ),
    )
    if limit <= 0:
        return []
    selected: list[_SnippetCandidate] = []
    used_detectors: set[int] = set()
    for candidate in ordered:
        if candidate.repair.detector_index in used_detectors:
            continue
        selected.append(candidate)
        used_detectors.add(candidate.repair.detector_index)
        if len(selected) == limit:
            return selected
    for candidate in ordered:
        if candidate in selected:
            continue
        selected.append(candidate)
        if len(selected) == limit:
            break
    return selected


def _plot_snippet(
    values: np.ndarray,
    *,
    context_start: int,
    center_sample: int,
    sample_interval: float,
    repair: AppliedRepair,
    local_repairs: tuple[AppliedRepair, ...],
    destination: Path,
    title: str,
    color: str,
    signal_units: str | None,
    y_limits: tuple[float, float],
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample_indices = np.arange(
        context_start,
        context_start + values.size,
        dtype=np.int64,
    )
    relative_time = (sample_indices - center_sample) * sample_interval
    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, axis = plt.subplots(
            figsize=(6.2, 3.1),
            constrained_layout=True,
        )
        axis.plot(relative_time, values, color=color, linewidth=1.0)
        additional_labeled = False
        for local_repair in local_repairs:
            is_selected = local_repair is repair
            label = None
            if is_selected:
                label = "Selected repair"
            elif not additional_labeled:
                label = "Other repair in window"
                additional_labeled = True
            axis.axvspan(
                (local_repair.start_sample - center_sample) * sample_interval,
                (local_repair.stop_sample - center_sample) * sample_interval,
                color="#E69F00" if is_selected else "#9CA3AF",
                alpha=0.20 if is_selected else 0.15,
                linewidth=0,
                label=label,
            )
        axis.axvline(0.0, color="#6B7280", linewidth=0.9, alpha=0.8)
        axis.set_xlim(relative_time[0], relative_time[-1])
        axis.set_ylim(*y_limits)
        axis.set_xlabel("Time relative to event (s)")
        ylabel = "Signal"
        if signal_units:
            ylabel += f" ({signal_units})"
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontweight="semibold")
        axis.legend(loc="best", fontsize=8, frameon=True)
        figure.savefig(destination, dpi=dpi, bbox_inches="tight")
        plt.close(figure)


def _context_bounds(
    center_sample: int,
    sample_count: int,
    sample_interval: float,
    context_seconds: float,
) -> tuple[int, int]:
    half_width = max(1, int(round(context_seconds / sample_interval / 2.0)))
    start = max(0, center_sample - half_width)
    stop = min(sample_count, center_sample + half_width + 1)
    return start, stop


def _shared_y_limits(
    raw_values: np.ndarray,
    cleaned_values: np.ndarray,
) -> tuple[float, float]:
    combined = np.concatenate((raw_values, cleaned_values))
    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        return (-1.0, 1.0)
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if lower == upper:
        padding = max(1.0, abs(lower) * 0.05)
    else:
        padding = (upper - lower) * 0.08
    return lower - padding, upper + padding


def _snippet_row_html(
    candidate: _SnippetCandidate,
    raw_path: Path,
    cleaned_path: Path,
    *,
    uid: str,
    sample_interval: float,
) -> str:
    repair = candidate.repair
    issue_label = ", ".join(
        issue.name.lower().replace("_", " ") for issue in repair.issue_types
    )
    uid_text = f" · apt_uid {escape(uid)}" if uid else ""
    jump_text = ""
    if repair.jump_corrections:
        offsets = ", ".join(
            f"{item.estimated_offset:.6g}" for item in repair.jump_corrections
        )
        jump_text = f" · jump offset {escape(offsets)}"
    heading = candidate.category.capitalize()
    meta = (
        f"detector {repair.detector_index}{uid_text} · sample "
        f"{candidate.event_sample} · detector time "
        f"{candidate.event_sample * sample_interval:.3f} s · "
        f"issues {escape(issue_label)} · noise σ {repair.noise_scale:.6g}"
        f"{jump_text}"
    )
    return (
        '<article class="snippet">'
        f"<h2>{escape(heading)}</h2>"
        f'<p class="meta">{meta}</p>'
        '<div class="comparison">'
        f'<figure><img src="{escape(raw_path.as_posix())}" '
        f'alt="Raw {escape(candidate.category)} event for detector '
        f'{repair.detector_index}"><figcaption>Raw</figcaption></figure>'
        f'<figure><img src="{escape(cleaned_path.as_posix())}" '
        f'alt="Cleaned {escape(candidate.category)} event for detector '
        f'{repair.detector_index}"><figcaption>Cleaned</figcaption></figure>'
        "</div></article>"
    )


def _report_intro(
    catalog: IssueCatalog,
    result: RemediationResult,
    candidates: list[_SnippetCandidate],
    config: RemediationQCConfig,
) -> str:
    category_counts = {
        category: sum(item.category == category for item in candidates)
        for category in ("spike", "jump", "other")
    }
    return (
        '<section class="summary">'
        f"<p>Network {catalog.network_id}: {result.detector_count:,} retained "
        f"detectors, {len(result.plan.removed_detector_indices):,} removed, "
        f"and {len(result.applied_repairs):,} applied repair intervals.</p>"
        f"<p>Showing {category_counts['spike']} spike, "
        f"{category_counts['jump']} jump, and "
        f"{category_counts['other']} other examples with "
        f"{config.context_seconds:g} seconds of context. Raw and cleaned plots "
        "use the same vertical scale. Orange shading marks the selected repair; "
        "grey shading marks additional repaired intervals in the same "
        "window.</p></section>"
    )


def _html_document(*, title: str, content: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
:root {{ color-scheme: light dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
body {{ margin: 0 auto; max-width: 1500px; padding: 24px; background: #f8fafc; color: #172033; }}
h1 {{ margin: 0 0 16px; font-size: 1.65rem; }}
h2 {{ margin: 0 0 4px; font-size: 1.15rem; }}
p {{ line-height: 1.45; }}
.summary {{ margin-bottom: 26px; }}
.summary p {{ margin: 5px 0; max-width: 1100px; }}
.snippet {{ padding: 20px 0 26px; border-top: 1px solid #cbd5e1; }}
.meta {{ margin: 0 0 12px; color: #475569; }}
.comparison {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 18px; }}
figure {{ margin: 0; min-width: 0; }}
figure img {{ display: block; width: 100%; background: white; border: 1px solid #cbd5e1; }}
figcaption {{ margin-top: 5px; color: #475569; font-size: .9rem; }}
.empty {{ padding: 28px 0; border-top: 1px solid #cbd5e1; }}
@media (max-width: 760px) {{ .comparison {{ grid-template-columns: 1fr; }} }}
@media (prefers-color-scheme: dark) {{
  body {{ background: #101827; color: #e5e7eb; }}
  .meta, figcaption {{ color: #aeb9ca; }}
  .snippet, .empty {{ border-color: #475569; }}
  figure img {{ border-color: #475569; }}
}}
</style>
</head>
<body>
<h1>{escape(title)}</h1>
{content}
</body>
</html>
"""


def _validate_inputs(
    raw: np.ndarray,
    catalog: IssueCatalog,
    result: RemediationResult,
    detector_metadata: Mapping[str, np.ndarray] | None,
) -> None:
    expected = (catalog.sample_count, catalog.detector_count)
    if raw.ndim != 2 or raw.shape != expected:
        raise ValueError(
            f"raw_signal shape {raw.shape} does not match catalog shape {expected}"
        )
    if result.plan.network_id != catalog.network_id:
        raise ValueError("result and catalog network IDs do not match")
    if result.sample_count != catalog.sample_count:
        raise ValueError("result and catalog sample counts do not match")
    result.assert_mapmaker_ready()
    if detector_metadata is not None:
        for name, values in detector_metadata.items():
            if len(values) != catalog.detector_count:
                raise ValueError(
                    f"{name} metadata must contain one value per detector"
                )


def _as_float_array(values: np.ndarray) -> np.ndarray:
    array = np.ma.asarray(values)
    if np.ma.getmaskarray(array).any():
        return np.asarray(np.ma.filled(array, np.nan), dtype=np.float64)
    return np.asarray(array.data, dtype=np.float64)


def _format_metadata_value(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return ""
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"
