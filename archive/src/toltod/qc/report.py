"""Generate static plots, CSV data, and sortable HTML network reports."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Mapping

import numpy as np

from ..detection.models import IssueCatalog, SampleIssue
from .aggregate import DetectorManifest, DetectorManifestRow, build_detector_manifest
from .config import QCReportConfig


@dataclass(frozen=True)
class NetworkQCReportPaths:
    """Files produced for one network report."""

    overview_plot: Path
    timeline_plot: Path
    manifest_csv: Path
    manifest_html: Path
    report_html: Path


def generate_network_qc_report(
    catalog: IssueCatalog,
    output_directory: str | Path,
    *,
    detector_metadata: Mapping[str, np.ndarray] | None = None,
    config: QCReportConfig | None = None,
) -> NetworkQCReportPaths:
    """Create all requested QC products for one detected network."""

    settings = config or QCReportConfig()
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"network_{catalog.network_id:02d}"
    paths = NetworkQCReportPaths(
        overview_plot=output / f"{stem}_overview.png",
        timeline_plot=output / f"{stem}_timeline.png",
        manifest_csv=output / f"{stem}_manifest.csv",
        manifest_html=output / f"{stem}_manifest.html",
        report_html=output / f"{stem}_report.html",
    )

    manifest = build_detector_manifest(
        catalog,
        detector_metadata=detector_metadata,
        config=settings,
    )
    from .plots import plot_issue_histograms, plot_network_timeline

    plot_issue_histograms(manifest, paths.overview_plot, config=settings)
    plot_network_timeline(catalog, paths.timeline_plot, config=settings)
    _write_manifest_csv(manifest, paths.manifest_csv)
    table = _manifest_table_html(manifest)
    paths.manifest_html.write_text(
        _html_document(
            title=f"Network {catalog.network_id} detector manifest",
            content=_manifest_intro(manifest, settings) + table,
        ),
        encoding="utf-8",
    )
    report_content = (
        f'<section class="figures">'
        f'<figure><img src="{escape(paths.overview_plot.name)}" '
        f'alt="Network {catalog.network_id} issue count histograms">'
        f'<figcaption>Per-detector event-count distributions</figcaption></figure>'
        f'<figure><img src="{escape(paths.timeline_plot.name)}" '
        f'alt="Network {catalog.network_id} network-wide issue timeline">'
        f'<figcaption>Network-wide issue timeline</figcaption></figure>'
        f'</section>'
        + _manifest_intro(manifest, settings)
        + table
    )
    paths.report_html.write_text(
        _html_document(
            title=f"TolTOD network {catalog.network_id} QC report",
            content=report_content,
        ),
        encoding="utf-8",
    )
    return paths


def _write_manifest_csv(manifest: DetectorManifest, path: Path) -> None:
    fieldnames = list(asdict(manifest.rows[0]).keys()) if manifest.rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for row in manifest.rows:
                values = asdict(row)
                values["quality_flags"] = ";".join(row.quality_flags)
                writer.writerow(values)


def _manifest_intro(
    manifest: DetectorManifest,
    config: QCReportConfig,
) -> str:
    threshold_labels = {
        "spike_samples": "spike",
        "jump_samples": "jump",
        "other_samples": "other-glitch",
        "affected_samples": "total affected",
    }
    threshold_items = "".join(
        "<li>"
        f"{escape(threshold_labels[column])}: "
        f"mean {threshold.mean:.2f} + {config.drastic_sigma:g}σ "
        f"({threshold.standard_deviation:.2f}) = "
        f"<strong>{threshold.threshold:.2f} samples</strong>"
        "</li>"
        for column, threshold in manifest.thresholds.items()
    )
    excluded = sum(row.excluded_from_thresholds for row in manifest.rows)
    return (
        '<section class="manifest-intro">'
        f'<h2>Detector manifest · network {manifest.network_id}</h2>'
        '<p>Issue cells show event count and detected sample impact. A red cell '
        f'exceeds the network population mean by more than {config.drastic_sigma:g} '
        'standard deviations in sample impact. Persistently source-flagged '
        f'detectors are muted and excluded from thresholds ({excluded} rows).</p>'
        '<p>“Other” combines non-finite, saturation, dropout, and flatline '
        'events. Source flags and correlated involvement remain separate.</p>'
        f'<ul class="thresholds">{threshold_items}</ul>'
        '</section>'
    )


def _manifest_table_html(manifest: DetectorManifest) -> str:
    rows = "".join(_manifest_row_html(row, manifest.sample_count) for row in manifest.rows)
    return (
        '<div class="table-wrap"><table id="detector-manifest">'
        '<thead><tr>'
        '<th data-key="network_position" data-type="number">Network position</th>'
        '<th data-key="detector_index" data-type="number">Source index</th>'
        '<th data-key="apt_uid" data-type="number">apt_uid</th>'
        '<th data-key="spike_samples" data-type="number">Spikes</th>'
        '<th data-key="jump_samples" data-type="number">Jumps</th>'
        '<th data-key="other_samples" data-type="number">Other glitches</th>'
        '<th data-key="source_flag_samples" data-type="number">Source flagged</th>'
        '<th data-key="correlated_samples" data-type="number">Correlated</th>'
        '<th data-key="affected_samples" data-type="number">Total affected</th>'
        '<th data-key="robust_noise" data-type="number">Robust noise</th>'
        '<th data-key="quality_flags" data-type="text">Diagnostics</th>'
        '</tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
    )


def _manifest_row_html(row: DetectorManifestRow, sample_count: int) -> str:
    row_class = "excluded" if row.excluded_from_thresholds else ""
    quality = ", ".join(flag.replace("_", " ") for flag in row.quality_flags)
    robust_noise = "" if row.robust_noise is None else f"{row.robust_noise:.6g}"
    return (
        f'<tr class="{row_class}">'
        f'{_plain_cell(row.network_position, "network_position")}'
        f'{_plain_cell(row.detector_index, "detector_index")}'
        f'{_plain_cell(row.apt_uid or "—", "apt_uid", row.apt_uid or -1)}'
        f'{_issue_cell(row.spike_events, row.spike_samples, sample_count, "spike_samples", row.spike_drastic)}'
        f'{_issue_cell(row.jump_events, row.jump_samples, sample_count, "jump_samples", row.jump_drastic)}'
        f'{_issue_cell(row.other_events, row.other_samples, sample_count, "other_samples", row.other_drastic)}'
        f'{_sample_cell(row.source_flag_samples, sample_count, "source_flag_samples")}'
        f'{_sample_cell(row.correlated_samples, sample_count, "correlated_samples")}'
        f'{_sample_cell(row.affected_samples, sample_count, "affected_samples", row.affected_drastic)}'
        f'{_plain_cell(robust_noise or "—", "robust_noise", row.robust_noise if row.robust_noise is not None else -1)}'
        f'{_plain_cell(quality or "—", "quality_flags", quality)}'
        '</tr>'
    )


def _plain_cell(value: object, key: str, sort_value: object | None = None) -> str:
    sortable = value if sort_value is None else sort_value
    return (
        f'<td data-key="{escape(key)}" data-sort="{escape(str(sortable))}">'
        f'{escape(str(value))}</td>'
    )


def _issue_cell(
    events: int,
    samples: int,
    sample_count: int,
    key: str,
    drastic: bool,
) -> str:
    cell_class = "issue-cell drastic" if drastic else "issue-cell"
    percent = samples / sample_count * 100.0
    return (
        f'<td class="{cell_class}" data-key="{key}" data-sort="{samples}">'
        f'<span class="primary">{events:,} events</span>'
        f'<span class="secondary">{samples:,} samples · {percent:.3f}%</span>'
        '</td>'
    )


def _sample_cell(
    samples: int,
    sample_count: int,
    key: str,
    drastic: bool = False,
) -> str:
    cell_class = "sample-cell drastic" if drastic else "sample-cell"
    percent = samples / sample_count * 100.0
    return (
        f'<td class="{cell_class}" data-key="{key}" data-sort="{samples}">'
        f'<span class="primary">{samples:,}</span>'
        f'<span class="secondary">{percent:.3f}%</span>'
        '</td>'
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
body {{ margin: 0 auto; max-width: 1600px; padding: 24px; background: #f8fafc; color: #172033; }}
h1 {{ margin: 0 0 22px; font-size: 1.65rem; }}
h2 {{ margin: 26px 0 8px; font-size: 1.25rem; }}
p {{ max-width: 1050px; line-height: 1.5; }}
.figures {{ display: grid; grid-template-columns: 1fr; gap: 20px; }}
figure {{ margin: 0; }}
figure img {{ display: block; width: 100%; background: white; border: 1px solid #cbd5e1; }}
figcaption {{ margin-top: 6px; color: #475569; font-size: .9rem; }}
.thresholds {{ display: flex; flex-wrap: wrap; gap: 8px 28px; padding-left: 22px; }}
.table-wrap {{ overflow: auto; max-height: 75vh; border: 1px solid #cbd5e1; background: white; }}
table {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: .86rem; }}
th {{ position: sticky; top: 0; z-index: 2; cursor: pointer; background: #e8eef6; text-align: left; }}
th, td {{ border-bottom: 1px solid #e2e8f0; padding: 8px 10px; white-space: nowrap; }}
tbody tr:hover td {{ background-color: #eff6ff; }}
td .primary, td .secondary {{ display: block; }}
td .secondary {{ color: #64748b; font-size: .76rem; margin-top: 2px; }}
td.drastic {{ background: #fecaca; color: #7f1d1d; font-weight: 700; }}
td.drastic .secondary {{ color: #991b1b; }}
tr.excluded td {{ background: #f1f5f9; color: #64748b; }}
tr.excluded td.drastic {{ background: #f1f5f9; color: #64748b; }}
@media (min-width: 1100px) {{ .figures {{ grid-template-columns: 1fr 1fr; }} }}
@media (prefers-color-scheme: dark) {{
  body {{ background: #101827; color: #e5e7eb; }}
  figure img, .table-wrap {{ background: #fff; border-color: #475569; }}
  figcaption, td .secondary {{ color: #aeb9ca; }}
  table {{ color: #e5e7eb; background: #172033; }}
  th {{ background: #263449; }}
  th, td {{ border-color: #334155; }}
  tbody tr:hover td {{ background-color: #24344d; }}
  td.drastic {{ background: #7f1d1d; color: #fee2e2; }}
  td.drastic .secondary {{ color: #fecaca; }}
  tr.excluded td, tr.excluded td.drastic {{ background: #26303f; color: #94a3b8; }}
}}
</style>
</head>
<body>
<h1>{escape(title)}</h1>
{content}
<script>
document.querySelectorAll('#detector-manifest th[data-key]').forEach((header) => {{
  let ascending = true;
  header.addEventListener('click', () => {{
    const table = header.closest('table');
    const body = table.querySelector('tbody');
    const key = header.dataset.key;
    const numeric = header.dataset.type === 'number';
    const rows = Array.from(body.querySelectorAll('tr'));
    rows.sort((a, b) => {{
      const av = a.querySelector(`td[data-key="${{key}}"]`).dataset.sort;
      const bv = b.querySelector(`td[data-key="${{key}}"]`).dataset.sort;
      const comparison = numeric ? Number(av) - Number(bv) : av.localeCompare(bv);
      return ascending ? comparison : -comparison;
    }});
    rows.forEach((row) => body.appendChild(row));
    ascending = !ascending;
  }});
}});
</script>
</body>
</html>
"""
