# TolTOD

TolTOD is a Python framework for reading, inspecting, and eventually cleaning
TolTEC time-ordered data (TOD) for downstream maximum-likelihood mapmaking.

## Current scope

The project currently covers ingestion, read-only issue detection, and
in-memory remediation:

- validate the required NetCDF dimensions and variables;
- expose observation and network metadata without loading the full TOD;
- group detector columns by the raw `apt_nw` value;
- read selected signal and time slices for one network at a time; and
- inspect detector-level `apt_*` metadata.
- detect non-finite data, source flags, configurable saturation, dropouts,
  flatlines, short spikes, persistent jumps, and network-coincident issues;
- calculate detector-level issue-rate and robust-noise diagnostics; and
- produce issue catalogs without changing the input signal;
- generate per-network event histograms and correlated-event timelines; and
- export a sortable detector manifest, CSV data, and combined HTML QC report.
- convert detected issues into immutable repair and detector-removal plans.
- dejump retained detectors, replace planned intervals with locally noisy
  linear interpolation, and omit rejected detector columns from cleaned arrays.

Source signal arrays are never mutated. Cleaned-file writing and a command that
persists cleaned output are not implemented yet. Detection and remediation
thresholds are initial conservative defaults and need validation against
representative observations before production use.

## Installation

Create an environment and install the package in editable mode:

```bash
python -m pip install -e ".[dev]"
```

## Inspect a file

The inspect command reads only schema and coordinate-sized metadata:

```bash
toltod inspect data/example.nc
toltod inspect data/example.nc --json
toltod detect data/example.nc --network 0
toltod detect data/example.nc --network 0 --json
toltod report data/example.nc --network 0 --config configs/default.toml
toltod remediation-report data/example.nc --network 0 --config configs/default.toml
```

## Python API

Signal data is loaded only when `read_signal` is called:

```python
from toltod import open_tod

with open_tod("data/example.nc") as tod:
    print(tod.info)

    network = tod.network(0)
    first_second = network.read_signal(samples=slice(0, 25))
    detector_time = network.read_detector_time(samples=slice(0, 25))
    telescope_time = network.read_telescope_time(samples=slice(0, 25))
    apt = network.read_detector_metadata(["apt_uid", "apt_array"])
```

Issue detection is read-only:

```python
from toltod import detect_network_issues, open_tod

with open_tod("data/example.nc") as tod:
    catalog = detect_network_issues(tod.network(0))

print(catalog.summary())
```

The summary's `sample_detector_issue_counts` values count cells in the
sample-by-detector mask, not unique time samples.

The algorithms, assumptions, current performance, and validation requirements
are described in [docs/detection.md](docs/detection.md).

The report products and sample-impact highlighting rules are described in
[docs/visualizations.md](docs/visualizations.md).

The remediation policy models and current repair-planning rules are described
in [docs/remediation.md](docs/remediation.md).

Network IDs are not renumbered. If a file contains networks `0`, `2`, and `7`,
those are the IDs exposed by the API.

Networks are discovered from the integer values actually present in `apt_nw`;
there is no fixed list of required TolTEC networks. An inactive network may be
absent from one observation without causing an error, and networks such as `6`
and `10` will be exposed automatically when they appear in later data.

Detector time and telescope time are intentionally separate. Detector time is
generated from `SAMPRATE / downsample_factor`; `TelTime` is exposed only as a
telescope coordinate. The included configuration uses a downsample factor of
5 for the current example files. Set it to the value used to create each input
product (including `1` for an undownsampled product).

## Repository layout

```text
src/toltod/io/       NetCDF schema validation and lazy network reads
src/toltod/detection Issue models, synthetic data, and read-only detectors
src/toltod/remediation Repair policy, execution, and audit models
src/toltod/pipeline/ Reserved for future orchestration and parallelism
src/toltod/qc/       Network aggregation, plots, manifests, and reports
tests/               Synthetic unit tests and optional example-file checks
configs/             Versionable default settings
```
