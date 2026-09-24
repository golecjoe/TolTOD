# Network QC visualizations

The `report` command runs read-only detection for one network and writes five
products:

```text
network_00_overview.png
network_00_timeline.png
network_00_manifest.csv
network_00_manifest.html
network_00_report.html
```

The combined report references the two PNG figures and contains the sortable
manifest. The standalone manifest is useful when only the detector table is
needed, while CSV preserves all values and highlighting decisions for later
analysis.

## Event histograms

The overview contains per-detector distributions for spike, jump, and other
event counts. Persistently source-flagged detectors are excluded. “Other” is the
sum of non-finite, saturation, dropout, and flatline events; source flags and
correlated classifications are reported separately.

When a very large outlier would compress the useful part of a histogram, the
last bin includes the upper one percent of the range and is labeled as an
overflow bin. The annotation still reports the actual maximum.

## Network timeline

The timeline uses generated detector time, not `TelTime`. It plots the fraction
of active detectors with local issue candidates and shades/marks every interval
classified as network-correlated. Persistently source-flagged detectors are
excluded from both the numerator and denominator. The right axis expresses the
same values as detector count.

## Detector manifest

Each detector row is keyed by its source column index and network position.
`apt_uid` is included as metadata but is not treated as unique. Issue cells show
both event count and detected sample count/fraction.

Red highlighting is based only on sample impact. For each of spike samples,
jump samples, other-glitch samples, and total affected samples, the report
calculates the population statistics across eligible detectors in that network:

```text
drastic threshold = mean sample count + 3 × population standard deviation
```

A cell is red only when its sample count is strictly greater than its threshold.
The threshold, mean, and standard deviation are printed above the table.
Persistently source-flagged rows are grey, are excluded from the statistics, and
cannot receive drastic highlighting.

The correlated bit is a classification of existing issue samples, so it is not
added again when calculating total affected samples.

## Configuration

The `[qc]` table controls the sigma multiplier, maximum histogram bins, output
DPI, and exclusion of persistently source-flagged detectors. The default sigma
multiplier is 3.0.

```bash
python -m toltod.cli report data/example.nc \
  --network 0 \
  --config configs/default.toml \
  --output-dir outputs/qc/example
```

## Remediation snippets

The `remediation-report` command detects and repairs one network in memory,
then creates a page of representative raw-versus-cleaned snippets. Each event
is one row with raw data on the left and the corresponding cleaned samples on
the right. Both panels use the same vertical scale. Orange shading marks the
selected repair, while grey shading identifies any additional repair interval
within the displayed context. Detector time is generated from the effective
detector sample rate; `TelTime` is not used.

Examples are selected independently for spikes, jumps, and other repaired
issues. The highest-severity events are prioritized while preferring different
detectors before showing multiple examples from one detector. Fully removed
detectors do not have cleaned snippets and are summarized at the top of the
page instead.

The report is memory bounded: only the configured short context slice is sent
to Matplotlib, one image is written at a time, and each figure is immediately
closed. Full timestream arrays are not embedded in the HTML.

```bash
python -m toltod.cli remediation-report data/example.nc \
  --network 0 \
  --config configs/default.toml \
  --output-dir outputs/remediation_qc/example
```

The `[remediation_qc]` table controls the total context duration, per-category
example limits, and image DPI. Defaults are ten seconds of context, six spike
examples, six jump examples, four other examples, and 120 DPI.
