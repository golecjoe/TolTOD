# Remediation policy and execution

The remediation layer converts a read-only `IssueCatalog` into an immutable
`RemediationPlan`, then applies that plan to an in-memory copy of the signal.
The source array is never modified. Rejected detector columns are omitted from
the returned cleaned array. Cleaned NetCDF writing is not implemented yet.

## Policy decisions

Primary issue bits—non-finite values, source flags, saturation, dropouts,
flatlines, spikes, and jumps—count toward the bad-sample fraction. The
`correlated` bit is a classification of existing local issues and is excluded
so it cannot create a repair or inflate the fraction on its own.

When a detector's bad-sample fraction is greater than or equal to
`detector_bad_fraction`, the complete detector timestream is marked for
removal. The default is 10 percent.

For retained detectors, each spike interval is expanded by
`spike_padding_seconds` on both sides. Jump intervals are expanded by
`jump_padding_seconds`. Overlapping windows are merged. A merged window that
contains at least one jump is assigned `dejump_and_interpolate_with_noise`;
other windows use `interpolate_with_noise`.

Every interpolation plan records:

- the half-open replacement interval;
- the good sample immediately before and after that interval;
- the surrounding context interval used for noise estimation;
- the number of currently available good noise samples; and
- every jump sample that must be corrected before interpolation.

The requested noise context extends `noise_context_seconds` on each side. All
detected bad samples and every planned replacement sample are excluded from the
available-noise count. If a repair lacks two bracketing good samples or fewer
than `minimum_noise_samples` remain, the detector is marked for removal because
the requested interpolation method cannot be applied safely.

Jump offsets use robust median levels from
`jump_level_context_seconds` immediately around the transition, excluding all
bad and planned-replacement samples. At least `minimum_jump_level_samples` must
be available on each side. Every accepted offset is subtracted from the jump
sample through the end of the detector before interpolation is performed.

For each replacement interval, the executor fits and removes a linear trend
from the surrounding good context, estimates its Gaussian noise scale with a
median absolute deviation, and draws zero-mean noise using `noise_seed`. The
draw is deterministically keyed by network and detector ID, so repeated runs
with the same configuration produce identical cleaned samples.

The replacement baseline is a straight line between the recorded good samples
immediately before and after the interval. The estimated noise is added to this
line. After every retained detector is repaired, the executor verifies that the
output is two-dimensional and entirely finite.

## Python API

```python
from toltod import (
    RemediationConfig,
    detect_network_issues,
    open_tod,
    remediate_array,
)

with open_tod("data/example.nc") as tod:
    network = tod.network(0)
    signal = network.read_signal()
    catalog = detect_network_issues(network)

policy = RemediationConfig.from_toml("configs/default.toml")
result = remediate_array(signal, catalog, config=policy)
result.assert_mapmaker_ready()
print(result.summary())
```

`result.cleaned_signal` is sample-by-retained-detector data.
`result.detector_indices` contains the corresponding original detector column
indices, while `result.source_network_positions` maps each output column back
to its position in the input network. `result.audit_dict()` contains repair and
rejection metadata without embedding the signal array.

The next output milestone will use this result to subset detector metadata,
write a lighter cleaned product, and save the audit dictionary beside it.

Before cleaned products are persisted, `remediation-report` generates bounded
raw-versus-cleaned snippets for visual review. See
[`visualizations.md`](visualizations.md) for the report layout and command.
