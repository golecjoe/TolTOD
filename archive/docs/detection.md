# Detection milestone

TolTOD detection is deliberately read-only. It produces an internal bitmask,
an event list, and detector diagnostics; it never changes signal samples and it
does not make detector-rejection decisions.

## Processing order

1. Direct validity problems: non-finite values and source `flags`.
2. Configured saturation limits, zero-valued dropouts, and exact flatlines.
3. Fourier-domain Gaussian step-filter responses, with nearby opposite-sign
   pairs rejected as spike-like events.
4. Positive difference-of-Gaussians matched-filter responses, excluding
   samples already explained by jumps or direct validity failures.
5. Network coincidence calculated from detector-local candidates.
6. Detector-level issue-rate and robust first-difference noise diagnostics.

This order allows later detectors to exclude samples already explained by a
more direct condition. Persistently source-flagged detectors are excluded from
the denominator used by the network-coincidence calculation.

## Time and sampling

All duration thresholds are configured in seconds. Detector time is generated
from sample index using

```text
effective detector rate = SAMPRATE / downsample_factor
detector time[i] = i / effective detector rate
```

The current examples have `SAMPRATE = 122.0703125 Hz` and were downsampled by
5, giving an effective stored-signal rate of `24.4140625 Hz`. The downsample
factor is explicit configuration because `CONFIG.DOWNSAMPLED` does not contain
a usable factor in these files. An undownsampled product should use a factor of
1.

`TelTime` is a separate telescope coordinate and is never used to set detector
windows or event durations. Its original units and values remain available via
`read_telescope_time` for later pointing work.

## Source flags

The two example files use a nonzero value in `flags` for unusable samples.
Detectors with `apt_flag == 1` are fully source-flagged and contain zero-valued
signals. The flag polarity is configurable with
`source_flag_nonzero_is_bad`; the detector-level `apt_flag` variable is not
itself used as a sample mask.

## Issue types

- `nonfinite`: NaN or infinite signal values.
- `source_flag`: unusable according to the input `flags` variable.
- `saturation`: outside configured lower or upper physical bounds.
- `dropout`: a sufficiently long run at the configured dropout value.
- `flatline`: a sufficiently long run with no sample-to-sample change.
- `spike`: a short, high-SNR response to the configured Fourier-domain
  difference-of-Gaussians template.
- `jump`: a high-SNR response to the configured Fourier-domain Gaussian step
  filter without a sufficiently strong nearby opposite-sign response.
- `correlated`: detector-local candidates coincident across a configured
  fraction of active network detectors.

Detector diagnostics are separate flags: excessive issue rate, excessive
noise relative to the network median, and persistent source flagging. These
states are evidence for later remediation policy, not rejection commands.

## Current performance and limitations

On the first example, network 0 contains 30,039 samples by 630 detectors. A
complete analysis takes roughly three seconds after reading the arrays and has
used about 650 MB peak resident memory in the development environment. Spike
and jump processing are detector-chunked; the full signal, source flag mask,
and issue bitmask are still resident for the network.

## Jump step filter

The jump filter compares Gaussian-smoothed levels on the two sides of each
sample. Its circular FFT kernel consists of a positive half-Gaussian at the
start and a negative half-Gaussian at the end, normalized so the positive and
negative areas are +1 and -1. Multiplication in Fourier space produces a large
one-sign response at a persistent step.

The default Gaussian width is 0.40 seconds, corresponding to 10 samples at the
current stored-signal rate. Each detector's threshold is 10 times the median
absolute filtered response. The detector repeatedly selects the strongest
remaining response, then suppresses a region extending two filter widths on
either side so the same transition is not counted more than once.

For every selected response, the detector searches one filter width on either
side for the largest response with the opposite sign. If its magnitude exceeds
half the primary response, the pair is classified as spike-like and is not
recorded as a jump. Accepted events record the filter response, detector noise
scale, opposite-sign response ratio, and effective filter width. Invalid input
samples are filled only for filtering, and outputs near invalid samples or the
circular FFT boundaries are excluded.

## Spike matched filter

The spike template is a narrow positive Gaussian plus a wider negative
Gaussian. The negative amplitude is calculated from the sampled kernels so
that the positive and negative areas cancel exactly. The resulting zero-DC
template is normalized to unit energy and applied along each detector with FFT
convolution. Because the template is real and symmetric, this convolution is
equivalent to matched-filter correlation.

The default Gaussian standard deviations are 0.02 and 0.04 seconds, and the
template is truncated at three times the wider standard deviation. At the
current 24.414 Hz stored-signal rate, these correspond to approximately 0.49
and 0.98 samples and produce a seven-sample template with only its central
sample positive. For every detector, the filtered response is centered by its
median and divided by a Gaussian-scaled MAD of the usable filtered samples.
`spike_sigma` is therefore the matched-filter SNR threshold; it defaults to 8.
Only positive responses are selected, matching the template's positive central
excursion. Above-threshold runs longer than `spike_max_width_seconds` are
discarded.

Invalid samples are linearly filled only for calculating the filter response.
Any output sample whose template footprint overlaps an invalid or previously
detected sample is excluded. Jumps are detected before spikes for this reason.
Each event records the exact peak sample, peak filtered response, filtered
noise scale, and template parameters.

The default thresholds are starting points. They have passed injected-event
tests but have not yet been calibrated against labeled TolTEC glitches or
astronomical signal-injection studies. Bright point-source crossings may
resemble the template, scan transitions may produce correlated candidates,
and a real common astronomical signal must not be classified as an instrument
glitch without additional context.

## Validation before remediation

Before any gap filling or correction is implemented, representative candidate
events should be plotted and reviewed by network and observation. Thresholds
should then be evaluated with synthetic issue injection, astronomical signal
injection, false-positive sampling, and repeatability across multiple
observations.
