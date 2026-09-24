"""Fourier-domain matched filtering for spike-shaped events."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import fftconvolve

from .config import DetectionConfig
from .models import IssueCatalog, IssueEvent, SampleIssue
from .utilities import fill_invalid_columns, retain_runs, robust_mad, true_runs


def detect_spikes(
    signal: np.ndarray,
    catalog: IssueCatalog,
    config: DetectionConfig,
) -> None:
    """Detect positive difference-of-Gaussians matched-filter responses."""

    template, negative_amplitude = _difference_of_gaussians_template(
        catalog.sample_interval,
        narrow_sigma_seconds=config.spike_narrow_sigma_seconds,
        wide_sigma_seconds=config.spike_wide_sigma_seconds,
        truncate=config.spike_template_truncate,
    )
    if template.size > signal.shape[0]:
        return

    maximum_width = config.samples_for(
        config.spike_max_width_seconds,
        catalog.sample_interval,
    )
    spikes = np.zeros(signal.shape, dtype=bool)
    events: list[IssueEvent] = []
    for chunk_start in range(0, signal.shape[1], config.detector_chunk_size):
        chunk_stop = min(
            signal.shape[1], chunk_start + config.detector_chunk_size
        )
        chunk = signal[:, chunk_start:chunk_stop]
        invalid = (
            catalog.sample_flags[:, chunk_start:chunk_stop] != 0
        ) | ~np.isfinite(chunk)
        filled = fill_invalid_columns(chunk, invalid)
        filtered = fftconvolve(
            filled,
            template[:, np.newaxis],
            mode="same",
            axes=0,
        )
        contaminated = maximum_filter1d(
            invalid.astype(np.uint8),
            size=template.size,
            axis=0,
            mode="constant",
            cval=1,
        ).astype(bool)
        candidate = np.zeros(chunk.shape, dtype=bool)
        scales = np.full(chunk.shape[1], np.nan, dtype=np.float64)
        centers = np.full(chunk.shape[1], np.nan, dtype=np.float64)
        for local_position in range(chunk.shape[1]):
            usable = ~contaminated[:, local_position]
            if not np.any(usable):
                continue
            center = float(np.median(filtered[usable, local_position]))
            scale = robust_mad(
                filtered[usable, local_position] - center
            )
            if not np.isfinite(scale) or scale < config.minimum_scale:
                continue
            centers[local_position] = center
            scales[local_position] = scale
            snr = (filtered[:, local_position] - center) / scale
            candidate[:, local_position] = (
                snr >= config.spike_sigma
            ) & usable

        chunk_spikes = retain_runs(candidate, maximum=maximum_width)
        spikes[:, chunk_start:chunk_stop] = chunk_spikes

        for local_position in range(chunk_spikes.shape[1]):
            position = chunk_start + local_position
            for start, stop in true_runs(chunk_spikes[:, local_position]):
                response = (
                    filtered[start:stop, local_position]
                    - centers[local_position]
                )
                peak_offset = int(np.argmax(response))
                peak_sample = start + peak_offset
                peak_response = float(response[peak_offset])
                severity = peak_response / scales[local_position]
                events.append(
                    IssueEvent(
                        issue_type=SampleIssue.SPIKE,
                        network_id=catalog.network_id,
                        detector_index=int(catalog.detector_indices[position]),
                        network_position=position,
                        start_sample=start,
                        stop_sample=stop,
                        severity=float(severity),
                        details={
                            "peak_sample": peak_sample,
                            "peak_filtered_response": peak_response,
                            "filtered_noise_scale": float(
                                scales[local_position]
                            ),
                            "template_narrow_sigma_seconds": (
                                config.spike_narrow_sigma_seconds
                            ),
                            "template_wide_sigma_seconds": (
                                config.spike_wide_sigma_seconds
                            ),
                            "template_negative_amplitude": (
                                negative_amplitude
                            ),
                        },
                    )
                )

    if np.any(spikes):
        catalog.add_sample_mask(SampleIssue.SPIKE, spikes)
        catalog.add_events(events)


def _difference_of_gaussians_template(
    sample_interval: float,
    *,
    narrow_sigma_seconds: float,
    wide_sigma_seconds: float,
    truncate: float,
) -> tuple[np.ndarray, float]:
    """Return a zero-DC, unit-energy difference-of-Gaussians template.

    The narrow Gaussian has unit positive peak. The wider Gaussian's negative
    amplitude is chosen from the sampled kernels so their discrete areas are
    equal. This removes DC and slowly varying baselines before the template is
    normalized for matched-filter SNR estimation.
    """

    radius = max(
        1,
        int(np.ceil(truncate * wide_sigma_seconds / sample_interval)),
    )
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    time = offsets * sample_interval
    narrow = np.exp(-0.5 * (time / narrow_sigma_seconds) ** 2)
    wide = np.exp(-0.5 * (time / wide_sigma_seconds) ** 2)
    negative_amplitude = float(np.sum(narrow) / np.sum(wide))
    template = narrow - negative_amplitude * wide
    template -= np.mean(template)
    energy = float(np.sqrt(np.sum(template**2)))
    if not np.isfinite(energy) or energy <= 0:
        raise ValueError("Spike template has no finite energy")
    return template / energy, -negative_amplitude
