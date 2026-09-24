"""Fourier-domain step filtering for persistent level jumps."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d

from .config import DetectionConfig
from .models import IssueCatalog, IssueEvent, SampleIssue
from .utilities import fill_invalid_columns


def detect_jumps(
    signal: np.ndarray,
    catalog: IssueCatalog,
    config: DetectionConfig,
) -> None:
    """Detect step responses while rejecting nearby opposite-sign pairs."""

    width = config.samples_for(
        config.jump_filter_width_seconds,
        catalog.sample_interval,
        minimum=2,
    )
    pad = config.jump_filter_pad_widths * width
    if 2 * pad >= signal.shape[0]:
        return

    step_filter = _gaussian_step_filter(signal.shape[0], width)
    filter_spectrum = np.fft.rfft(step_filter)
    jump_mask = np.zeros(signal.shape, dtype=bool)
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
        filtered = np.fft.irfft(
            np.fft.rfft(filled.T, axis=1) * filter_spectrum[np.newaxis, :],
            n=signal.shape[0],
            axis=1,
        )

        contaminated = maximum_filter1d(
            invalid.T.astype(np.uint8),
            size=2 * pad + 1,
            axis=1,
            mode="constant",
            cval=1,
        ).astype(bool)
        contaminated[:, :pad] = True
        contaminated[:, -pad:] = True

        for local_position in range(filtered.shape[0]):
            usable = ~contaminated[local_position]
            if not np.any(usable):
                continue
            noise_scale = float(
                np.median(np.abs(filtered[local_position, usable]))
            )
            if (
                not np.isfinite(noise_scale)
                or noise_scale < config.minimum_scale
            ):
                continue
            threshold = config.jump_sigma * noise_scale
            search = filtered[local_position].copy()
            search[~usable] = 0.0

            while True:
                filter_index = int(np.argmax(np.abs(search)))
                response = float(search[filter_index])
                if abs(response) <= threshold:
                    break

                neighbor_start = max(0, filter_index - width)
                neighbor_stop = min(signal.shape[0], filter_index + width + 1)
                neighborhood = filtered[
                    local_position, neighbor_start:neighbor_stop
                ]
                if response > 0:
                    opposite_response = float(np.min(neighborhood))
                else:
                    opposite_response = float(np.max(neighborhood))
                opposite_ratio = abs(opposite_response / response)

                event_sample = min(filter_index + 1, signal.shape[0] - 1)
                if opposite_ratio <= config.jump_spike_ratio:
                    position = chunk_start + local_position
                    jump_mask[event_sample, position] = True
                    events.append(
                        IssueEvent(
                            issue_type=SampleIssue.JUMP,
                            network_id=catalog.network_id,
                            detector_index=int(
                                catalog.detector_indices[position]
                            ),
                            network_position=position,
                            start_sample=event_sample,
                            stop_sample=event_sample + 1,
                            severity=float(abs(response) / noise_scale),
                            details={
                                "filtered_response": response,
                                "filtered_noise_scale": noise_scale,
                                "opposite_response": opposite_response,
                                "opposite_response_ratio": opposite_ratio,
                                "filter_width_samples": width,
                            },
                        )
                    )

                suppress_start = max(0, filter_index - pad)
                suppress_stop = min(signal.shape[0], filter_index + pad + 1)
                search[suppress_start:suppress_stop] = 0.0

    if np.any(jump_mask):
        catalog.add_sample_mask(SampleIssue.JUMP, jump_mask)
        catalog.add_events(events)


def _gaussian_step_filter(sample_count: int, width: int) -> np.ndarray:
    """Return a unit-step-response filter for circular FFT convolution."""

    sample = np.arange(sample_count, dtype=np.float64)
    step_filter = np.exp(-0.5 * (sample / width) ** 2)
    step_filter -= np.exp(
        -0.5 * ((sample - sample_count) / width) ** 2
    )
    step_filter -= np.mean(step_filter)
    normalization = float(np.sum(np.abs(step_filter)) / 2.0)
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("Jump filter has no finite normalization")
    return step_filter / normalization
