"""Synthetic TOD generators with recorded issue truth for detector tests."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import IssueEvent, SampleIssue


@dataclass
class SyntheticTOD:
    """Mutable synthetic network used to inject known detector issues."""

    signal: np.ndarray
    time: np.ndarray
    source_flags: np.ndarray
    network_id: int = 0
    detector_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    truth: list[IssueEvent] = field(default_factory=list)

    def inject_spike(
        self,
        detector: int,
        sample: int,
        amplitude: float,
        *,
        width: int = 1,
    ) -> None:
        self.signal[sample : sample + width, detector] += amplitude
        self._record(SampleIssue.SPIKE, detector, sample, sample + width)

    def inject_jump(self, detector: int, sample: int, amplitude: float) -> None:
        self.signal[sample:, detector] += amplitude
        self._record(SampleIssue.JUMP, detector, sample, sample + 1)

    def inject_dropout(
        self,
        detector: int,
        start: int,
        stop: int,
        *,
        value: float = 0.0,
    ) -> None:
        self.signal[start:stop, detector] = value
        self._record(SampleIssue.DROPOUT, detector, start, stop)

    def inject_flatline(self, detector: int, start: int, stop: int) -> None:
        self.signal[start:stop, detector] = self.signal[start - 1, detector]
        self._record(SampleIssue.FLATLINE, detector, start, stop)

    def inject_source_flag(self, detector: int, start: int, stop: int) -> None:
        self.source_flags[start:stop, detector] = 1
        self._record(SampleIssue.SOURCE_FLAG, detector, start, stop)

    def inject_nonfinite(self, detector: int, start: int, stop: int) -> None:
        self.signal[start:stop, detector] = np.nan
        self._record(SampleIssue.NONFINITE, detector, start, stop)

    def inject_correlated_spike(
        self,
        detectors: list[int],
        sample: int,
        amplitude: float,
    ) -> None:
        for detector in detectors:
            self.inject_spike(detector, sample, amplitude)

    def _record(
        self,
        issue: SampleIssue,
        detector: int,
        start: int,
        stop: int,
    ) -> None:
        self.truth.append(
            IssueEvent(
                issue_type=issue,
                network_id=self.network_id,
                detector_index=int(self.detector_indices[detector]),
                network_position=detector,
                start_sample=start,
                stop_sample=stop,
            )
        )


def make_synthetic_tod(
    *,
    sample_count: int = 1024,
    detector_count: int = 8,
    sample_rate: float = 25.0,
    noise_std: float = 1.0,
    seed: int = 0,
    network_id: int = 0,
) -> SyntheticTOD:
    """Create independent Gaussian detector noise with no injected issues."""

    if sample_count < 2 or detector_count < 1 or sample_rate <= 0:
        raise ValueError("sample_count, detector_count, and sample_rate must be positive")
    rng = np.random.default_rng(seed)
    signal = rng.normal(0.0, noise_std, size=(sample_count, detector_count))
    time = np.arange(sample_count, dtype=np.float64) / sample_rate
    return SyntheticTOD(
        signal=signal,
        time=time,
        source_flags=np.zeros_like(signal, dtype=np.uint8),
        network_id=network_id,
        detector_indices=np.arange(detector_count, dtype=np.int64),
    )

