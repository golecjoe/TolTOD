"""Small, immutable metadata models used by the reader."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NetworkInfo:
    """Metadata describing one raw ``apt_nw`` detector group."""

    network_id: int
    detector_count: int
    detector_start: int
    detector_stop: int
    detector_indices_contiguous: bool
    array_ids: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible metadata."""

        return asdict(self)


@dataclass(frozen=True)
class ObservationInfo:
    """Metadata that can be obtained without loading a full signal array."""

    path: Path
    sample_count: int
    detector_count: int
    signal_variable: str
    signal_units: str | None
    time_variable: str
    time_units: str | None
    telescope_sample_interval: float | None
    raw_detector_sample_rate: float
    downsample_factor: int
    detector_sample_interval: float
    detector_sample_rate: float
    network_variable: str
    networks: tuple[NetworkInfo, ...]
    observation_number: int | None = None
    subobservation_number: int | None = None
    scan_number: int | None = None
    date_obs: str | None = None

    @property
    def network_ids(self) -> tuple[int, ...]:
        """Return raw network identifiers in sorted order."""

        return tuple(network.network_id for network in self.networks)

    @property
    def sample_interval(self) -> float:
        """Backward-compatible alias for the detector sample interval."""

        return self.detector_sample_interval

    @property
    def sample_rate(self) -> float:
        """Backward-compatible alias for the effective detector sample rate."""

        return self.detector_sample_rate

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible metadata."""

        result = asdict(self)
        result["path"] = str(self.path)
        result["networks"] = [network.to_dict() for network in self.networks]
        return result
