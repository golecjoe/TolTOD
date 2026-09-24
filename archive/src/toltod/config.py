"""Configuration models for TolTOD ingestion."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    tomllib = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ReaderConfig:
    """Names and validation rules used to interpret a TolTEC NetCDF file."""

    signal_variable: str = "signal"
    network_variable: str = "apt_nw"
    time_variable: str = "TelTime"
    raw_detector_sample_rate_variable: str = "SAMPRATE"
    downsample_factor: int = 5
    sample_dimension: str = "n_pts"
    detector_dimension: str = "n_dets"
    validate_time_order: bool = True

    def __post_init__(self) -> None:
        if self.downsample_factor < 1:
            raise ValueError("downsample_factor must be at least 1")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ReaderConfig":
        """Construct a config while rejecting unknown keys."""

        known = {field.name for field in fields(cls)}
        unknown = set(values) - known
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown reader configuration key(s): {names}")
        return cls(**dict(values))

    @classmethod
    def from_toml(cls, path: str | Path) -> "ReaderConfig":
        """Load the ``[reader]`` section of a TOML configuration file."""

        if tomllib is None:
            raise RuntimeError("Reading TOML requires Python 3.11 or newer")
        with Path(path).open("rb") as stream:
            document = tomllib.load(stream)
        reader = document.get("reader", {})
        if not isinstance(reader, dict):
            raise ValueError("The TOML [reader] section must be a table")
        return cls.from_mapping(reader)
