"""Configuration for read-only TOD issue detection."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    tomllib = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DetectionConfig:
    """Thresholds used by the first-pass issue detectors.

    Duration settings are expressed in seconds and converted to sample counts
    using the time coordinate supplied to the detector.
    """

    source_flag_variable: str | None = "flags"
    source_flag_nonzero_is_bad: bool = True
    require_source_flags: bool = False
    permanent_source_flag_fraction: float = 0.95

    dropout_value: float = 0.0
    dropout_atol: float = 0.0
    dropout_min_seconds: float = 0.20
    flatline_atol: float = 0.0
    flatline_min_seconds: float = 0.50
    saturation_lower: float | None = None
    saturation_upper: float | None = None

    spike_sigma: float = 8.0
    spike_narrow_sigma_seconds: float = 0.02
    spike_wide_sigma_seconds: float = 0.04
    spike_template_truncate: float = 3.0
    spike_max_width_seconds: float = 0.12

    jump_sigma: float = 10.0
    jump_filter_width_seconds: float = 0.40
    jump_filter_pad_widths: int = 2
    jump_spike_ratio: float = 0.50

    correlated_detector_fraction: float = 0.10
    correlated_min_detectors: int = 3
    correlated_min_seconds: float = 0.04

    excess_issue_fraction: float = 0.10
    excess_noise_factor: float = 5.0
    minimum_scale: float = 1.0e-12
    detector_chunk_size: int = 64

    def __post_init__(self) -> None:
        fractions = {
            "permanent_source_flag_fraction": self.permanent_source_flag_fraction,
            "jump_spike_ratio": self.jump_spike_ratio,
            "correlated_detector_fraction": self.correlated_detector_fraction,
            "excess_issue_fraction": self.excess_issue_fraction,
        }
        for name, value in fractions.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        positive = {
            "dropout_min_seconds": self.dropout_min_seconds,
            "flatline_min_seconds": self.flatline_min_seconds,
            "spike_sigma": self.spike_sigma,
            "spike_narrow_sigma_seconds": self.spike_narrow_sigma_seconds,
            "spike_wide_sigma_seconds": self.spike_wide_sigma_seconds,
            "spike_template_truncate": self.spike_template_truncate,
            "spike_max_width_seconds": self.spike_max_width_seconds,
            "jump_sigma": self.jump_sigma,
            "jump_filter_width_seconds": self.jump_filter_width_seconds,
            "correlated_min_seconds": self.correlated_min_seconds,
            "excess_noise_factor": self.excess_noise_factor,
            "minimum_scale": self.minimum_scale,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.spike_wide_sigma_seconds <= self.spike_narrow_sigma_seconds:
            raise ValueError(
                "spike_wide_sigma_seconds must exceed "
                "spike_narrow_sigma_seconds"
            )
        if self.correlated_min_detectors < 1:
            raise ValueError("correlated_min_detectors must be at least 1")
        if self.detector_chunk_size < 1:
            raise ValueError("detector_chunk_size must be at least 1")
        if self.jump_filter_pad_widths < 1:
            raise ValueError("jump_filter_pad_widths must be at least 1")
        if (
            self.saturation_lower is not None
            and self.saturation_upper is not None
            and self.saturation_lower >= self.saturation_upper
        ):
            raise ValueError("saturation_lower must be less than saturation_upper")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "DetectionConfig":
        known = {field.name for field in fields(cls)}
        unknown = set(values) - known
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown detection configuration key(s): {names}")
        return cls(**dict(values))

    @classmethod
    def from_toml(cls, path: str | Path) -> "DetectionConfig":
        """Load the ``[detection]`` section of a TOML configuration file."""

        if tomllib is None:
            raise RuntimeError("Reading TOML requires Python 3.11 or newer")
        with Path(path).open("rb") as stream:
            document = tomllib.load(stream)
        detection = document.get("detection", {})
        if not isinstance(detection, dict):
            raise ValueError("The TOML [detection] section must be a table")
        return cls.from_mapping(detection)

    def samples_for(
        self,
        duration_seconds: float,
        sample_interval: float,
        *,
        minimum: int = 1,
        odd: bool = False,
    ) -> int:
        """Convert a configured duration to a stable integer sample count."""

        count = max(minimum, int(round(duration_seconds / sample_interval)))
        if odd and count % 2 == 0:
            count += 1
        return count
