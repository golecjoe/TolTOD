"""Configuration for converting detected issues into remediation plans."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    tomllib = None  # type: ignore[assignment]


@dataclass(frozen=True)
class RemediationConfig:
    """Policy settings used to plan repairs and detector rejection.

    Duration settings are expressed in seconds. ``noise_context_seconds`` is
    the amount of context requested on each side of a repair span.
    """

    spike_padding_seconds: float = 0.08
    jump_padding_seconds: float = 0.08
    other_padding_seconds: float = 0.0
    noise_context_seconds: float = 2.0
    minimum_noise_samples: int = 20
    jump_level_context_seconds: float = 0.40
    minimum_jump_level_samples: int = 5
    detector_bad_fraction: float = 0.10
    noise_seed: int = 0

    def __post_init__(self) -> None:
        nonnegative = {
            "spike_padding_seconds": self.spike_padding_seconds,
            "jump_padding_seconds": self.jump_padding_seconds,
            "other_padding_seconds": self.other_padding_seconds,
        }
        for name, value in nonnegative.items():
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.noise_context_seconds <= 0:
            raise ValueError("noise_context_seconds must be positive")
        if self.jump_level_context_seconds <= 0:
            raise ValueError("jump_level_context_seconds must be positive")
        if self.minimum_noise_samples < 2:
            raise ValueError("minimum_noise_samples must be at least 2")
        if self.minimum_jump_level_samples < 2:
            raise ValueError("minimum_jump_level_samples must be at least 2")
        if not 0.0 <= self.detector_bad_fraction <= 1.0:
            raise ValueError("detector_bad_fraction must be between 0 and 1")
        if self.noise_seed < 0:
            raise ValueError("noise_seed must be nonnegative")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RemediationConfig":
        known = {field.name for field in fields(cls)}
        unknown = set(values) - known
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(
                f"Unknown remediation configuration key(s): {names}"
            )
        return cls(**dict(values))

    @classmethod
    def from_toml(cls, path: str | Path) -> "RemediationConfig":
        """Load the ``[remediation]`` section of a TOML file."""

        if tomllib is None:
            raise RuntimeError("Reading TOML requires Python 3.11 or newer")
        with Path(path).open("rb") as stream:
            document = tomllib.load(stream)
        remediation = document.get("remediation", {})
        if not isinstance(remediation, dict):
            raise ValueError("The TOML [remediation] section must be a table")
        return cls.from_mapping(remediation)

    def samples_for(
        self,
        duration_seconds: float,
        sample_interval: float,
        *,
        minimum: int = 0,
    ) -> int:
        """Convert a duration to a stable nonnegative sample count."""

        if sample_interval <= 0:
            raise ValueError("sample_interval must be positive")
        return max(minimum, int(round(duration_seconds / sample_interval)))
