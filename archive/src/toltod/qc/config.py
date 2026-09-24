"""Configuration for network-level QC visualization products."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    tomllib = None  # type: ignore[assignment]


@dataclass(frozen=True)
class QCReportConfig:
    """Presentation and outlier settings for one network report."""

    drastic_sigma: float = 3.0
    histogram_max_bins: int = 40
    figure_dpi: int = 160
    exclude_persistent_source_flags: bool = True

    def __post_init__(self) -> None:
        if self.drastic_sigma <= 0:
            raise ValueError("drastic_sigma must be positive")
        if self.histogram_max_bins < 5:
            raise ValueError("histogram_max_bins must be at least 5")
        if self.figure_dpi < 72:
            raise ValueError("figure_dpi must be at least 72")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "QCReportConfig":
        known = {field.name for field in fields(cls)}
        unknown = set(values) - known
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown QC configuration key(s): {names}")
        return cls(**dict(values))

    @classmethod
    def from_toml(cls, path: str | Path) -> "QCReportConfig":
        """Load the ``[qc]`` section of a TOML configuration file."""

        if tomllib is None:
            raise RuntimeError("Reading TOML requires Python 3.11 or newer")
        with Path(path).open("rb") as stream:
            document = tomllib.load(stream)
        qc = document.get("qc", {})
        if not isinstance(qc, dict):
            raise ValueError("The TOML [qc] section must be a table")
        return cls.from_mapping(qc)


@dataclass(frozen=True)
class RemediationQCConfig:
    """Settings for bounded raw-versus-cleaned repair snippets."""

    context_seconds: float = 10.0
    max_spike_examples: int = 6
    max_jump_examples: int = 6
    max_other_examples: int = 4
    figure_dpi: int = 120

    def __post_init__(self) -> None:
        if self.context_seconds <= 0:
            raise ValueError("context_seconds must be positive")
        limits = {
            "max_spike_examples": self.max_spike_examples,
            "max_jump_examples": self.max_jump_examples,
            "max_other_examples": self.max_other_examples,
        }
        for name, value in limits.items():
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.figure_dpi < 72:
            raise ValueError("figure_dpi must be at least 72")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RemediationQCConfig":
        known = {field.name for field in fields(cls)}
        unknown = set(values) - known
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(
                f"Unknown remediation QC configuration key(s): {names}"
            )
        return cls(**dict(values))

    @classmethod
    def from_toml(cls, path: str | Path) -> "RemediationQCConfig":
        """Load the ``[remediation_qc]`` section of a TOML file."""

        if tomllib is None:
            raise RuntimeError("Reading TOML requires Python 3.11 or newer")
        with Path(path).open("rb") as stream:
            document = tomllib.load(stream)
        remediation_qc = document.get("remediation_qc", {})
        if not isinstance(remediation_qc, dict):
            raise ValueError(
                "The TOML [remediation_qc] section must be a table"
            )
        return cls.from_mapping(remediation_qc)
