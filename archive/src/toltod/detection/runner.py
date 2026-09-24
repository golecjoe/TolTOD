"""Network-level orchestration for read-only issue detection."""

from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from typing import Sequence

import numpy as np

from ..errors import TolTODReadError
from ..io.reader import NetworkTOD
from .basic import detect_basic_issues
from .config import DetectionConfig
from .correlated import detect_correlated_issues
from .jumps import detect_jumps
from .models import IssueCatalog
from .quality import calculate_detector_metrics
from .spikes import detect_spikes


def detect_network_issues(
    network: NetworkTOD,
    *,
    config: DetectionConfig | None = None,
) -> IssueCatalog:
    """Load and analyze one complete network without changing source data."""

    settings = config or DetectionConfig()
    signal = network.read_signal()
    time = network.read_detector_time()
    source_flags: np.ndarray | None = None
    if settings.source_flag_variable is not None:
        if network.has_sample_data(settings.source_flag_variable):
            raw_flags = network.read_sample_data(settings.source_flag_variable)
            source_flags = _source_bad_mask(raw_flags, settings)
            del raw_flags
        elif settings.require_source_flags:
            raise TolTODReadError(
                f"Required source-flag variable "
                f"{settings.source_flag_variable!r} is missing or has "
                "incompatible dimensions"
            )
    return detect_array_issues(
        signal,
        time,
        network_id=network.network_id,
        detector_indices=network.detector_indices,
        source_flags=source_flags,
        source_flags_are_bad_mask=True,
        config=settings,
        source_path=str(network.source_path),
    )


def detect_array_issues(
    signal: np.ndarray,
    time: np.ndarray,
    *,
    network_id: int = 0,
    detector_indices: Sequence[int] | np.ndarray | None = None,
    source_flags: np.ndarray | None = None,
    source_flags_are_bad_mask: bool = False,
    config: DetectionConfig | None = None,
    source_path: str | None = None,
) -> IssueCatalog:
    """Analyze an in-memory sample-by-detector array.

    This entry point supports synthetic tests and algorithm development without
    requiring a NetCDF file.
    """

    started = perf_counter()
    settings = config or DetectionConfig()
    values = _as_float_array(signal)
    timestamps = _as_float_array(time)
    if values.ndim != 2:
        raise ValueError("signal must be a sample x detector array")
    if timestamps.ndim != 1 or timestamps.size != values.shape[0]:
        raise ValueError("time must be one-dimensional with one value per sample")
    if not np.isfinite(timestamps).all() or timestamps.size < 2:
        raise ValueError("time must contain at least two finite samples")
    differences = np.diff(timestamps)
    if not np.all(differences > 0):
        raise ValueError("time must be strictly increasing for issue detection")
    sample_interval = float(np.median(differences))
    if sample_interval <= 0:
        raise ValueError("time must have a positive sample interval")

    if detector_indices is None:
        indices = np.arange(values.shape[1], dtype=np.int64)
    else:
        indices = np.asarray(detector_indices, dtype=np.int64)
    if indices.shape != (values.shape[1],):
        raise ValueError("detector_indices must contain one index per signal column")

    flags: np.ndarray | None = None
    if source_flags is not None:
        flags = (
            np.asarray(source_flags, dtype=bool)
            if source_flags_are_bad_mask
            else _source_bad_mask(source_flags, settings)
        )
        if flags.shape != values.shape:
            raise ValueError("source_flags must have the same shape as signal")

    catalog = IssueCatalog.empty(
        network_id=int(network_id),
        sample_count=values.shape[0],
        detector_indices=indices,
        sample_interval=sample_interval,
        source_path=source_path,
        detection_settings=asdict(settings),
    )
    detect_basic_issues(values, catalog, settings, source_bad_mask=flags)
    detect_jumps(values, catalog, settings)
    detect_spikes(values, catalog, settings)
    detect_correlated_issues(catalog, settings)
    calculate_detector_metrics(values, catalog, settings)
    catalog.elapsed_seconds = perf_counter() - started
    return catalog


def _source_bad_mask(
    source_flags: np.ndarray,
    config: DetectionConfig,
) -> np.ndarray:
    values = np.ma.asarray(source_flags)
    masked = np.ma.getmaskarray(values)
    raw = np.asarray(values.data)
    nonfinite = ~np.isfinite(raw)
    if config.source_flag_nonzero_is_bad:
        return masked | nonfinite | (raw != 0)
    return masked | nonfinite | (raw == 0)


def _as_float_array(values: np.ndarray) -> np.ndarray:
    array = np.ma.asarray(values)
    if np.ma.getmaskarray(array).any():
        return np.asarray(np.ma.filled(array, np.nan), dtype=np.float64)
    return np.asarray(array.data, dtype=np.float64)
