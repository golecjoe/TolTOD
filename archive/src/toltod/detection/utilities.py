"""Numerical helpers shared by issue detectors."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def true_runs(mask: np.ndarray) -> Iterator[tuple[int, int]]:
    """Yield half-open intervals for contiguous true values in a 1-D mask."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("true_runs expects a one-dimensional mask")
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    yield from zip(starts.tolist(), stops.tolist())


def retain_runs(
    mask: np.ndarray,
    *,
    minimum: int = 1,
    maximum: int | None = None,
) -> np.ndarray:
    """Retain only per-detector true runs within the requested lengths."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("retain_runs expects a sample x detector mask")
    retained = np.zeros_like(values)
    for detector in range(values.shape[1]):
        for start, stop in true_runs(values[:, detector]):
            length = stop - start
            if length >= minimum and (maximum is None or length <= maximum):
                retained[start:stop, detector] = True
    return retained


def robust_mad(values: np.ndarray) -> float:
    """Return a finite-safe, Gaussian-scaled median absolute deviation."""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    center = np.median(finite)
    return float(1.4826 * np.median(np.abs(finite - center)))


def fill_invalid_columns(values: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    """Linearly fill invalid values for calculating local baselines only."""

    filled = np.asarray(values, dtype=np.float64).copy()
    bad = np.asarray(invalid, dtype=bool)
    sample_positions = np.arange(filled.shape[0])
    for detector in range(filled.shape[1]):
        good = ~bad[:, detector] & np.isfinite(filled[:, detector])
        if not np.any(good):
            filled[:, detector] = 0.0
        elif np.count_nonzero(good) == 1:
            filled[:, detector] = filled[good, detector][0]
        elif not np.all(good):
            filled[:, detector] = np.interp(
                sample_positions,
                sample_positions[good],
                filled[good, detector],
            )
    return filled

