"""Validation and metadata extraction for TolTEC NetCDF files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset

from ..config import ReaderConfig
from ..errors import SchemaError
from ..models import NetworkInfo, ObservationInfo


@dataclass(frozen=True)
class ValidatedSchema:
    """Validated coordinate data needed by lazy network views."""

    info: ObservationInfo
    network_ids: np.ndarray
    detector_indices: dict[int, np.ndarray]


def validate_dataset(
    dataset: Dataset,
    path: Path,
    config: ReaderConfig,
) -> ValidatedSchema:
    """Validate required structure without reading the full signal variable."""

    _require_dimensions(dataset, config)
    signal = _require_variable(dataset, config.signal_variable)
    network = _require_variable(dataset, config.network_variable)
    time = _require_variable(dataset, config.time_variable)
    raw_sample_rate_variable = _require_variable(
        dataset, config.raw_detector_sample_rate_variable
    )

    expected_signal_dims = (config.sample_dimension, config.detector_dimension)
    if signal.dimensions != expected_signal_dims:
        raise SchemaError(
            f"Variable {config.signal_variable!r} has dimensions "
            f"{signal.dimensions!r}; expected {expected_signal_dims!r}"
        )
    if not np.issubdtype(signal.dtype, np.number):
        raise SchemaError(f"Variable {config.signal_variable!r} must be numeric")

    expected_network_dims = (config.detector_dimension,)
    if network.dimensions != expected_network_dims:
        raise SchemaError(
            f"Variable {config.network_variable!r} has dimensions "
            f"{network.dimensions!r}; expected {expected_network_dims!r}"
        )

    expected_time_dims = (config.sample_dimension,)
    if time.dimensions != expected_time_dims:
        raise SchemaError(
            f"Variable {config.time_variable!r} has dimensions "
            f"{time.dimensions!r}; expected {expected_time_dims!r}"
        )

    network_ids = _read_integer_coordinate(network, config.network_variable)
    if network_ids.size != signal.shape[1]:
        raise SchemaError(
            f"Variable {config.network_variable!r} has {network_ids.size} entries, "
            f"but {config.signal_variable!r} has {signal.shape[1]} detectors"
        )

    telescope_sample_interval = _inspect_time(
        time,
        config.time_variable,
        require_strict_order=config.validate_time_order,
    )
    raw_detector_sample_rate = _read_required_positive_scalar(
        raw_sample_rate_variable,
        config.raw_detector_sample_rate_variable,
    )
    detector_sample_rate = (
        raw_detector_sample_rate / config.downsample_factor
    )
    detector_sample_interval = 1.0 / detector_sample_rate

    detector_indices = {
        int(network_id): _readonly(np.flatnonzero(network_ids == network_id))
        for network_id in np.unique(network_ids)
    }
    array_ids = _read_optional_integer_coordinate(
        dataset, "apt_array", config.detector_dimension
    )
    networks = tuple(
        _network_info(network_id, indices, array_ids)
        for network_id, indices in sorted(detector_indices.items())
    )

    info = ObservationInfo(
        path=path,
        sample_count=signal.shape[0],
        detector_count=signal.shape[1],
        signal_variable=config.signal_variable,
        signal_units=_optional_text_attribute(signal, "units"),
        time_variable=config.time_variable,
        time_units=_optional_text_attribute(time, "units"),
        telescope_sample_interval=telescope_sample_interval,
        raw_detector_sample_rate=raw_detector_sample_rate,
        downsample_factor=config.downsample_factor,
        detector_sample_interval=detector_sample_interval,
        detector_sample_rate=detector_sample_rate,
        network_variable=config.network_variable,
        networks=networks,
        observation_number=_optional_integer_scalar(dataset, "Header.Dcs.ObsNum"),
        subobservation_number=_optional_integer_scalar(
            dataset, "Header.Dcs.SubObsNum"
        ),
        scan_number=_optional_integer_scalar(dataset, "Header.Dcs.ScanNum"),
        date_obs=_optional_text_scalar(dataset, "DATEOBS0"),
    )
    return ValidatedSchema(
        info=info,
        network_ids=_readonly(network_ids),
        detector_indices=detector_indices,
    )


def _require_dimensions(dataset: Dataset, config: ReaderConfig) -> None:
    for name in (config.sample_dimension, config.detector_dimension):
        if name not in dataset.dimensions:
            raise SchemaError(f"Required dimension {name!r} is missing")
        if len(dataset.dimensions[name]) == 0:
            raise SchemaError(f"Required dimension {name!r} is empty")


def _require_variable(dataset: Dataset, name: str) -> Any:
    try:
        return dataset.variables[name]
    except KeyError as exc:
        raise SchemaError(f"Required variable {name!r} is missing") from exc


def _read_integer_coordinate(variable: Any, name: str) -> np.ndarray:
    values = np.ma.asarray(variable[:])
    if np.ma.getmaskarray(values).any():
        raise SchemaError(f"Variable {name!r} contains masked values")
    raw = np.asarray(values, dtype=np.float64)
    if not np.isfinite(raw).all():
        raise SchemaError(f"Variable {name!r} contains non-finite values")
    rounded = np.rint(raw)
    if not np.array_equal(raw, rounded):
        raise SchemaError(f"Variable {name!r} contains non-integer values")
    return rounded.astype(np.int64)


def _read_optional_integer_coordinate(
    dataset: Dataset,
    name: str,
    detector_dimension: str,
) -> np.ndarray | None:
    if name not in dataset.variables:
        return None
    variable = dataset.variables[name]
    if variable.dimensions != (detector_dimension,):
        return None
    try:
        return _read_integer_coordinate(variable, name)
    except SchemaError:
        return None


def _inspect_time(
    variable: Any,
    name: str,
    *,
    require_strict_order: bool,
) -> float | None:
    values = np.ma.asarray(variable[:])
    if np.ma.getmaskarray(values).any():
        raise SchemaError(f"Time variable {name!r} contains masked samples")
    raw = np.asarray(values, dtype=np.float64)
    if not np.isfinite(raw).all():
        raise SchemaError(f"Time variable {name!r} contains non-finite samples")
    if raw.size < 2:
        return None
    differences = np.diff(raw)
    if require_strict_order and not np.all(differences > 0):
        raise SchemaError(f"Time variable {name!r} is not strictly increasing")
    positive = differences[differences > 0]
    if positive.size == 0:
        return None
    return float(np.median(positive))


def _network_info(
    network_id: int,
    indices: np.ndarray,
    array_ids: np.ndarray | None,
) -> NetworkInfo:
    contiguous = indices.size < 2 or bool(np.all(np.diff(indices) == 1))
    arrays: tuple[int, ...] = ()
    if array_ids is not None:
        arrays = tuple(int(value) for value in np.unique(array_ids[indices]))
    return NetworkInfo(
        network_id=int(network_id),
        detector_count=int(indices.size),
        detector_start=int(indices[0]),
        detector_stop=int(indices[-1] + 1),
        detector_indices_contiguous=contiguous,
        array_ids=arrays,
    )


def _optional_integer_scalar(dataset: Dataset, name: str) -> int | None:
    if name not in dataset.variables:
        return None
    values = np.ma.asarray(dataset.variables[name][:]).reshape(-1)
    if values.size != 1 or np.ma.getmaskarray(values).any():
        return None
    value = float(values[0])
    if not np.isfinite(value) or value != round(value):
        return None
    return int(value)


def _read_required_positive_scalar(variable: Any, name: str) -> float:
    values = np.ma.asarray(variable[:]).reshape(-1)
    if values.size != 1 or np.ma.getmaskarray(values).any():
        raise SchemaError(f"Variable {name!r} must contain one unmasked value")
    value = float(values[0])
    if not np.isfinite(value) or value <= 0:
        raise SchemaError(f"Variable {name!r} must contain a positive value")
    return value


def _optional_text_scalar(dataset: Dataset, name: str) -> str | None:
    if name not in dataset.variables:
        return None
    values = np.asarray(dataset.variables[name][:]).reshape(-1)
    if values.size != 1:
        return None
    value = values[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _optional_text_attribute(variable: Any, name: str) -> str | None:
    if name not in variable.ncattrs():
        return None
    value = variable.getncattr(name)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _readonly(values: np.ndarray) -> np.ndarray:
    values.setflags(write=False)
    return values
