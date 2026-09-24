"""Lazy, network-oriented access to TolTEC NetCDF time streams."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Iterable, Sequence

import numpy as np
from netCDF4 import Dataset

from ..config import ReaderConfig
from ..errors import ReaderClosedError, TolTODReadError, UnknownNetworkError
from ..models import ObservationInfo
from .schema import validate_dataset


class TolTECFile:
    """An open, validated TolTEC TOD file.

    Construction reads the small time and detector-coordinate variables needed
    for validation. The two-dimensional signal array remains on disk until a
    :class:`NetworkTOD` view explicitly requests a slice.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        config: ReaderConfig | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.config = config or ReaderConfig()
        self._dataset: Dataset | None = None
        try:
            self._dataset = Dataset(self.path, mode="r")
            schema = validate_dataset(self._dataset, self.path, self.config)
        except Exception:
            self.close()
            raise
        self.info = schema.info
        self._network_ids = schema.network_ids
        self._detector_indices = schema.detector_indices

    @property
    def closed(self) -> bool:
        """Whether the underlying NetCDF handle has been closed."""

        return self._dataset is None

    @property
    def network_ids(self) -> tuple[int, ...]:
        """Raw ``apt_nw`` values present in this observation."""

        return self.info.network_ids

    @property
    def available_detector_metadata(self) -> tuple[str, ...]:
        """Detector-length ``apt_*`` variables available in the file."""

        dataset = self._require_open()
        detector_dimension = self.config.detector_dimension
        return tuple(
            sorted(
                name
                for name, variable in dataset.variables.items()
                if name.startswith("apt_")
                and variable.dimensions == (detector_dimension,)
            )
        )

    def network(self, network_id: int) -> "NetworkTOD":
        """Return a lazy view of detectors with the requested ``apt_nw`` value."""

        self._require_open()
        try:
            indices = self._detector_indices[int(network_id)]
        except KeyError as exc:
            available = ", ".join(str(value) for value in self.network_ids)
            raise UnknownNetworkError(
                f"Network {network_id!r} is absent; available IDs: {available}"
            ) from exc
        return NetworkTOD(self, int(network_id), indices)

    def networks(self) -> Iterable["NetworkTOD"]:
        """Iterate through lazy network views in raw ID order."""

        for network_id in self.network_ids:
            yield self.network(network_id)

    def close(self) -> None:
        """Close the underlying NetCDF file if it is open."""

        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None

    def _require_open(self) -> Dataset:
        if self._dataset is None:
            raise ReaderClosedError(f"TolTEC file is closed: {self.path}")
        return self._dataset

    def __enter__(self) -> "TolTECFile":
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class NetworkTOD:
    """A lazy view of signal columns belonging to one ``apt_nw`` value."""

    def __init__(
        self,
        source: TolTECFile,
        network_id: int,
        detector_indices: np.ndarray,
    ) -> None:
        self._source = source
        self.network_id = network_id
        self._detector_indices = detector_indices

    @property
    def sample_count(self) -> int:
        return self._source.info.sample_count

    @property
    def source_path(self) -> Path:
        return self._source.path

    @property
    def detector_count(self) -> int:
        return int(self._detector_indices.size)

    @property
    def detector_indices(self) -> np.ndarray:
        """Read-only detector column indices in the source file."""

        return self._detector_indices

    @property
    def signal_units(self) -> str | None:
        return self._source.info.signal_units

    def read_signal(
        self,
        *,
        samples: slice | None = None,
        detectors: slice | Sequence[int] | None = None,
    ) -> np.ndarray:
        """Read a ``sample x detector`` signal slice for this network.

        ``detectors`` uses positions local to this network, not global source
        columns. Supplying a sample slice is recommended for exploratory reads.
        """

        dataset = self._source._require_open()
        sample_selector = samples if samples is not None else slice(None)
        if not isinstance(sample_selector, slice):
            raise TypeError("samples must be a slice or None")
        global_indices = self._select_detector_indices(detectors)
        file_selector = _compact_selector(global_indices)
        values = dataset.variables[self._source.config.signal_variable][
            sample_selector, file_selector
        ]
        return np.asanyarray(values)

    def read_time(self, *, samples: slice | None = None) -> np.ndarray:
        """Read telescope timestamps from the configured source variable.

        This is not the detector sample-time coordinate. Detection code should
        use :meth:`read_detector_time` instead.
        """

        dataset = self._source._require_open()
        sample_selector = samples if samples is not None else slice(None)
        if not isinstance(sample_selector, slice):
            raise TypeError("samples must be a slice or None")
        values = dataset.variables[self._source.config.time_variable][sample_selector]
        return np.asanyarray(values)

    def read_telescope_time(self, *, samples: slice | None = None) -> np.ndarray:
        """Explicit alias for :meth:`read_time`."""

        return self.read_time(samples=samples)

    def read_detector_time(self, *, samples: slice | None = None) -> np.ndarray:
        """Generate relative detector timestamps in seconds.

        The cadence is ``SAMPRATE / downsample_factor`` (using configured
        names), independent of the telescope timestamp coordinate.
        """

        self._source._require_open()
        sample_selector = samples if samples is not None else slice(None)
        if not isinstance(sample_selector, slice):
            raise TypeError("samples must be a slice or None")
        indices = np.arange(self.sample_count, dtype=np.float64)[sample_selector]
        return indices * self._source.info.detector_sample_interval

    def read_sample_data(
        self,
        name: str,
        *,
        samples: slice | None = None,
        detectors: slice | Sequence[int] | None = None,
    ) -> np.ndarray:
        """Read another ``sample x detector`` variable for this network.

        This is primarily intended for source data such as ``flags``. The
        requested variable must use the same dimensions and orientation as the
        configured signal array.
        """

        dataset = self._source._require_open()
        if name not in dataset.variables:
            raise TolTODReadError(f"Sample data variable {name!r} is missing")
        variable = dataset.variables[name]
        expected = (
            self._source.config.sample_dimension,
            self._source.config.detector_dimension,
        )
        if variable.dimensions != expected:
            raise TolTODReadError(
                f"Variable {name!r} has dimensions {variable.dimensions!r}; "
                f"expected {expected!r}"
            )

        sample_selector = samples if samples is not None else slice(None)
        if not isinstance(sample_selector, slice):
            raise TypeError("samples must be a slice or None")
        global_indices = self._select_detector_indices(detectors)
        file_selector = _compact_selector(global_indices)
        return np.asanyarray(variable[sample_selector, file_selector])

    def has_sample_data(self, name: str) -> bool:
        """Whether a variable with the signal's sample/detector shape exists."""

        dataset = self._source._require_open()
        if name not in dataset.variables:
            return False
        expected = (
            self._source.config.sample_dimension,
            self._source.config.detector_dimension,
        )
        return dataset.variables[name].dimensions == expected

    def read_detector_metadata(
        self,
        names: Sequence[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Read selected detector-length metadata for this network.

        With no names supplied, all one-dimensional ``apt_*`` variables are
        returned. Metadata keys are validated before any values are read.
        """

        dataset = self._source._require_open()
        selected_names = (
            tuple(names)
            if names is not None
            else self._source.available_detector_metadata
        )
        detector_dimension = self._source.config.detector_dimension
        for name in selected_names:
            if name not in dataset.variables:
                raise TolTODReadError(f"Detector metadata variable {name!r} is missing")
            if dataset.variables[name].dimensions != (detector_dimension,):
                raise TolTODReadError(
                    f"Variable {name!r} is not indexed solely by "
                    f"{detector_dimension!r}"
                )

        selector = _compact_selector(self._detector_indices)
        return {
            name: np.asanyarray(dataset.variables[name][selector])
            for name in selected_names
        }

    def _select_detector_indices(
        self,
        selection: slice | Sequence[int] | None,
    ) -> np.ndarray:
        if selection is None:
            return self._detector_indices
        if isinstance(selection, slice):
            return self._detector_indices[selection]
        local = np.asarray(selection, dtype=np.int64)
        if local.ndim != 1:
            raise IndexError("detectors must be a one-dimensional sequence")
        if np.any(local < 0) or np.any(local >= self.detector_count):
            raise IndexError("a local detector position is out of bounds")
        return self._detector_indices[local]


def open_tod(
    path: str | Path,
    *,
    config: ReaderConfig | None = None,
) -> TolTECFile:
    """Open and validate a TolTEC TOD file."""

    return TolTECFile(path, config=config)


def _compact_selector(indices: np.ndarray) -> slice | np.ndarray:
    """Use an efficient slice when selected source columns are contiguous."""

    if indices.size == 0:
        return indices
    if indices.size == 1 or np.all(np.diff(indices) == 1):
        return slice(int(indices[0]), int(indices[-1]) + 1)
    return indices
