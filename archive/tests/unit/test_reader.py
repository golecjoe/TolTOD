from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

from toltod import ReaderClosedError, ReaderConfig, SchemaError, open_tod


@pytest.fixture()
def tod_path(tmp_path: Path) -> Path:
    path = tmp_path / "small.nc"
    with Dataset(path, "w") as dataset:
        dataset.createDimension("n_pts", 6)
        dataset.createDimension("n_dets", 4)
        dataset.createDimension("scalar", 1)

        signal = dataset.createVariable("signal", "f8", ("n_pts", "n_dets"))
        signal.units = "mJy/beam"
        signal[:] = np.arange(24, dtype=np.float64).reshape(6, 4)

        time = dataset.createVariable("TelTime", "f8", ("n_pts",))
        time.units = "s"
        time[:] = np.arange(6, dtype=np.float64) * 0.04

        sample_rate = dataset.createVariable("SAMPRATE", "f8", ("scalar",))
        sample_rate[:] = [100.0]

        network = dataset.createVariable("apt_nw", "f8", ("n_dets",))
        network[:] = [2, 7, 2, 7]

        flags = dataset.createVariable("flags", "f8", ("n_pts", "n_dets"))
        flags[:] = 0
        flags[2, 1] = 1

        array = dataset.createVariable("apt_array", "f8", ("n_dets",))
        array[:] = [0, 1, 0, 1]

        uid = dataset.createVariable("apt_uid", "f8", ("n_dets",))
        uid[:] = [10, 11, 12, 13]

        obsnum = dataset.createVariable("Header.Dcs.ObsNum", "f8", ("scalar",))
        obsnum[:] = [152390]
    return path


def test_reader_groups_signal_columns_by_raw_apt_nw(tod_path: Path) -> None:
    with open_tod(tod_path) as tod:
        assert tod.network_ids == (2, 7)
        assert tod.info.sample_count == 6
        assert tod.info.detector_count == 4
        assert tod.info.telescope_sample_interval == pytest.approx(0.04)
        assert tod.info.raw_detector_sample_rate == pytest.approx(100.0)
        assert tod.info.downsample_factor == 5
        assert tod.info.detector_sample_interval == pytest.approx(0.05)
        assert tod.info.detector_sample_rate == pytest.approx(20.0)
        assert [item.detector_count for item in tod.info.networks] == [2, 2]
        assert not tod.info.networks[0].detector_indices_contiguous

        network = tod.network(2)
        values = network.read_signal(samples=slice(1, 4))

        expected = np.arange(24, dtype=np.float64).reshape(6, 4)[1:4, [0, 2]]
        np.testing.assert_array_equal(values, expected)
        np.testing.assert_array_equal(network.detector_indices, [0, 2])


def test_reader_reads_network_sample_data(tod_path: Path) -> None:
    with open_tod(tod_path) as tod:
        assert tod.network(7).has_sample_data("flags")
        assert not tod.network(7).has_sample_data("apt_flag")
        flags = tod.network(7).read_sample_data(
            "flags", samples=slice(1, 4), detectors=[0]
        )

    np.testing.assert_array_equal(flags, [[0], [1], [0]])


def test_detector_time_is_generated_independently_of_telescope_time(
    tod_path: Path,
) -> None:
    with open_tod(tod_path) as tod:
        network = tod.network(2)
        detector_time = network.read_detector_time(samples=slice(1, 4))
        telescope_time = network.read_telescope_time(samples=slice(1, 4))

    np.testing.assert_allclose(detector_time, [0.05, 0.10, 0.15])
    np.testing.assert_allclose(telescope_time, [0.04, 0.08, 0.12])


def test_downsample_factor_changes_effective_detector_rate(tod_path: Path) -> None:
    config = ReaderConfig(downsample_factor=4)

    with open_tod(tod_path, config=config) as tod:
        assert tod.info.detector_sample_rate == pytest.approx(25.0)
        np.testing.assert_allclose(
            tod.network(2).read_detector_time(samples=slice(0, 3)),
            [0.0, 0.04, 0.08],
        )


def test_reader_reads_network_detector_metadata(tod_path: Path) -> None:
    with open_tod(tod_path) as tod:
        metadata = tod.network(7).read_detector_metadata(["apt_uid", "apt_nw"])

    np.testing.assert_array_equal(metadata["apt_uid"], [11, 13])
    np.testing.assert_array_equal(metadata["apt_nw"], [7, 7])


def test_reader_discovers_returning_networks_six_and_ten(tod_path: Path) -> None:
    with Dataset(tod_path, "a") as dataset:
        dataset.variables["apt_nw"][:] = [6, 10, 6, 10]

    with open_tod(tod_path) as tod:
        assert tod.network_ids == (6, 10)
        np.testing.assert_array_equal(tod.network(6).detector_indices, [0, 2])
        np.testing.assert_array_equal(tod.network(10).detector_indices, [1, 3])


def test_network_view_fails_after_file_is_closed(tod_path: Path) -> None:
    tod = open_tod(tod_path)
    network = tod.network(2)
    tod.close()

    with pytest.raises(ReaderClosedError):
        network.read_time()


def test_schema_rejects_non_integer_network_ids(tod_path: Path) -> None:
    with Dataset(tod_path, "a") as dataset:
        dataset.variables["apt_nw"][0] = 0.5

    with pytest.raises(SchemaError, match="non-integer"):
        open_tod(tod_path)


def test_time_order_validation_can_be_disabled(tod_path: Path) -> None:
    with Dataset(tod_path, "a") as dataset:
        dataset.variables["TelTime"][:] = [0, 1, 1, 2, 3, 4]

    with pytest.raises(SchemaError, match="not strictly increasing"):
        open_tod(tod_path)

    config = ReaderConfig(validate_time_order=False)
    with open_tod(tod_path, config=config) as tod:
        assert tod.info.sample_count == 6
