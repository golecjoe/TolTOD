from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from toltod import detect_array_issues, open_tod


@pytest.mark.integration
def test_local_example_files_can_be_inspected_and_sliced() -> None:
    paths = sorted(Path("data").glob("*.nc"))
    if not paths:
        pytest.skip("No local example NetCDF files are available")

    for path in paths:
        with open_tod(path) as tod:
            assert tod.info.sample_count > 0
            assert tod.info.detector_count > 0
            assert sum(item.detector_count for item in tod.info.networks) == (
                tod.info.detector_count
            )
            assert tod.network_ids == tuple(sorted(tod.network_ids))

            network = tod.network(tod.network_ids[0])
            signal = network.read_signal(
                samples=slice(0, 4),
                detectors=slice(0, 3),
            )
            time = network.read_telescope_time(samples=slice(0, 4))
            detector_time = network.read_detector_time(samples=slice(0, 4))

            assert signal.shape == (4, 3)
            assert time.shape == (4,)
            assert detector_time.shape == (4,)
            assert np.isfinite(signal).all()
            assert np.all(np.diff(time) > 0)
            assert np.all(np.diff(detector_time) > 0)
            assert tod.info.detector_sample_rate == pytest.approx(
                tod.info.raw_detector_sample_rate / tod.info.downsample_factor
            )


@pytest.mark.integration
def test_detection_runs_on_small_slices_from_local_examples() -> None:
    paths = sorted(Path("data").glob("*.nc"))
    if not paths:
        pytest.skip("No local example NetCDF files are available")

    for path in paths:
        with open_tod(path) as tod:
            network = tod.network(tod.network_ids[0])
            samples = slice(0, 512)
            detectors = slice(0, 64)
            signal = network.read_signal(samples=samples, detectors=detectors)
            time = network.read_detector_time(samples=samples)
            source_flags = network.read_sample_data(
                "flags", samples=samples, detectors=detectors
            )
            catalog = detect_array_issues(
                signal,
                time,
                network_id=network.network_id,
                detector_indices=network.detector_indices[detectors],
                source_flags=source_flags,
                source_path=str(path),
            )

        assert catalog.sample_flags.shape == (512, 64)
        assert len(catalog.detector_metrics) == 64
        assert catalog.source_path == str(path)
