from __future__ import annotations

import pytest

from toltod import (
    DetectionConfig,
    QCReportConfig,
    ReaderConfig,
    RemediationConfig,
    RemediationQCConfig,
)


def test_reader_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="Unknown reader configuration"):
        ReaderConfig.from_mapping({"not_a_setting": True})


def test_reader_config_loads_toml(tmp_path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text(
        '[reader]\ntime_variable = "PpsTime"\nvalidate_time_order = false\n',
        encoding="utf-8",
    )

    config = ReaderConfig.from_toml(path)

    assert config.time_variable == "PpsTime"
    assert config.validate_time_order is False


def test_detection_config_loads_toml(tmp_path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text(
        '[detection]\nspike_sigma = 12.0\nsource_flag_variable = "flags"\n',
        encoding="utf-8",
    )

    config = DetectionConfig.from_toml(path)

    assert config.spike_sigma == 12.0
    assert config.source_flag_variable == "flags"


def test_detection_config_requires_wider_negative_gaussian() -> None:
    with pytest.raises(ValueError, match="must exceed"):
        DetectionConfig(
            spike_narrow_sigma_seconds=0.10,
            spike_wide_sigma_seconds=0.05,
        )


def test_detection_config_validates_jump_filter_padding() -> None:
    with pytest.raises(ValueError, match="jump_filter_pad_widths"):
        DetectionConfig(jump_filter_pad_widths=0)


def test_detection_config_validates_fractions() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        DetectionConfig(correlated_detector_fraction=1.1)


def test_qc_config_loads_toml(tmp_path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text(
        "[qc]\ndrastic_sigma = 4.0\nfigure_dpi = 120\n",
        encoding="utf-8",
    )

    config = QCReportConfig.from_toml(path)

    assert config.drastic_sigma == 4.0
    assert config.figure_dpi == 120


def test_remediation_config_loads_toml(tmp_path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text(
        "[remediation]\n"
        "spike_padding_seconds = 0.12\n"
        "detector_bad_fraction = 0.20\n",
        encoding="utf-8",
    )

    config = RemediationConfig.from_toml(path)

    assert config.spike_padding_seconds == 0.12
    assert config.detector_bad_fraction == 0.20


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"spike_padding_seconds": -0.1}, "nonnegative"),
        ({"noise_context_seconds": 0.0}, "positive"),
        ({"jump_level_context_seconds": 0.0}, "positive"),
        ({"minimum_noise_samples": 1}, "at least 2"),
        ({"minimum_jump_level_samples": 1}, "at least 2"),
        ({"detector_bad_fraction": 1.1}, "between 0 and 1"),
        ({"noise_seed": -1}, "nonnegative"),
    ],
)
def test_remediation_config_validates_policy_settings(
    settings: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RemediationConfig(**settings)


def test_remediation_qc_config_loads_toml(tmp_path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text(
        "[remediation_qc]\n"
        "context_seconds = 8.0\n"
        "max_spike_examples = 3\n",
        encoding="utf-8",
    )

    config = RemediationQCConfig.from_toml(path)

    assert config.context_seconds == 8.0
    assert config.max_spike_examples == 3


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"context_seconds": 0.0}, "positive"),
        ({"max_jump_examples": -1}, "nonnegative"),
        ({"figure_dpi": 71}, "at least 72"),
    ],
)
def test_remediation_qc_config_validates_settings(
    settings: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RemediationQCConfig(**settings)
