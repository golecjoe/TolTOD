from __future__ import annotations

from toltod.cli import build_parser


def test_remediation_report_command_parses_network_and_output() -> None:
    args = build_parser().parse_args(
        [
            "remediation-report",
            "data/example.nc",
            "--network",
            "7",
            "--config",
            "configs/default.toml",
            "--output-dir",
            "outputs/checks",
            "--json",
        ]
    )

    assert args.command == "remediation-report"
    assert args.network == 7
    assert str(args.output_dir) == "outputs/checks"
    assert args.json is True
