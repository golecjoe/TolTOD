"""Command-line interface for TolTOD ingestion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import ReaderConfig
from .detection import (
    DetectionConfig,
    IssueCatalog,
    detect_array_issues,
    detect_network_issues,
)
from .errors import TolTODReadError
from .io.reader import NetworkTOD, open_tod
from .qc import (
    QCReportConfig,
    RemediationQCConfig,
    generate_network_qc_report,
    generate_remediation_qc_report,
)
from .remediation import RemediationConfig, remediate_array


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="toltod")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="validate a TOD file and summarize its networks",
    )
    inspect_parser.add_argument("path", type=Path)
    inspect_parser.add_argument(
        "--config",
        type=Path,
        help="TOML file containing a [reader] table",
    )
    inspect_parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON",
    )

    detect_parser = subparsers.add_parser(
        "detect",
        help="run read-only issue detection for one network",
    )
    detect_parser.add_argument("path", type=Path)
    detect_parser.add_argument("--network", type=int, required=True)
    detect_parser.add_argument(
        "--config",
        type=Path,
        help="TOML file containing [reader] and [detection] tables",
    )
    detect_parser.add_argument(
        "--json",
        action="store_true",
        help="emit a machine-readable summary",
    )

    report_parser = subparsers.add_parser(
        "report",
        help="detect issues and generate network QC visualizations",
    )
    report_parser.add_argument("path", type=Path)
    report_parser.add_argument("--network", type=int, required=True)
    report_parser.add_argument(
        "--output-dir",
        type=Path,
        help="destination (default: outputs/qc/<input filename>)",
    )
    report_parser.add_argument(
        "--config",
        type=Path,
        help="TOML file containing [reader], [detection], and [qc] tables",
    )
    report_parser.add_argument(
        "--json",
        action="store_true",
        help="emit generated paths as JSON",
    )

    remediation_report_parser = subparsers.add_parser(
        "remediation-report",
        help="detect, repair in memory, and compare raw with cleaned snippets",
    )
    remediation_report_parser.add_argument("path", type=Path)
    remediation_report_parser.add_argument("--network", type=int, required=True)
    remediation_report_parser.add_argument(
        "--output-dir",
        type=Path,
        help="destination (default: outputs/remediation_qc/<input filename>)",
    )
    remediation_report_parser.add_argument(
        "--config",
        type=Path,
        help=(
            "TOML file containing [reader], [detection], [remediation], and "
            "[remediation_qc] tables"
        ),
    )
    remediation_report_parser.add_argument(
        "--json",
        action="store_true",
        help="emit generated paths and remediation summary as JSON",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        config = ReaderConfig.from_toml(args.config) if args.config else None
        with open_tod(args.path, config=config) as tod:
            if args.json:
                print(json.dumps(tod.info.to_dict(), indent=2))
            else:
                _print_summary(tod.info)
        return 0
    if args.command == "detect":
        reader_config = ReaderConfig.from_toml(args.config) if args.config else None
        detection_config = (
            DetectionConfig.from_toml(args.config) if args.config else None
        )
        with open_tod(args.path, config=reader_config) as tod:
            catalog = detect_network_issues(
                tod.network(args.network),
                config=detection_config,
            )
        if args.json:
            print(json.dumps(catalog.summary(), indent=2))
        else:
            _print_detection_summary(catalog)
        return 0
    if args.command == "report":
        reader_config = ReaderConfig.from_toml(args.config) if args.config else None
        detection_config = (
            DetectionConfig.from_toml(args.config) if args.config else None
        )
        qc_config = QCReportConfig.from_toml(args.config) if args.config else None
        output_directory = args.output_dir or Path("outputs/qc") / args.path.stem
        with open_tod(args.path, config=reader_config) as tod:
            network = tod.network(args.network)
            metadata = network.read_detector_metadata(["apt_uid"])
            catalog = detect_network_issues(network, config=detection_config)
        paths = generate_network_qc_report(
            catalog,
            output_directory,
            detector_metadata=metadata,
            config=qc_config,
        )
        result = {name: str(path) for name, path in vars(paths).items()}
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"network report: {paths.report_html}")
            print(f"manifest CSV: {paths.manifest_csv}")
        return 0
    if args.command == "remediation-report":
        reader_config = ReaderConfig.from_toml(args.config) if args.config else None
        detection_config = (
            DetectionConfig.from_toml(args.config)
            if args.config
            else DetectionConfig()
        )
        remediation_config = (
            RemediationConfig.from_toml(args.config)
            if args.config
            else RemediationConfig()
        )
        remediation_qc_config = (
            RemediationQCConfig.from_toml(args.config)
            if args.config
            else RemediationQCConfig()
        )
        output_directory = (
            args.output_dir
            or Path("outputs/remediation_qc") / args.path.stem
        )
        with open_tod(args.path, config=reader_config) as tod:
            network = tod.network(args.network)
            metadata = network.read_detector_metadata(["apt_uid"])
            signal, catalog = _detect_loaded_network(
                network,
                detection_config,
            )
            remediation = remediate_array(
                signal,
                catalog,
                config=remediation_config,
            )
            paths = generate_remediation_qc_report(
                signal,
                catalog,
                remediation,
                output_directory,
                detector_metadata=metadata,
                signal_units=tod.info.signal_units,
                config=remediation_qc_config,
            )
        if args.json:
            print(
                json.dumps(
                    {
                        "paths": paths.to_dict(),
                        "summary": remediation.summary(),
                    },
                    indent=2,
                )
            )
        else:
            print(f"remediation report: {paths.report_html}")
            print(
                f"detectors: {remediation.detector_count} retained, "
                f"{len(remediation.plan.removed_detector_indices)} removed"
            )
            print(f"snippet comparisons: {len(paths.raw_plots)}")
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


def _detect_loaded_network(
    network: NetworkTOD,
    config: DetectionConfig,
) -> tuple[np.ndarray, IssueCatalog]:
    """Load one signal array and reuse it for detection and remediation."""

    signal = network.read_signal()
    time = network.read_detector_time()
    source_flags = None
    if config.source_flag_variable is not None:
        if network.has_sample_data(config.source_flag_variable):
            source_flags = network.read_sample_data(config.source_flag_variable)
        elif config.require_source_flags:
            raise TolTODReadError(
                f"Required source-flag variable "
                f"{config.source_flag_variable!r} is missing or has "
                "incompatible dimensions"
            )
    catalog = detect_array_issues(
        signal,
        time,
        network_id=network.network_id,
        detector_indices=network.detector_indices,
        source_flags=source_flags,
        config=config,
        source_path=str(network.source_path),
    )
    return signal, catalog


def _print_summary(info: object) -> None:
    # Kept local to avoid formatting concerns in the I/O layer.
    from .models import ObservationInfo

    if not isinstance(info, ObservationInfo):
        raise TypeError("Expected ObservationInfo")
    print(f"file: {info.path}")
    print(f"samples: {info.sample_count}")
    print(f"detectors: {info.detector_count}")
    print(f"signal: {info.signal_variable} [{info.signal_units or 'units unknown'}]")
    print(
        f"telescope time: {info.time_variable} "
        f"[{info.time_units or 'units unknown'}]"
    )
    print(
        f"detector sample rate: {info.detector_sample_rate:.9g} Hz "
        f"({info.raw_detector_sample_rate:.9g} Hz / "
        f"{info.downsample_factor})"
    )
    if info.telescope_sample_interval is not None:
        print(
            "telescope coordinate cadence: "
            f"{1.0 / info.telescope_sample_interval:.9g} per source time unit"
        )
    if info.observation_number is not None:
        print(f"observation: {info.observation_number}")
    print(f"networks ({len(info.networks)}):")
    for network in info.networks:
        arrays = ",".join(str(value) for value in network.array_ids) or "unknown"
        print(
            f"  {network.network_id}: {network.detector_count} detectors "
            f"(source columns {network.detector_start}:{network.detector_stop}, "
            f"array {arrays})"
        )


def _print_detection_summary(catalog: IssueCatalog) -> None:
    summary = catalog.summary()
    print(f"file: {summary['source_path']}")
    print(f"network: {summary['network_id']}")
    print(f"shape: {summary['sample_count']} samples x {summary['detector_count']} detectors")
    print(f"elapsed: {summary['elapsed_seconds']:.3f} seconds")
    print("sample issues:")
    for name, count in summary["sample_detector_issue_counts"].items():
        print(f"  {name}: {count}")
    print("detector diagnostics:")
    for name, count in summary["detector_issue_counts"].items():
        print(f"  {name}: {count}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
