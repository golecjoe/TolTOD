"""Quality-control aggregation, plots, and network reports."""

from .aggregate import DetectorManifest, DetectorManifestRow, build_detector_manifest
from .config import QCReportConfig, RemediationQCConfig
from .remediation_report import (
    RemediationQCReportPaths,
    generate_remediation_qc_report,
)
from .report import NetworkQCReportPaths, generate_network_qc_report

__all__ = [
    "DetectorManifest",
    "DetectorManifestRow",
    "NetworkQCReportPaths",
    "QCReportConfig",
    "RemediationQCConfig",
    "RemediationQCReportPaths",
    "build_detector_manifest",
    "generate_network_qc_report",
    "generate_remediation_qc_report",
]
