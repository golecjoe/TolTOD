"""TolTEC time-ordered data ingestion and quality-control framework."""

from .config import ReaderConfig
from .detection import (
    DetectionConfig,
    DetectorIssue,
    DetectorMetrics,
    IssueCatalog,
    IssueEvent,
    SampleIssue,
    detect_array_issues,
    detect_network_issues,
)
from .errors import (
    RemediationError,
    ReaderClosedError,
    SchemaError,
    TolTODReadError,
    UnknownNetworkError,
)
from .io.reader import NetworkTOD, TolTECFile, open_tod
from .models import NetworkInfo, ObservationInfo
from .qc import (
    DetectorManifest,
    DetectorManifestRow,
    NetworkQCReportPaths,
    QCReportConfig,
    RemediationQCConfig,
    RemediationQCReportPaths,
    build_detector_manifest,
    generate_network_qc_report,
    generate_remediation_qc_report,
)
from .remediation import (
    AppliedRepair,
    DetectorDisposition,
    DetectorRemediation,
    JumpCorrection,
    RemediationConfig,
    RemediationPlan,
    RemediationResult,
    RemovalReason,
    RepairAction,
    RepairSpan,
    apply_remediation_plan,
    build_remediation_plan,
    remediate_array,
)

__all__ = [
    "DetectionConfig",
    "AppliedRepair",
    "DetectorManifest",
    "DetectorManifestRow",
    "DetectorIssue",
    "DetectorMetrics",
    "DetectorDisposition",
    "DetectorRemediation",
    "IssueCatalog",
    "IssueEvent",
    "JumpCorrection",
    "NetworkInfo",
    "NetworkQCReportPaths",
    "NetworkTOD",
    "ObservationInfo",
    "QCReportConfig",
    "RemediationQCConfig",
    "RemediationQCReportPaths",
    "ReaderClosedError",
    "ReaderConfig",
    "RemediationError",
    "RemediationConfig",
    "RemediationPlan",
    "RemediationResult",
    "RemovalReason",
    "RepairAction",
    "RepairSpan",
    "SchemaError",
    "SampleIssue",
    "TolTECFile",
    "TolTODReadError",
    "UnknownNetworkError",
    "build_detector_manifest",
    "apply_remediation_plan",
    "build_remediation_plan",
    "detect_array_issues",
    "detect_network_issues",
    "generate_network_qc_report",
    "generate_remediation_qc_report",
    "open_tod",
    "remediate_array",
]

__version__ = "0.1.0"
