"""Repair and detector-rejection planning without signal mutation."""

from .config import RemediationConfig
from .executor import apply_remediation_plan, remediate_array
from .models import (
    AppliedRepair,
    DetectorDisposition,
    DetectorRemediation,
    JumpCorrection,
    RemediationPlan,
    RemediationResult,
    RemovalReason,
    RepairAction,
    RepairSpan,
)
from .policy import ACTIONABLE_ISSUES, build_remediation_plan

__all__ = [
    "ACTIONABLE_ISSUES",
    "AppliedRepair",
    "DetectorDisposition",
    "DetectorRemediation",
    "JumpCorrection",
    "RemediationConfig",
    "RemediationPlan",
    "RemediationResult",
    "RemovalReason",
    "RepairAction",
    "RepairSpan",
    "apply_remediation_plan",
    "build_remediation_plan",
    "remediate_array",
]
