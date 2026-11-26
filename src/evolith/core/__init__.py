"""Engine-agnostic Evolith core primitives: deterministic SystemIdentity,
Envelope Failure Log, and Declarative Surface / Action Ladder analysis.
"""

from __future__ import annotations

from evolith.core.envelope import EnvelopeFailureLog, EnvelopeGap, EnvelopeGapCategory
from evolith.core.formal_replay import (
    DoubleBufferSlotTracker,
    KvPrefixSpecLedger,
    ProtocolSuiteReport,
    ProtocolTrace,
    WorkbenchTxParticipant,
    run_protocol_verification_suite,
)
from evolith.core.identity import (
    IdentityDiff,
    SystemIdentity,
    compute_system_identity,
    diff_identities,
    materialize_readonly_snapshot,
)
from evolith.core.surface import (
    ActionRung,
    BlastRadiusPlan,
    ConfigBindingMode,
    ConfigKeyDecl,
    StateResourceDecl,
    TargetSurface,
    UnitDecl,
    analyze_changeset_blast_radius,
    are_patches_closure_orthogonal,
    default_zstep_surface,
)

__all__ = [
    "ActionRung",
    "BlastRadiusPlan",
    "ConfigBindingMode",
    "ConfigKeyDecl",
    "DoubleBufferSlotTracker",
    "EnvelopeFailureLog",
    "EnvelopeGap",
    "EnvelopeGapCategory",
    "IdentityDiff",
    "KvPrefixSpecLedger",
    "ProtocolSuiteReport",
    "ProtocolTrace",
    "StateResourceDecl",
    "SystemIdentity",
    "TargetSurface",
    "UnitDecl",
    "WorkbenchTxParticipant",
    "analyze_changeset_blast_radius",
    "are_patches_closure_orthogonal",
    "compute_system_identity",
    "default_zstep_surface",
    "diff_identities",
    "materialize_readonly_snapshot",
    "run_protocol_verification_suite",
]
