"""Replaceable frontend policy registry."""

from .interfaces import (
    FusionPartition,
    GlobalActionDAGBuilder,
    InterDiePolicy,
    IntraDiePolicy,
    ProjectToIR2,
    StandaloneCollectivePolicy,
)
from .registry import (
    POLICY_SELECTION_SCHEMA_VERSION,
    PolicyRegistry,
    PolicySelection,
    Registration,
    RegistrationState,
    RegistryKind,
    ResolvedPolicy,
    default_registry,
    production_registry,
)
from .naive_fusion_partition import NaiveFusionPartition
from .naive_intra_die import (
    NAIVE_INTRADIE_POLICY_SCHEMA_VERSION,
    NaiveIntraDiePolicy,
)
from .naive_inter_die import DirectAllGatherPolicy, NaiveInterDiePolicy
from .optimized_intra_die import (
    OPTIMIZED_INTRADIE_POLICY_SCHEMA_VERSION,
    OptimizedIntraDiePolicy,
)
from .naive_project_to_ir2 import NaiveProjectToIR2
from .intra_die_v2_search import evaluate_intra_die_v2_candidates

__all__ = [
    "FusionPartition",
    "GlobalActionDAGBuilder",
    "InterDiePolicy",
    "IntraDiePolicy",
    "evaluate_intra_die_v2_candidates",
    "NaiveFusionPartition",
    "NAIVE_INTRADIE_POLICY_SCHEMA_VERSION",
    "NaiveIntraDiePolicy",
    "OPTIMIZED_INTRADIE_POLICY_SCHEMA_VERSION",
    "OptimizedIntraDiePolicy",
    "NaiveInterDiePolicy",
    "NaiveProjectToIR2",
    "DirectAllGatherPolicy",
    "PolicyRegistry",
    "POLICY_SELECTION_SCHEMA_VERSION",
    "PolicySelection",
    "Registration",
    "RegistrationState",
    "ResolvedPolicy",
    "default_registry",
    "production_registry",
    "ProjectToIR2",
    "RegistryKind",
    "StandaloneCollectivePolicy",
]
from .swizzle_topo import SwizzleFusionPartition, SwizzlePlanner
__all__ += [
    "SwizzleFusionPartition",
    "SwizzlePlanner",
]
