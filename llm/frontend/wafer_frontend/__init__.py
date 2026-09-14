"""Deterministic frontend contracts for WaferAI-SIM."""

from .compiler import (
    RECT_MESH_COMPILATION_SCHEMA_VERSION,
    NaiveCompilation,
    RectMeshCompilation,
    compile_naive,
    compile_rect_mesh,
)
from .flexible_mesh_compiler import (
    FlexibleMeshCapabilityStatus,
    FlexibleMeshCompilation,
    FlexibleMeshExecutionPlan,
    FlexibleMeshWorkloadCapabilityReport,
    compile_flexible_mesh_workload,
)
from .passes.flexible_dense_train import (
    build_flexible_dense_train_plan,
    materialize_flexible_dense_train_forward,
)
from .passes.flexible_dense_backward import (
    build_flexible_dense_backward_program_io,
    materialize_flexible_dense_backward,
)
from .passes.flexible_moe import (
    adapt_ep4_scale_spec,
    build_round_robin_flexible_moe_spec,
    compile_flexible_moe_baseline,
)
from .lowering.flexible_moe_standard import (
    build_flexible_moe_standard_program_io_plan,
    plan_flexible_moe_standard_mapping,
)
from .lowering.flexible_moe_production import (
    FlexibleMoeProductionArtifacts,
    FlexibleMoeRuntimeEvidence,
    build_flexible_moe_production_program_io,
    lower_link_flexible_moe_production,
    observe_flexible_moe_production_runtime,
)
from .lowering.swizzle_meshslice_fallback import (
    link_meshslice_unfused_fallback_program,
    MeshSliceExecutableFallback,
    MeshSliceFallbackReason,
    MeshSliceSelectedPath,
)
from .policies.swizzle.meshslice_2d import (
    MeshSliceExecutionMode,
    meshslice_execution_mode,
)
from .runner import (
    NAIVE_RUN_REPORT_SCHEMA_VERSION,
    NaiveRunCase,
    NaiveRunReport,
    NaiveRunRequest,
    NaiveRunResult,
    NaiveRunValidation,
    run_naive,
)
from .schema.experiment import EXPERIMENT_SCHEMA_VERSION, ExperimentSpec
from .schema.rect_mesh import RectMeshSpec
from .schema.rect_mesh_compile import (
    RectMeshCompileCapabilityReport,
    RectMeshCompileChain,
    RectMeshCompileMode,
    RectMeshFallbackReason,
)
from .schema.flexible_dense_train import (
    FlexibleDenseTrainForwardCarrier,
    FlexibleDenseTrainPlan,
    FlexibleDenseTrainSpec,
)
from .schema.flexible_dense_backward import (
    FlexibleDenseBackwardLinkedProgram,
)
from .schema.flexible_mesh_capacity import (
    FlexibleMeshCapacityDemand,
    FlexibleMeshCapacityProfile,
)
from .schema.flexible_mesh_groups import (
    FlexibleMeshAxisMapping,
    FlexibleMeshGroupRegistry,
)
from .schema.flexible_mesh_workload import (
    FlexibleMeshSliceMode,
    FlexibleMeshSliceOperation,
    FlexibleMeshSliceSpec,
    FlexibleMeshWorkloadKind,
    FlexibleMeshWorkloadSpec,
    FlexibleTrainSpec,
)
from .schema.flexible_mesh_runtime import (
    FlexibleMeshArtifactCapacityEvidence,
    FlexibleMeshRuntimeBaseline,
    FlexibleMeshRuntimeCase,
    FlexibleMeshRuntimeEvidence,
    FlexibleMeshRuntimeFailure,
    FlexibleMeshRuntimeMarker,
    FlexibleMeshRuntimeResidual,
    FlexibleMeshRuntimeStage,
    FlexibleMeshRuntimeStageStatus,
)
from .schema.flexible_moe import (
    FlexibleMoeExecutablePlan,
    FlexibleMoeLimits,
    FlexibleMoeMode,
    FlexibleMoeSpec,
    MoeRectStaticTrace,
)
from .schema.flexible_moe_standard import (
    FlexibleMoeStandardLoweringPlan,
    FlexibleMoeStandardProgramIoPlan,
)

__all__ = [
    "EXPERIMENT_SCHEMA_VERSION",
    "ExperimentSpec",
    "FlexibleDenseBackwardLinkedProgram",
    "FlexibleDenseTrainForwardCarrier",
    "FlexibleDenseTrainPlan",
    "FlexibleDenseTrainSpec",
    "FlexibleMeshAxisMapping",
    "FlexibleMeshCapabilityStatus",
    "FlexibleMeshCapacityDemand",
    "FlexibleMeshCapacityProfile",
    "FlexibleMeshCompilation",
    "FlexibleMeshExecutionPlan",
    "FlexibleMeshGroupRegistry",
    "FlexibleMeshArtifactCapacityEvidence",
    "FlexibleMeshRuntimeBaseline",
    "FlexibleMeshRuntimeCase",
    "FlexibleMeshRuntimeEvidence",
    "FlexibleMeshRuntimeFailure",
    "FlexibleMeshRuntimeMarker",
    "FlexibleMeshRuntimeResidual",
    "FlexibleMeshRuntimeStage",
    "FlexibleMeshRuntimeStageStatus",
    "FlexibleMeshSliceMode",
    "FlexibleMeshSliceOperation",
    "FlexibleMeshSliceSpec",
    "FlexibleMeshWorkloadCapabilityReport",
    "FlexibleMeshWorkloadKind",
    "FlexibleMeshWorkloadSpec",
    "FlexibleMoeExecutablePlan",
    "FlexibleMoeLimits",
    "FlexibleMoeMode",
    "FlexibleMoeProductionArtifacts",
    "FlexibleMoeRuntimeEvidence",
    "FlexibleMoeSpec",
    "FlexibleMoeStandardLoweringPlan",
    "FlexibleMoeStandardProgramIoPlan",
    "FlexibleTrainSpec",
    "MeshSliceExecutionMode",
    "MeshSliceExecutableFallback",
    "MeshSliceFallbackReason",
    "MeshSliceSelectedPath",
    "MoeRectStaticTrace",
    "NaiveCompilation",
    "RECT_MESH_COMPILATION_SCHEMA_VERSION",
    "RectMeshCompilation",
    "RectMeshCompileCapabilityReport",
    "RectMeshCompileChain",
    "RectMeshCompileMode",
    "RectMeshFallbackReason",
    "RectMeshSpec",
    "NAIVE_RUN_REPORT_SCHEMA_VERSION",
    "NaiveRunCase",
    "NaiveRunReport",
    "NaiveRunRequest",
    "NaiveRunResult",
    "NaiveRunValidation",
    "compile_naive",
    "compile_flexible_mesh_workload",
    "compile_flexible_moe_baseline",
    "compile_rect_mesh",
    "adapt_ep4_scale_spec",
    "build_flexible_dense_backward_program_io",
    "build_flexible_dense_train_plan",
    "build_flexible_moe_production_program_io",
    "build_round_robin_flexible_moe_spec",
    "build_flexible_moe_standard_program_io_plan",
    "link_meshslice_unfused_fallback_program",
    "materialize_flexible_dense_train_forward",
    "materialize_flexible_dense_backward",
    "meshslice_execution_mode",
    "lower_link_flexible_moe_production",
    "observe_flexible_moe_production_runtime",
    "plan_flexible_moe_standard_mapping",
    "run_naive",
]
