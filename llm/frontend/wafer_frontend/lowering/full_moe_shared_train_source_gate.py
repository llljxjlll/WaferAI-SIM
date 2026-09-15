"""Fail closed if a MoE full-TRAIN carrier retains displaced Dense MLP work.

The source E2E MoE graph and Dense template must agree exactly, including
all retained StateDecl IDs, expert/routed gate owners and both SGD steps.
Until a *real* full-MoE IR0→IR1/projection/schedule source graph replaces
Dense down→residual with combine→residual, this gate cannot certify runtime.
"""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.artifact_manifest import LinkedProgramManifest
from ..schema.flexible_dense_train import FlexibleDenseTrainPlan
from ..schema.full_dense_gradient_requirements import DenseFullTrainRequirements
from ..schema.full_moe_shared_train_requirements import (
    FullMoeSharedTrainRequirements,
)
from ..schema.full_training_physical_dag import FullTrainingPhysicalDAG
from ..schema.moe_compile_sequence import MoeCompileSequence


def require_moe_shared_train_no_dense_mlp(
    manifest: LinkedProgramManifest,
    dense_plan: FlexibleDenseTrainPlan,
    dense: DenseFullTrainRequirements,
    moe: MoeCompileSequence,
    requirements: FullMoeSharedTrainRequirements,
) -> None:
    """Reject real Dense 15-parameter motifs or just-relabelled MoE programs."""
    requirements.validate_against(dense_plan, dense, moe)
    manifest.validate("moe_train_replacement_physical_carrier")
    removed = set(requirements.excluded_dense_parameter_state_refs)
    actual = {abi.state_ref for fragment in manifest.fragments
              for abi in fragment.state_abi}
    if removed & actual:
        raise SchemaError("MoE TRAIN carrier physically retained displaced Dense MLP HBM state",
                          path="full_moe_train_source_replacement.state_abi")
    shared = set(requirements.shared_dense_parameter_state_refs)
    if actual & {template.state_ref for template
                 in dense_plan.parameter_templates} != shared:
        raise SchemaError("MoE TRAIN lacks exactly the 11 retained shared Dense source states",
                          path="full_moe_train_source_replacement.shared_spine")


def require_moe_full_train_production_source(
    manifest: LinkedProgramManifest,
    dag: FullTrainingPhysicalDAG,
    dense_plan: FlexibleDenseTrainPlan,
    dense: DenseFullTrainRequirements,
    moe: MoeCompileSequence,
    requirements: FullMoeSharedTrainRequirements,
) -> None:
    """One replacement source for every layer/step, or refuse full TRAIN."""
    require_moe_shared_train_no_dense_mlp(manifest, dense_plan, dense,
                                           moe, requirements)
    dag.validate()
    actions = {(action.step, action.operation_ref): action
               for action in dag.actions}
    operation_refs = {action.operation_ref for action in dag.actions}
    replaced = {ref for layer in requirements.layer_replacements
                for ref in layer.replaced_dense_forward_refs}
    excluded_backward = {f"backward::{ref}" for ref in replaced}
    if operation_refs & (replaced | excluded_backward):
        raise SchemaError("MoE TRAIN physically executes displaced Dense MLP action",
                          path="full_moe_train_source_replacement.actions")
    if moe.id not in dag.source_artifact_ids or any(
        unit.id not in dag.source_artifact_ids for unit in moe.units
    ):
        raise SchemaError("four real MoE TRAIN units absent from trusted source artifacts",
                          path="full_moe_train_source_replacement.source_artifact_ids")
    for layer in requirements.layer_replacements:
        for step in range(dense.steps):
            if any((step, ref) not in actions for ref in (
                *layer.moe_forward_refs_by_step[step],
                *layer.moe_backward_refs_by_step[step],
            )):
                raise SchemaError("MoE replacement lacks one of its source router/expert/gradient ops",
                                  path=f"full_moe_train_source_replacement.step{step}.layer{layer.layer}")
    if not requirements.source_ir0_replacement_materialized:
        raise SchemaError("MoE source IR0 has no typed router/expert→combine→shared residual replacement",
                          path="full_moe_train_source_replacement.full_source_ir0")


__all__ = ["require_moe_shared_train_no_dense_mlp",
           "require_moe_full_train_production_source"]
