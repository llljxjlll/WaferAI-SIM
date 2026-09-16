"""Compile one two-layer/two-step Dense TRAIN source to real native work."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..errors import SchemaError, UnsupportedFeatureError
from ..lowering.full_dense_gradient_physical_gate import (
    require_full_dense_physical_gradient_paths,
)
from ..lowering.full_dense_two_step_physical_dag import (
    build_full_dense_two_step_physical_dag,
    dense_two_step_native_opcode_contract,
)
from ..lowering.full_training_program_merger import (
    require_full_training_opcode_matrix,
    require_independent_ce_loss_gradient_seed,
)
from ..schema.flexible_dense_train import FlexibleDenseTrainPlan
from ..schema.full_dense_gradient_requirements import (
    DenseFullTrainRequirements, build_dense_full_train_requirements,
)
from ..schema.full_training_physical_dag import FullTrainingPhysicalDAG
from ..schema.ir0 import OpKind
from ..schema.train_n6 import TrainLinkedProgram
from .full_dense_training_two_step_ir0 import build_full_dense_training_two_step_ir0
from .fusion_partition import partition_train_forward
from .inter_die_plan import plan_train_forward
from .intra_die_schedule import schedule_train_forward
from .load_fabric import hbm_address_spaces_from_data, physical_fabric_from_data
from .placement import place_train_forward_ir0
from .project_to_ir2 import project_train_forward
from .train_global_action import build_train_global_action
from .train_link_program import link_train
from .train_lower_program import lower_train
from ..policies.registry import RegistryKind, production_registry
from ..schema.n4 import FusionPartitionContext, InterDiePlanningContext
from ..schema.n5 import IntraDieSchedulingContext, ProjectToIR2Context
from ..schema.placement import PlacementContext


@dataclass(frozen=True, slots=True)
class FullDenseTwoStepNative:
    program: TrainLinkedProgram
    physical_dag: FullTrainingPhysicalDAG
    requirements: DenseFullTrainRequirements
    loss_gradient_seed_abi_by_step: Mapping[int, str]


def compile_full_dense_two_step_native(
    plan: FlexibleDenseTrainPlan,
    hardware_data: Mapping[str, object],
) -> FullDenseTwoStepNative:
    """Require exact IR0→IR1→IR2→schedule→actions→native gradient closure."""
    plan.validate("dense_two_step_plan")
    if plan.spec.mesh.rank_count != 1:
        raise UnsupportedFeatureError(
            "complete two-step native Dense backward currently requires TP1/DP1",
            path="plan.mesh",
        )
    producer = "full_dense_training_two_step_native"
    source = build_full_dense_training_two_step_ir0(plan)
    placed = place_train_forward_ir0(source, PlacementContext.create(
        producer_pass=producer,
        fabric=physical_fabric_from_data(hardware_data),
        placement=plan.source_experiment.placement,
        hbm_address_spaces=hbm_address_spaces_from_data(hardware_data),
    ))
    partitioned = partition_train_forward(
        placed, FusionPartitionContext.create(producer_pass=producer),
    )
    registry = production_registry()
    planned = plan_train_forward(partitioned, InterDiePlanningContext.create(
        producer_pass=producer,
        fused_policy=registry.instantiate(
            RegistryKind.INTER_DIE, "naive").selection,
        standalone_policy=registry.instantiate(
            RegistryKind.STANDALONE_COLLECTIVE,
            "direct_all_gather").selection,
    ))
    projected = project_train_forward(planned, ProjectToIR2Context.create(
        producer_pass=producer, state_transfers=(),
    ))
    scheduled = schedule_train_forward(projected, IntraDieSchedulingContext.create(
        producer_pass=producer,
        policy=registry.instantiate(RegistryKind.INTRA_DIE,
                                    "naive").selection,
    ))
    program = link_train(lower_train(build_train_global_action(scheduled)))
    requirements = build_dense_full_train_requirements(plan, steps=2)
    physical = build_full_dense_two_step_physical_dag(program, plan, requirements)
    backward, wgrad = dense_two_step_native_opcode_contract(plan)
    require_full_dense_physical_gradient_paths(
        program.manifest, plan, requirements, physical,
        required_backward_opcodes=backward,
        required_wgrad_opcodes=wgrad,
    )
    require_full_training_opcode_matrix(physical, require_moe=False)
    graph = program.source.replicas[0].lowering_context.ir1
    gradients = {node.id: node.inputs[2] for node in graph.nodes
                 if node.kind is OpKind.CE_BACKWARD}
    if len(gradients) != 2:
        raise SchemaError("two native CE backward nodes require distinct dLoss inputs",
                          path="program.source")
    abis = {abi.value_id: abi.id for fragment in program.manifest.fragments
            for abi in fragment.buffer_abi
            if abi.value_id in gradients.values()}
    seeds = {}
    for node_ref, gradient_ref in gradients.items():
        for step in (0, 1):
            if node_ref.endswith(f"::step{step}_backward__dp0"):
                if gradient_ref not in abis or step in seeds:
                    raise SchemaError("independent CE dLoss BufferABI is missing",
                                      path=node_ref)
                seeds[step] = abis[gradient_ref]
                break
        else:
            raise SchemaError("CE backward lacks exact step identity", path=node_ref)
    require_independent_ce_loss_gradient_seed(
        program.manifest, physical, seed_abi_by_step=seeds,
    )
    return FullDenseTwoStepNative(program, physical, requirements, seeds)


__all__ = ["FullDenseTwoStepNative", "compile_full_dense_two_step_native"]
