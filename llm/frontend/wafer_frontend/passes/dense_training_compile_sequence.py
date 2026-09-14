"""Bind two P3 Dense SGD steps to the real flexible Dense lower/link path."""

from __future__ import annotations

from ..errors import SchemaError, UnsupportedFeatureError
from ..schema.e2e_workload_graph import E2EOperationKind
from ..schema.experiment import ExperimentSpec, WorkloadMode
from ..schema.flexible_dense_train import FlexibleDenseTrainActionKind
from ..schema.ir1 import PhysicalFabric
from ..schema.memory_plan import MemoryPlanExecution
from ..schema.persistent_state import HbmAddressSpace
from ..schema.rect_mesh import RectMeshSpec
from ..schema.serde import canonical_digest
from ..schema.workload_materialization import (
    WorkloadMaterializationManifest,
    WorkloadMaterializationStatus,
)
from ..schema.workload_run import (
    WorkloadExecutionStrategy,
    WorkloadFamily,
    WorkloadMemoryMode,
    WorkloadModelArchitecture,
    WorkloadOptimizerKind,
)
from ..schema.dense_training_compile_sequence import (
    DenseTrainingCompileSegment,
    DenseTrainingCompileSequence,
    DenseTrainingLegacyShardBinding,
    DenseTrainingParameterBinding,
    DenseTrainingSyncLowering,
)
from .flexible_dense_backward import materialize_flexible_dense_backward
from .flexible_dense_train import materialize_flexible_dense_train_forward


def _reject(reasons: list[str]) -> None:
    if reasons:
        raise UnsupportedFeatureError(
            "request is outside Dense training compile-sequence subset: "
            + ", ".join(sorted(set(reasons))),
            path="dense_training_compile_sequence",
            code="dense_training_compile_sequence_unsupported",
        )


def _validate_inputs(
    manifest: WorkloadMaterializationManifest,
    legacy_template: ExperimentSpec,
    fabric: PhysicalFabric,
    hbm_address_spaces: tuple[HbmAddressSpace, ...],
) -> None:
    manifest.validate("manifest")
    legacy_template.validate("legacy_template")
    fabric.validate("fabric")
    request = manifest.request
    reasons: list[str] = []
    training = request.steps.training
    if manifest.status is not WorkloadMaterializationStatus.PARTIAL:
        reasons.append("manifest.partial_required")
    if (
        request.family is not WorkloadFamily.DENSE_TRAINING
        or request.model.architecture is not WorkloadModelArchitecture.LLAMA_DENSE
    ):
        reasons.append("request.dense_training_required")
    if training is None or training.step_count != 2:
        reasons.append("request.exactly_two_steps_required")
    if (
        request.optimizer is None
        or request.optimizer.kind is not WorkloadOptimizerKind.SGD
    ):
        reasons.append("request.sgd_required")
    if request.memory.mode is not WorkloadMemoryMode.RESIDENT_HBM:
        reasons.append("request.resident_hbm_required")
    if manifest.memory_plan.execution is not MemoryPlanExecution.NO_EXTERNAL_TRANSPORT_REQUIRED:
        reasons.append("manifest.external_transport_not_supported")
    if (
        not request.execution.timing
        or request.execution.functional
        or request.execution.strategy is not WorkloadExecutionStrategy.BASELINE
    ):
        reasons.append("request.baseline_timing_only_required")
    if (
        request.parallel.tp != request.mesh.columns
        or request.parallel.dp != request.mesh.rows
        or (request.parallel.ep, request.parallel.pp) != (1, 1)
    ):
        reasons.append("request.tp_columns_dp_rows_required")
    expected_dies = tuple(range(request.mesh.rank_count))
    if request.parallel.active_die_ids and request.parallel.active_die_ids != expected_dies:
        reasons.append("request.full_row_major_mesh_required")
    if manifest.placement.active_die_ids != expected_dies:
        reasons.append("manifest.full_row_major_mesh_required")
    if fabric.die_grid != (request.mesh.columns, request.mesh.rows):
        reasons.append("fabric.mesh_shape_mismatch")
    if len(fabric.dies) != request.mesh.rank_count:
        reasons.append("fabric.complete_rectangle_required")
    if legacy_template.workload.mode is not WorkloadMode.TRAIN:
        reasons.append("legacy.train_template_required")
    else:
        legacy_train = legacy_template.workload.train
        assert legacy_train is not None
        if training is not None and (
            legacy_train.global_batch,
            legacy_train.micro_batch,
            legacy_train.seq_len,
        ) != (
            training.global_batch_size,
            training.micro_batch_size,
            training.sequence_length,
        ):
            reasons.append("legacy.training_shape_mismatch")
    instances = legacy_template.parallel.instances
    if len(instances) != 1:
        reasons.append("legacy.single_instance_required")
    else:
        instance = instances[0]
        if (
            instance.tp,
            instance.dp,
            instance.ep,
            instance.pp,
        ) != (
            request.parallel.tp,
            request.parallel.dp,
            1,
            1,
        ):
            reasons.append("legacy.parallel_mismatch")
    model = legacy_template.model
    target = request.model
    if (
        model.V,
        model.H,
        model.I,
        model.L,
        model.NH,
        model.KVH,
        model.DH,
        model.max_position_embeddings,
        model.dtype,
    ) != (
        target.vocabulary_size,
        target.hidden_size,
        target.intermediate_size,
        target.num_layers,
        target.num_attention_heads,
        target.num_kv_heads,
        target.head_dim,
        target.max_sequence_length,
        target.dtype,
    ):
        reasons.append("legacy.model_mismatch")
    if (
        type(hbm_address_spaces) is not tuple
        or len(hbm_address_spaces) != request.mesh.rank_count
    ):
        reasons.append("hbm.one_address_space_per_die_required")
    else:
        for index, address_space in enumerate(hbm_address_spaces):
            if type(address_space) is not HbmAddressSpace:
                raise SchemaError(
                    "must be an HbmAddressSpace",
                    path=f"hbm_address_spaces[{index}]",
                )
            address_space.validate(f"hbm_address_spaces[{index}]")
        if tuple(item.die_id for item in hbm_address_spaces) != expected_dies:
            reasons.append("hbm.row_major_die_order_required")
    _reject(reasons)


def _legacy_tensor_ref(instance_id: str, parameter_ref: str) -> str:
    direct = {
        "embedding.weight": "tok_embeddings.weight",
        "final_norm.weight": "final_norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    suffix = direct.get(parameter_ref)
    if suffix is None:
        parts = parameter_ref.split(".")
        if len(parts) != 4 or parts[0] != "layer" or parts[3] != "weight":
            raise SchemaError("unrecognized P3 Dense parameter", path="parameter_ref")
        layer = parts[1]
        component = {
            "input_norm": "w_norm1",
            "qkv": "w_qkv",
            "attention_out": "w_o",
            "post_norm": "w_norm2",
            "mlp_gate": "w_gate_up",
            "mlp_up": "w_gate_up",
            "mlp_down": "w_down",
        }.get(parts[2])
        if component is None:
            raise SchemaError("unrecognized P3 Dense parameter", path="parameter_ref")
        suffix = f"layer{layer}.{component}"
    return f"{instance_id}.{suffix}"


def _operation(graph, kind, step, parameter_ref):
    matches = tuple(
        item
        for item in graph.operations
        if item.kind is kind
        and item.step == step
        and item.parameter_ref == parameter_ref
    )
    if len(matches) != 1:
        raise SchemaError(
            "requires exactly one P3 parameter operation",
            path=f"logical_graph.{kind.value}",
        )
    return matches[0]


def _legacy_shards(
    linked_program,
    p3_graph,
    input_state_ref: str,
    parameter_ref: str,
) -> tuple[DenseTrainingLegacyShardBinding, ...]:
    plan = linked_program.plan
    instance_id = plan.source_experiment.parallel.instances[0].id
    tensor_ref = _legacy_tensor_ref(instance_id, parameter_ref)
    templates = tuple(
        sorted(
            (
                item
                for item in plan.parameter_templates
                if item.tensor_ref == tensor_ref
            ),
            key=lambda item: item.tp_shard_index,
        )
    )
    if len(templates) != plan.spec.tp_degree:
        raise SchemaError("legacy TP parameter coverage is incomplete", path="legacy_program")
    state_abis = tuple(
        item
        for fragment in linked_program.manifest.fragments
        for item in fragment.state_abi
    )
    actions = linked_program.plan.rank_actions
    p3_values = tuple(
        item
        for item in p3_graph.tensor_values
        if item.state_ref == input_state_ref
    )
    result = []
    for template in templates:
        shard_values = tuple(
            sorted(
                (
                    item
                    for item in p3_values
                    if item.tp_shard == template.tp_shard_index
                ),
                key=lambda item: item.logical_rank,
            )
        )
        sizes = {item.size_bytes for item in shard_values}
        if len(sizes) != 1:
            raise SchemaError("P3 parameter shard bytes are ambiguous", path="logical_graph")
        logical_bytes = sizes.pop()
        if parameter_ref.endswith(".mlp_gate.weight"):
            byte_offset = 0
        elif parameter_ref.endswith(".mlp_up.weight"):
            byte_offset = logical_bytes
        elif template.weight_bytes == logical_bytes:
            byte_offset = 0
        elif template.weight_bytes == logical_bytes * plan.spec.tp_degree:
            byte_offset = template.tp_shard_index * logical_bytes
        else:
            raise SchemaError(
                "legacy parameter cannot represent the P3 TP slice",
                path="legacy_program",
            )
        def action_refs(kind):
            return tuple(sorted(
                item.id
                for item in actions
                if item.kind is kind and item.state_ref == template.state_ref
            ))
        abis = tuple(sorted(
            (
                item for item in state_abis
                if item.state_ref == template.state_ref
            ),
            key=lambda item: item.die_id,
        ))
        result.append(DenseTrainingLegacyShardBinding.create(
            tp_shard=template.tp_shard_index,
            owner_ranks=tuple(item.logical_rank for item in shard_values),
            legacy_tensor_ref=template.tensor_ref,
            legacy_state_ref=template.state_ref,
            legacy_byte_offset=byte_offset,
            logical_bytes=logical_bytes,
            load_action_refs=action_refs(
                FlexibleDenseTrainActionKind.PARAMETER_LOAD
            ),
            wgrad_action_refs=action_refs(
                FlexibleDenseTrainActionKind.WEIGHT_GRADIENT
            ),
            sync_action_refs=action_refs(
                FlexibleDenseTrainActionKind.GRADIENT_SYNC
            ),
            sgd_action_refs=action_refs(
                FlexibleDenseTrainActionKind.SGD_UPDATE
            ),
            store_action_refs=action_refs(
                FlexibleDenseTrainActionKind.PARAMETER_STORE
            ),
            state_abi_refs=tuple(item.id for item in abis),
            hbm_addresses=tuple(item.address for item in abis),
            sync_lowering=(
                DenseTrainingSyncLowering.SINGLETON_NOOP
                if plan.spec.dp_degree == 1
                else DenseTrainingSyncLowering.DP_TREE
            ),
        ))
    return tuple(result)


def _parameter_binding(linked_program, graph, step, parameter_ref):
    load = _operation(graph, E2EOperationKind.PARAMETER_LOAD, step, parameter_ref)
    wgrad = _operation(graph, E2EOperationKind.WGRAD, step, parameter_ref)
    sync = _operation(graph, E2EOperationKind.GRADIENT_SYNC, step, parameter_ref)
    sgd = _operation(graph, E2EOperationKind.SGD_UPDATE, step, parameter_ref)
    store = _operation(graph, E2EOperationKind.PARAMETER_STORE, step, parameter_ref)
    if (
        len(load.reads) != 1
        or len(wgrad.writes) != 1
        or len(sync.writes) != 1
        or len(sgd.writes) != 1
    ):
        raise SchemaError("P3 parameter state arity drifted", path="logical_graph")
    return DenseTrainingParameterBinding.create(
        step=step,
        parameter_ref=parameter_ref,
        layer=load.layer,
        load_operation_ref=load.id,
        wgrad_operation_ref=wgrad.id,
        sync_operation_ref=sync.id,
        sgd_operation_ref=sgd.id,
        store_operation_ref=store.id,
        input_parameter_version=step,
        input_parameter_state_ref=load.reads[0],
        raw_gradient_state_ref=wgrad.writes[0],
        synced_gradient_state_ref=sync.writes[0],
        output_parameter_version=step + 1,
        output_parameter_state_ref=sgd.writes[0],
        legacy_shards=_legacy_shards(
            linked_program,
            graph,
            load.reads[0],
            parameter_ref,
        ),
    )


def compile_dense_training_sequence(
    manifest: WorkloadMaterializationManifest,
    legacy_template: ExperimentSpec,
    fabric: PhysicalFabric,
    *,
    hbm_address_spaces: tuple[HbmAddressSpace, ...],
) -> DenseTrainingCompileSequence:
    """Compile a real one-step program and bind it to both P3 SGD steps."""

    _validate_inputs(manifest, legacy_template, fabric, hbm_address_spaces)
    mesh = RectMeshSpec(manifest.request.mesh.rows, manifest.request.mesh.columns)
    forward = materialize_flexible_dense_train_forward(
        legacy_template,
        mesh,
        fabric,
        hbm_address_spaces,
        producer_pass="dense_training_compile_sequence",
    )
    linked_program = materialize_flexible_dense_backward(
        forward,
        fabric,
        hbm_address_spaces,
    )
    request = manifest.request
    assert request.optimizer is not None
    if linked_program.plan.spec.learning_rate != request.optimizer.learning_rate:
        raise UnsupportedFeatureError(
            "legacy learning rate differs from P3 SGD request",
            path="dense_training_compile_sequence",
        )
    graph = manifest.logical_graph
    parameters = tuple(sorted({
        item.parameter_ref
        for item in graph.operations
        if item.kind is E2EOperationKind.PARAMETER_LOAD
        and item.parameter_ref is not None
    }))
    segments = tuple(
        DenseTrainingCompileSegment.create(
            step=step,
            parameter_bindings=tuple(
                _parameter_binding(linked_program, graph, step, parameter)
                for parameter in parameters
            ),
            linked_program=linked_program,
            linked_manifest_digest=canonical_digest(linked_program.manifest),
            one_shot_workload_end=step == 1,
        )
        for step in range(2)
    )
    return DenseTrainingCompileSequence.create(
        materialization=manifest,
        segments=segments,
    )


__all__ = ["compile_dense_training_sequence"]
