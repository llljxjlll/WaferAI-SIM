"""Source-only Dense-spine→MoE-block replacement obligations for two-step TRAIN.

The analytic Dense graph is a *template*: its MLP gate/up/SwiGLU/down must
be replaced before any physical MoE full-model program is certified.  This
artifact names those exclusions and exact MoE parameter owners/lineage; it
does not invent a complete MoE IR0, IR1, projection, or executable timeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .common import DType
from .e2e_workload_graph import E2EStateKind
from .flexible_dense_train import FlexibleDenseTrainPlan
from .full_dense_gradient_requirements import DenseFullTrainRequirements
from .moe_compile_sequence import MoeCompileSequence
from .serde import canonical_digest
from .workload_run import WorkloadFamily, WorkloadOptimizerKind


@dataclass(frozen=True, slots=True)
class MoeLayerTrainReplacement:
    layer: int
    replaced_dense_forward_refs: tuple[str, str, str]
    removed_dense_parameter_state_refs: tuple[str, ...]
    retained_dense_residual_ref: str
    moe_forward_refs_by_step: tuple[tuple[str, ...], tuple[str, ...]]
    moe_backward_refs_by_step: tuple[tuple[str, ...], tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class MoeTrainParameterRequirement:
    step: int
    layer: int
    expert: int | None
    source_parameter_ref: str
    source_state_ref: str
    owner_rank: int
    wgrad_operation_ref: str
    sync_operation_ref: str
    sgd_operation_ref: str
    store_operation_ref: str
    zero_token_expert: bool
    read_version: int
    write_version: int


@dataclass(frozen=True, slots=True)
class MoeSharedParameterVersionRequirement:
    dense_parameter_state_ref: str
    dense_source_tensor_ref: str
    moe_source_parameter_name: str
    owner_rank: int
    source_e2e_state_versions: tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class FullMoeSharedTrainRequirements:
    """An independent exclusion/bijection oracle; never a runtime receipt."""

    dense_plan_id: str
    dense_plan_digest: str
    dense_source_requirements_digest: str
    moe_sequence_id: str
    moe_sequence_digest: str
    source_case_id: str
    layer_replacements: tuple[MoeLayerTrainReplacement, ...]
    shared_dense_parameter_state_refs: tuple[str, ...]
    shared_parameter_version_bindings: tuple[MoeSharedParameterVersionRequirement, ...]
    shared_dense_gradient_paths: tuple[tuple[int, str, int], ...]
    moe_parameter_requirements: tuple[MoeTrainParameterRequirement, ...]
    source_ir0_replacement_materialized: bool

    @property
    def excluded_dense_parameter_state_refs(self) -> tuple[str, ...]:
        return tuple(sorted(ref for layer in self.layer_replacements
                            for ref in layer.removed_dense_parameter_state_refs))

    def validate_against(self, dense_plan: FlexibleDenseTrainPlan,
                         dense: DenseFullTrainRequirements,
                         moe: MoeCompileSequence) -> None:
        expected = build_full_moe_shared_train_requirements(dense_plan, dense, moe)
        if self != expected:
            raise SchemaError("MoE source replacement or parameter lineage drifted",
                              path="full_moe_shared_train_requirements")


def build_full_moe_shared_train_requirements(
    dense_plan: FlexibleDenseTrainPlan,
    dense: DenseFullTrainRequirements,
    moe: MoeCompileSequence,
) -> FullMoeSharedTrainRequirements:
    """Require real source model identity and one-to-one per-layer replacement."""
    dense.validate_against(dense_plan)
    moe.validate("source_full_moe_shared_train")
    model = dense_plan.source_experiment.model
    request = moe.materialization.request
    other = request.model
    training = request.steps.training
    optimizer = request.optimizer
    if (request.family is not WorkloadFamily.MOE_TRAINING
            or training is None or training.step_count != dense.steps
            or training.sequence_length !=
                dense_plan.source_experiment.workload.train.seq_len
            or optimizer is None or optimizer.kind is not WorkloadOptimizerKind.SGD
            or (other.vocabulary_size, other.hidden_size,
                other.intermediate_size, other.num_layers,
                other.num_attention_heads, other.num_kv_heads,
                other.head_dim, other.max_sequence_length, other.dtype)
            != (model.V, model.H, model.I, model.L, model.NH, model.KVH,
                model.DH, model.max_position_embeddings, DType.FP16)
            or (request.parallel.tp, request.parallel.dp, request.parallel.pp,
                request.parallel.ep) != (1, 1, 1, 2)
            or request.mesh.rank_count != 2 or dense_plan.spec.mesh.rank_count != 1
            or any(unit.source_rank_policy != "rank0_shared_spine"
                   for unit in moe.units)):
        raise SchemaError("Dense source identity/sequence or MoE TRAIN model/mesh differs",
                          path="full_moe_shared_train_source")
    source_nodes = {node.id: node for node in dense_plan.forward_graph.nodes}
    expected_units = {(step, layer) for step in range(dense.steps)
                      for layer in range(model.L)}
    actual_units = {(unit.step, unit.layer): unit for unit in moe.units}
    if set(actual_units) != expected_units or len(moe.units) != len(actual_units):
        raise SchemaError("one real MoE block is required for each step/layer",
                          path="full_moe_shared_train_source.units")
    replacements, parameters = [], []
    excluded = set()
    for layer in range(model.L):
        prefix = f"T0.layer{layer}."
        old = tuple(prefix + suffix for suffix in ("gate_up", "swiglu", "down"))
        residual = prefix + "residual2"
        if not set((*old, residual)) <= set(source_nodes):
            raise SchemaError("Dense source MLP or residual successor is absent",
                              path=f"full_moe_shared_train_source.layer{layer}")
        omitted = tuple(sorted(template.state_ref for template
                               in dense_plan.parameter_templates
                               if set(template.forward_consumer_refs) & set(old)))
        if len(omitted) != 2 or any(ref in excluded for ref in omitted):
            raise SchemaError("replaced Dense MLP must remove exactly two owned weight shards",
                              path=f"full_moe_shared_train_source.layer{layer}")
        excluded.update(omitted)
        forwards, backwards = [], []
        for step in range(dense.steps):
            unit = actual_units[(step, layer)]
            binding = unit.operation_binding
            forwards.append((binding.router_operation_ref,
                             binding.route_freeze_operation_ref,
                             binding.dispatch_operation_ref,
                             *binding.expert_forward_operation_refs,
                             binding.combine_operation_ref))
            backwards.append((binding.grad_dispatch_operation_ref,
                              *binding.expert_backward_operation_refs,
                              binding.dx_combine_operation_ref))
            if any(ref is None for ref in backwards[-1]):
                raise SchemaError("MoE expert backward source is incomplete",
                                  path=f"full_moe_shared_train_source.layer{layer}.step{step}")
            if len(unit.parameter_bindings) != other.num_experts + 1:
                raise SchemaError("router and every expert need trainable parameter bindings",
                                  path=f"full_moe_shared_train_source.layer{layer}.step{step}")
            for group in unit.parameter_bindings:
                count = 1 if group.expert is None else 3
                if (len(group.parameter_refs) != count
                        or len(group.production_parameter_state_refs) !=
                           (2 if group.expert is None else 1)):
                    raise SchemaError("router replicated gate or expert fused homes absent",
                                      path=f"full_moe_shared_train_source.layer{layer}.step{step}")
                state_abis = {abi.state_ref: abi for fragment
                              in unit.linked_manifest.fragments for abi
                              in fragment.state_abi if abi.state_ref
                              in group.production_parameter_state_refs}
                if (set(state_abis) != set(group.production_parameter_state_refs)
                        or {abi.die_id for abi in state_abis.values()} !=
                           ({0, 1} if group.expert is None else {group.expert})):
                    raise SchemaError("gate replicas/expert state HBM ownership differ",
                                      path=f"full_moe_shared_train_source.layer{layer}.step{step}")
                zero = group.expert is not None and (
                    unit.spec.trace.expert_histogram[group.expert] == 0)
                for state_ref in group.production_parameter_state_refs:
                    owner = state_abis[state_ref].die_id
                    for index, parameter_ref in enumerate(group.parameter_refs):
                        parameters.append(MoeTrainParameterRequirement(
                            step, layer, group.expert, parameter_ref, state_ref,
                            owner, group.gradient_operation_refs[index],
                            group.sync_operation_refs[index],
                            group.sgd_operation_refs[index],
                            group.store_operation_refs[index], zero, step, step + 1,
                        ))
        replacements.append(MoeLayerTrainReplacement(
            layer, old, omitted, residual, tuple(forwards), tuple(backwards),
        ))
    shared = tuple(sorted(set(template.state_ref for template
                              in dense_plan.parameter_templates) - excluded))
    versions = {(state.logical_name, state.version): state for state
                in moe.materialization.logical_graph.state_versions
                if state.kind is E2EStateKind.PARAMETER}
    if len(versions) != len([state for state in
                             moe.materialization.logical_graph.state_versions
                             if state.kind is E2EStateKind.PARAMETER]):
        raise SchemaError("MoE source has ambiguous parameter versions",
                          path="full_moe_shared_train_source.versions")
    semantic_names = {
        "T0.tok_embeddings.weight": "embedding.weight",
        "T0.final_norm.weight": "final_norm.weight",
        "T0.lm_head.weight": "lm_head.weight",
        **{f"T0.layer{layer}.{legacy}": f"layer.{layer}.{moe_name}"
           for layer in range(model.L)
           for legacy, moe_name in (
               ("w_norm1", "input_norm.weight"),
               ("w_qkv", "qkv.weight"),
               ("w_o", "attention_out.weight"),
               ("w_norm2", "post_norm.weight"),
           )},
    }
    shared_bindings = []
    for template in dense_plan.parameter_templates:
        if template.state_ref not in shared:
            continue
        if template.tensor_ref not in semantic_names:
            raise SchemaError("shared source parameter has no exact Dense↔MoE identity",
                              path=f"full_moe_shared_train_source[{template.tensor_ref}]")
        name = semantic_names[template.tensor_ref]
        refs = tuple(versions[(name, index)] for index in range(3)
                     if (name, index) in versions)
        if (len(refs) != 3 or refs[0].producer_op_id is not None
                or any(refs[index].producer_op_id is None
                       for index in (1, 2))):
            raise SchemaError("shared source parameter lacks explicit two-step 0→1→2 lineage",
                              path=f"full_moe_shared_train_source[{template.tensor_ref}]")
        shared_bindings.append(MoeSharedParameterVersionRequirement(
            template.state_ref, template.tensor_ref, name, 0,
            tuple(item.id for item in refs),
        ))
    paths = tuple(sorted((path.step, path.parameter_state_ref, path.rank)
                         for path in dense.paths
                         if path.parameter_state_ref in shared))
    if (len(shared) != len(dense_plan.parameter_templates) - 2 * model.L
            or {binding.dense_parameter_state_ref for binding
                in shared_bindings} != set(shared)
            or len(paths) != dense.steps * len(shared)
            or len(parameters) != dense.steps * model.L *
                (2 + 3 * other.num_experts)
            or len({(entry.step, entry.layer, entry.source_parameter_ref,
                     entry.owner_rank)
                    for entry in parameters}) != len(parameters)):
        raise SchemaError("Dense/MoE parameter replacement/source gradient bijection absent",
                          path="full_moe_shared_train_source.parameters")
    return FullMoeSharedTrainRequirements(
        dense_plan.id, canonical_digest(dense_plan), canonical_digest(dense),
        moe.id, canonical_digest(moe), request.case_id,
        tuple(replacements), shared,
        tuple(sorted(shared_bindings, key=lambda binding:
                     binding.dense_parameter_state_ref)),
        paths, tuple(parameters),
        source_ir0_replacement_materialized=False,
    )


__all__ = ["MoeLayerTrainReplacement", "MoeTrainParameterRequirement",
           "MoeSharedParameterVersionRequirement",
           "FullMoeSharedTrainRequirements",
           "build_full_moe_shared_train_requirements"]
