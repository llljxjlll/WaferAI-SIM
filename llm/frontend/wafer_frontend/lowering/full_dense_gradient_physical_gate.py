"""Exact per-source-state gradient producer→sync→SGD→STORE physical gate.

The independent source oracle names every TP shard/DP owner/step and its
forward/backward producers.  No StateABI or operation label may be inferred
by substring or converted from an AdamW E2E carrier count.  This gate fails
closed while the Dense native reverse source/physical chain is incomplete.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Mapping

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    BufferABI, LinkedProgramManifest, RecordOpcode, SemanticOperandId,
    StateABI,
)
from ..schema.common import DType
from ..schema.flexible_dense_train import FlexibleDenseTrainPlan
from ..schema.full_dense_gradient_requirements import (
    DenseFullTrainRequirements, DenseRequiredGradientPath,
    DenseGradientDPReduction, DenseGradientLossObjective,
)
from ..schema.full_training_physical_dag import FullTrainingPhysicalDAG
from ..schema.ir0 import CollectiveKind, OpKind, ResidualWorkload, SwiGluWorkload
from ..schema.n6 import _leaf_fragments
from ..schema.persistent_state import (
    PersistentStateAccess, PersistentStateDecl, PersistentStateIdentity,
    PersistentStateLifetime, StateKind,
)


_DENSE_DX_UPSTREAM_OUTPUTS = {
    RecordOpcode.CROSS_ENTROPY_BACKWARD: (
        SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
    ),
    RecordOpcode.GEMM_DX_TIMING: (
        SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
    ),
    RecordOpcode.SWIGLU_BACKWARD_TIMING: (
        SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
    ),
    RecordOpcode.RMSNORM_BACKWARD_TIMING: (
        SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
    ),
    RecordOpcode.ATTENTION_BACKWARD_TIMING: (
        SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
    ),
    RecordOpcode.ROPE_BACKWARD_TIMING: (
        SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
    ),
    RecordOpcode.RESIDUAL_BACKWARD_TIMING: (
        SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
        SemanticOperandId.COMPUTE_AUX_ADDRESS,
    ),
    RecordOpcode.LOCAL_REDUCE: (
        SemanticOperandId.DESTINATION_ADDRESS,
    ),
}


def dense_dx_upstream_output_operands(
    opcode: RecordOpcode,
) -> tuple[SemanticOperandId, ...]:
    """Return the native output closures that may feed a source GEMM dX."""
    outputs = _DENSE_DX_UPSTREAM_OUTPUTS.get(opcode)
    if outputs is None:
        raise SchemaError("opcode is not a native Dense derivative producer",
                          path="dense_dx_upstream_opcode")
    return outputs


def require_exact_dense_parameter_state_inventory(
    manifest: LinkedProgramManifest,
    plan: FlexibleDenseTrainPlan,
    requirements: DenseFullTrainRequirements,
) -> Mapping[tuple[str, int], StateABI]:
    """Prove real per-shard source StateDecl IDs, owners and FP16 extent."""
    requirements.validate_against(plan)
    manifest.validate("full_dense_gradient_parameter_source")
    templates = {template.state_ref: template for template
                 in plan.parameter_templates}
    if set(templates) != {path.parameter_state_ref for path in
                         requirements.paths}:
        raise SchemaError("source parameter templates and gradient oracle lack a bijection",
                          path="dense_parameter_source")
    expected = {(path.parameter_state_ref, path.rank)
                for path in requirements.paths}
    declarations = {decl.id: decl for decl in
                    plan.forward_graph.persistent_states}
    trainable_to_source = {}
    for source_id, declaration in declarations.items():
        identity = PersistentStateIdentity.create(
            kind=StateKind.TRAINABLE_PARAMETER,
            instance_ref=declaration.identity.instance_ref,
            mesh_ref=declaration.identity.mesh_ref,
            request_ref=None, layer_index=None,
            tensor_ref=declaration.identity.tensor_ref,
            shard_index=declaration.identity.shard_index,
            generation=declaration.identity.generation,
        )
        trainable = PersistentStateDecl.create(
            identity=identity, shape=declaration.shape, dtype=declaration.dtype,
            layout=declaration.layout,
            lifetime=PersistentStateLifetime.PERSISTENT,
            access=PersistentStateAccess.READ_WRITE,
        )
        trainable_to_source[trainable.id] = source_id
    actual = {}
    for fragment in _leaf_fragments(manifest.fragments):
        for abi in fragment.state_abi:
            source_id = trainable_to_source.get(abi.state_ref, abi.state_ref)
            if source_id not in templates:
                continue
            key = (source_id, abi.die_id)
            previous = actual.setdefault(key, abi)
            if previous != abi:
                raise SchemaError("source StateDecl shard changes physical HBM ABI across phases",
                                  path=f"dense_parameter_source[{key}]")
    if set(actual) != expected:
        raise SchemaError("source StateDecl TP/DP owner↔StateABI exact bijection absent",
                          path="dense_parameter_source")
    for key, abi in actual.items():
        template = templates[key[0]]
        if (key[1] not in template.owner_ranks
                or key[1] % plan.spec.tp_degree != template.tp_shard_index
                or abi.kind is not StateKind.TRAINABLE_PARAMETER
                or abi.access is not PersistentStateAccess.READ_WRITE
                or abi.dtype is not DType.FP16
                or abi.shape != declarations[key[0]].shape
                or abi.size_bytes != template.weight_bytes):
            raise SchemaError("source parameter lacks owned physical TRAINABLE_PARAMETER home",
                              path=f"dense_parameter_source[{key}]")
    return actual


def require_source_gemm_wgrad_geometry(
    plan: FlexibleDenseTrainPlan,
    path: DenseRequiredGradientPath,
    *,
    m: int, n: int, k: int,
) -> None:
    """WGRAD (M,N,K) outputs source shard M×N, K is forward rank rows."""
    states = {decl.id: decl for decl in plan.forward_graph.persistent_states}
    nodes = {node.id: node for node in plan.forward_graph.nodes}
    state = states[path.parameter_state_ref]
    forward = nodes[path.forward_op_refs[0]]
    if (forward.kind is not OpKind.GEMM or len(state.shape) != 2
            or type(m) is not int or type(n) is not int or type(k) is not int
            or min(m, n, k) < 1 or state.shape != (m, n)
            or k != forward.workload.rank_shape[0]
            or forward.workload.rank_shape[1:] != (n, m)):
        raise SchemaError("native WGRAD M×N and K differ from source weight shard and rows",
                          path=f"gradient_path[{path.parameter_state_ref}].geometry")


def require_named_gemm_wgrad_typed_buffers(
    *, rank: int, m: int, n: int, k: int,
    activation: BufferABI, upstream: BufferABI, gradient: BufferABI,
    literals: Mapping[str, int], path: str,
) -> None:
    """Prove the three native 0x25 SRAM values and their FP32 gradient extent."""
    if (type(m) is not int or type(n) is not int or type(k) is not int
            or min(m, n, k) < 1
            or any(literals.get(field) != value for field, value in (
                ("m", m), ("n", n), ("k", k),
                ("activation_datatype", 1), ("upstream_datatype", 1),
                ("gradient_datatype", 3),
            ))):
        raise SchemaError("native GEMM FP32 WGRAD dimensions or datatypes differ",
                          path=path)
    for role, abi, dtype, size in (
        ("activation", activation, DType.FP16, 2 * k * m),
        ("upstream", upstream, DType.FP16, 2 * k * n),
        ("gradient", gradient, DType.FP32, 4 * m * n),
    ):
        if (abi.dtype is not dtype or abi.size_bytes != size
                or abi.logical_core.die_id != rank):
            raise SchemaError(
                f"native {role} physical footprint/dtype differs from GEMM FP32 WGRAD",
                path=f"{path}.{role}",
            )
    if len({activation.storage_id, upstream.storage_id, gradient.storage_id}) != 3:
        raise SchemaError("GEMM FP32 WGRAD operands require independent SRAM storage",
                          path=path)


def require_source_gemm_dx_geometry(
    plan: FlexibleDenseTrainPlan, *, source_forward_ref: str,
    source_parameter_state_ref: str, m: int, n: int, k: int,
) -> None:
    """DGRAD dX[K,M] uses the actual forward FP16 W[M,N] shard and rows K."""
    states = {decl.id: decl for decl in plan.forward_graph.persistent_states}
    nodes = {node.id: node for node in plan.forward_graph.nodes}
    state = states.get(source_parameter_state_ref)
    forward = nodes.get(source_forward_ref)
    if (state is None or forward is None or forward.kind is not OpKind.GEMM
            or len(state.shape) != 2
            or any(type(value) is not int or value < 1
                   for value in (m, n, k))
            or state.shape != (m, n)
            or forward.workload.rank_shape != (k, n, m)):
        raise SchemaError("native dX K×M, FP16 W M×N and source GEMM rows disagree",
                          path=f"gemm_dx_source[{source_forward_ref}]")


def require_named_gemm_dx_typed_buffers(
    *, rank: int, m: int, n: int, k: int,
    weight: BufferABI, upstream: BufferABI, dx: BufferABI,
    literals: Mapping[str, int], path: str,
) -> None:
    """Require the public FP16 activation-gradient GEMM dX contract."""
    if (type(m) is not int or type(n) is not int or type(k) is not int
            or min(m, n, k) < 1
            or any(literals.get(field) != value for field, value in (
                ("m", m), ("n", n), ("k", k),
                ("weight_datatype", 1), ("upstream_datatype", 1),
                ("dx_datatype", 1),
            ))):
        raise SchemaError("native GEMM FP16 dX dimensions or datatypes differ",
                          path=path)
    for role, abi, dtype, size in (
        ("weight", weight, DType.FP16, 2 * m * n),
        ("upstream", upstream, DType.FP16, 2 * k * n),
        ("dx", dx, DType.FP16, 2 * k * m),
    ):
        if (abi.dtype is not dtype or abi.size_bytes != size
                or abi.logical_core.die_id != rank):
            raise SchemaError(
                f"native {role} physical footprint/dtype differs from GEMM FP16 dX",
                path=f"{path}.{role}",
            )
    if len({weight.storage_id, upstream.storage_id, dx.storage_id}) != 3:
        raise SchemaError("GEMM FP16 dX operands require independent SRAM storage",
                          path=path)


def require_named_gemm_dx_state_load(
    manifest: LinkedProgramManifest, *, state_ref: str, rank: int,
    compute_fragment_id: str, compute_record_index: int,
    weight: BufferABI, required_state: StateABI | None = None,
    allowed_load_records: frozenset[tuple[str, int]] | None = None,
) -> tuple[str, int]:
    """Require one earlier blocking HBM READ loading the exact weight SRAM value."""
    states = {abi.id: abi for fragment in _leaf_fragments(manifest.fragments)
              for abi in fragment.state_abi}
    buffers = {abi.id: abi for fragment in _leaf_fragments(manifest.fragments)
               for abi in fragment.buffer_abi}
    by_fragment = {fragment.id: fragment for fragment in _leaf_fragments(manifest.fragments)}
    load_candidates: list[tuple[str, int]] = []
    for binding in manifest.state_operand_bindings:
        if binding.operand_id is not SemanticOperandId.HBM_ADDRESS:
            continue
        state = states[binding.state_abi_id]
        if state.state_ref != state_ref or state.die_id != rank:
            continue
        if (allowed_load_records is not None and
                (binding.fragment_id, binding.fragment_record_index)
                not in allowed_load_records):
            continue
        if (required_state is not None and state != required_state
                or state.dtype is not DType.FP16
                or state.size_bytes != weight.size_bytes
                or state.access is PersistentStateAccess.RESERVED):
            raise SchemaError("named dX weight HBM StateABI READ differs from source",
                              path=f"gemm_dx_state[{state_ref}]")
        fragment = by_fragment[binding.fragment_id]
        stream = next((stream for stream in fragment.core_streams
                       if stream.logical_core == binding.logical_core), None)
        if stream is None or binding.fragment_record_index >= len(stream.records):
            raise SchemaError("named dX StateABI LOAD record is dangling",
                              path=f"gemm_dx_state[{state_ref}]")
        record = stream.records[binding.fragment_record_index]
        if record.opcode is not RecordOpcode.LSU_LOAD:
            continue
        local = next((closure for closure in manifest.address_operand_bindings
                      if closure.fragment_id == binding.fragment_id
                      and closure.logical_core == binding.logical_core
                      and closure.fragment_record_index == binding.fragment_record_index
                      and closure.operand_id is SemanticOperandId.DESTINATION_ADDRESS), None)
        loaded = (buffers[local.buffer_abi_ids[0]] if local is not None
                  and len(local.buffer_abi_ids) == 1 else None)
        size = next((item.literal_value for item in record.operands
                     if item.name == "size_bytes"), None)
        if loaded is None or loaded.id != weight.id:
            # One StateABI may have several independent forward/dX READs in
            # this step. Only the READ bound to this dX input is relevant.
            continue
        if (loaded.storage_id != weight.storage_id
                or loaded.value_id != weight.value_id
                or loaded.logical_core.die_id != rank
                or size != weight.size_bytes):
            raise SchemaError("HBM LSU_LOAD does not fill exact named dX FP16 weight SRAM",
                              path=f"gemm_dx_state[{state_ref}]")
        load_candidates.append((binding.fragment_id, binding.fragment_record_index))
    if len(load_candidates) != 1:
        raise SchemaError("named dX needs exactly one source StateABI HBM LOAD",
                          path=f"gemm_dx_state[{state_ref}]")
    core = weight.logical_core
    linked = next((stream for stream in manifest.core_streams
                   if stream.logical_core == core), None)
    if linked is None:
        raise SchemaError("named dX core stream is absent",
                          path=f"gemm_dx_state[{state_ref}]")
    refs = [(ref.fragment_id, ref.fragment_record_index)
            for ref in linked.records]
    if (load_candidates[0] not in refs
            or (compute_fragment_id, compute_record_index) not in refs
            or refs.index(load_candidates[0]) >=
               refs.index((compute_fragment_id, compute_record_index))):
        raise SchemaError("source HBM LOAD must precede named dX record",
                          path=f"gemm_dx_state[{state_ref}]")
    return load_candidates[0]


def require_full_dense_physical_gradient_paths(
    manifest: LinkedProgramManifest,
    plan: FlexibleDenseTrainPlan,
    requirements: DenseFullTrainRequirements,
    dag: FullTrainingPhysicalDAG,
    *,
    required_backward_opcodes: Mapping[str, RecordOpcode],
    required_wgrad_opcodes: Mapping[str, RecordOpcode],
) -> None:
    """Require every named native reverse producer and its actual SGD chain.

    `required_backward_opcodes` is an independent producer-owned versioned
    primitive contract for *all* required reverse operations.  A legacy
    MATMUL reused for attention/norm/backbone may not satisfy those contracts.
    Gradient BufferABI closures, exact source parameter state IDs, record
    dimensions, SGD input and state versions must agree on each physical die.
    """
    homes = require_exact_dense_parameter_state_inventory(
        manifest, plan, requirements,
    )
    if (requirements.loss_objective is not
            DenseGradientLossObjective.PER_ROW_CE_SUM
            or requirements.loss_gradient_seed_per_row != 1.0
            or requirements.dp_reduction is not
            DenseGradientDPReduction.FP32_RANK_MAJOR_SUM
            or requirements.optimizer_gradient_normalization):
        raise SchemaError("loss seed, physical DP SUM and unscaled SGD math contract differ",
                          path="full_dense_gradient_math_contract")
    source_nodes = {node.id: node for node in plan.forward_graph.nodes}
    inverse_collectives = {
        f"backward::{node.id}": (
            CollectiveKind.REDUCE_SCATTER if node.workload.collective is
            CollectiveKind.ALL_GATHER else CollectiveKind.ALL_GATHER
        )
        for node in source_nodes.values() if node.kind is OpKind.COLLECTIVE
    }
    if (set(inverse_collectives) - set(requirements.required_backbone_backward_refs)
            or set(required_backward_opcodes) !=
            set(requirements.required_backbone_backward_refs) - set(inverse_collectives)):
        raise SchemaError("every native reverse opcode or typed TP inverse collective must have an exact source",
                          path="required_backward_opcodes")
    if set(required_wgrad_opcodes) != {
            path.named_wgrad_op_ref for path in requirements.paths}:
        raise SchemaError("all named parameter derivatives need independent native opcode contract",
                          path="required_wgrad_opcodes")
    reverse_native = {
        OpKind.GEMM: getattr(RecordOpcode, "GEMM_DX_TIMING", None),
        OpKind.NORM: getattr(RecordOpcode, "RMSNORM_BACKWARD_TIMING", None),
        OpKind.ATTENTION: getattr(RecordOpcode, "ATTENTION_BACKWARD_TIMING", None),
        OpKind.ROPE: getattr(RecordOpcode, "ROPE_BACKWARD_TIMING", None),
        OpKind.EMBEDDING: getattr(RecordOpcode, "EMBEDDING_TABLE_WGRAD_TIMING", None),
    }
    for reverse_ref in requirements.required_backbone_backward_refs:
        if not reverse_ref.startswith("backward::") or (
            reverse_ref[len("backward::"):] not in source_nodes
        ):
            raise SchemaError("reverse source identity is not one validated forward node",
                              path=f"required_backward_opcodes[{reverse_ref}]")
        source = source_nodes[reverse_ref[len("backward::"):]]
        if reverse_ref in inverse_collectives:
            if source.kind is not OpKind.COLLECTIVE:
                raise SchemaError("typed inverse must originate in one TP collective",
                                  path=reverse_ref)
            continue
        if source.kind is OpKind.ELEMENTWISE:
            expected_opcode = (
                RecordOpcode.SWIGLU_BACKWARD_TIMING
                if type(source.workload) is SwiGluWorkload else
                getattr(RecordOpcode, "RESIDUAL_BACKWARD_TIMING", None)
                if type(source.workload) is ResidualWorkload else None
            )
        else:
            expected_opcode = reverse_native.get(source.kind)
        if (expected_opcode is None
                or required_backward_opcodes[reverse_ref] is not expected_opcode):
            raise SchemaError("reverse native opcode lacks real source derivative family",
                              path=f"required_backward_opcodes[{reverse_ref}]")
    for template in plan.parameter_templates:
        families = {source_nodes[ref].kind for ref in template.forward_consumer_refs}
        opcode = required_wgrad_opcodes[template.wgrad_ref]
        allowed = {
            OpKind.GEMM: RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING,
            OpKind.NORM: getattr(RecordOpcode, "NORM_GAMMA_WGRAD_TIMING", None),
            OpKind.EMBEDDING: getattr(RecordOpcode, "EMBEDDING_TABLE_WGRAD_TIMING", None),
        }
        if (len(families) != 1 or next(iter(families)) not in allowed
                or opcode is not allowed[next(iter(families))]):
            raise SchemaError("WGRAD opcode must implement the actual source derivative family",
                              path=f"required_wgrad_opcodes[{template.state_ref}]")
    dag.validate_against(
        _leaf_fragments(manifest.fragments), manifest.core_streams,
        required_operation_ids=tuple(sorted({
            *(path.named_wgrad_op_ref for path in requirements.paths),
            *(path.named_sync_op_ref for path in requirements.paths
              if len(path.dp_group_ranks) > 1),
            *(path.named_optimizer_op_ref for path in requirements.paths),
            *(path.named_store_op_ref for path in requirements.paths),
            *requirements.required_backbone_backward_refs,
        })),
    )
    fragments = {fragment.id: fragment for fragment in _leaf_fragments(manifest.fragments)}
    buffers = {abi.id: abi for fragment in _leaf_fragments(manifest.fragments)
               for abi in fragment.buffer_abi}
    states = {abi.id: abi for fragment in _leaf_fragments(manifest.fragments)
              for abi in fragment.state_abi}
    closures = {
        (entry.fragment_id, entry.logical_core, entry.fragment_record_index,
         entry.operand_id): entry
        for entry in manifest.address_operand_bindings
    }
    state_uses = {
        (entry.fragment_id, entry.logical_core, entry.fragment_record_index,
         entry.operand_id): entry
        for entry in manifest.state_operand_bindings
    }
    by_action = {item.id: item for item in dag.actions}
    named = defaultdict(list)
    for action in dag.actions:
        named[(action.step, action.logical_core.die_id,
               action.operation_ref)].append(action)

    def one(step: int, rank: int, ref: str, opcode: RecordOpcode):
        matches = [(action, fragment_id, index)
                   for action in named[(step, rank, ref)]
                   for fragment_id, index, physical_opcode
                   in action.executable_records if physical_opcode is opcode]
        if len(matches) != 1:
            raise SchemaError("one exact named native physical record is required",
                              path=f"gradient_path.step{step}.rank{rank}.{ref}.{opcode.name}")
        action, fragment_id, index = matches[0]
        fragment = fragments[fragment_id]
        local = next(stream for stream in fragment.core_streams
                     if stream.logical_core == action.logical_core)
        return action, local.records[index], fragment_id, index

    def buffer(action, fragment_id, index, operand: SemanticOperandId) -> BufferABI:
        closure = closures.get((fragment_id, action.logical_core, index, operand))
        if closure is None or len(closure.buffer_abi_ids) != 1:
            raise SchemaError("native producer/consumer lacks one exact physical BufferABI",
                              path=f"gradient_path[{action.operation_ref}].{operand.name}")
        return buffers[closure.buffer_abi_ids[0]]

    def same_physical_value(a: BufferABI, b: BufferABI) -> bool:
        return (a.id == b.id and a.storage_id == b.storage_id
                and a.binding_id == b.binding_id
                and a.value_id == b.value_id
                and a.logical_core == b.logical_core)

    def depends_on(later, earlier) -> bool:
        pending = list(later.depends_on)
        found = set()
        while pending:
            current = pending.pop()
            if current == earlier.id:
                return True
            if current in found:
                continue
            found.add(current)
            pending.extend(by_action[current].depends_on)
        return False

    # Native inverse collectives are explicit rank-local DTE/SUM programs,
    # not a fictional single derivative COMPUTE opcode.  The earlier source
    # DAG gate binds their IR1/N4/N5 origin, while this independent gradient
    # gate demands their physical executable records and exact peer edges.
    tp = plan.spec.tp_degree
    if inverse_collectives and tp < 2:
        raise SchemaError("singleton TP cannot satisfy a cross-die inverse collective",
                          path="inverse_collectives")
    for inverse_ref, inverse_kind in inverse_collectives.items():
        for step in range(requirements.steps):
            matched_edges = tuple((send, recv) for send, recv in dag.transport_edges
                                  if by_action[send].step == step
                                  and by_action[recv].step == step
                                  and by_action[send].operation_ref == inverse_ref
                                  and by_action[recv].operation_ref == inverse_ref)
            if len(matched_edges) != tp * (tp - 1):
                raise SchemaError("inverse collective misses exact peer physical D2D edges",
                                  path=f"{inverse_ref}.step{step}")
            for rank in range(tp):
                group = named[(step, rank, inverse_ref)]
                functional = Counter(
                    opcode for action in group
                    for _, _, opcode in action.executable_records
                    if opcode not in (RecordOpcode.SRAM_ALLOC_AT, RecordOpcode.SRAM_FREE)
                )
                if inverse_kind is CollectiveKind.REDUCE_SCATTER:
                    expected = Counter({RecordOpcode.DTE_ISSUE: 1,
                                        RecordOpcode.DTE_WAIT: tp,
                                        RecordOpcode.DTE_SEND: tp - 1,
                                        RecordOpcode.DTE_RECV: tp - 1,
                                        RecordOpcode.LOCAL_REDUCE: 1})
                else:
                    expected = Counter({RecordOpcode.DTE_ISSUE: 1,
                                        RecordOpcode.DTE_WAIT: 1,
                                        RecordOpcode.DTE_SEND: tp - 1,
                                        RecordOpcode.DTE_RECV: tp - 1,
                                        RecordOpcode.EVENT_SET: tp - 1 if rank == 0 else 1,
                                        RecordOpcode.EVENT_WAIT: tp - 1 if rank == 0 else 1})
                if functional != expected:
                    raise SchemaError(
                        "inverse collective lacks rank-local executable DTE/SUM records: "
                        f"actual={dict(functional)} expected={dict(expected)}",
                        path=f"{inverse_ref}.step{step}.rank{rank}",
                    )

    for path in requirements.paths:
        step, rank = path.step, path.rank
        for reverse_ref in path.backward_producer_refs:
            suffix = f"::{path.parameter_state_ref}"
            if not reverse_ref.endswith(suffix):
                raise SchemaError("parameter reverse reference must retain exact source StateDecl",
                                  path=f"gradient_path.step{step}.rank{rank}.{reverse_ref}")
            physical_ref = reverse_ref.removesuffix(suffix)
            if physical_ref == f"backward::T0.embedding":
                physical_ref = path.named_wgrad_op_ref
                opcode = required_wgrad_opcodes.get(physical_ref)
            else:
                opcode = required_backward_opcodes.get(physical_ref)
            if physical_ref in inverse_collectives:
                if opcode is not None:
                    raise SchemaError("TP inverse collective cannot substitute a compute opcode",
                                      path=f"gradient_path.step{step}.rank{rank}.{reverse_ref}")
                # The complete physical rank-local peer program was checked
                # above; the source gradient path keeps this named producer.
                continue
            if opcode is None:
                raise SchemaError("parameter reverse source lacks independent native opcode",
                                  path=f"gradient_path.step{step}.rank{rank}.{reverse_ref}")
            reverse, reverse_record, reverse_fid, reverse_idx = one(
                step, rank, physical_ref, opcode)
            if opcode is RecordOpcode.GEMM_DX_TIMING:
                source_ref = physical_ref.removeprefix("backward::")
                owners = [template.state_ref for template in
                          plan.parameter_templates if
                          source_ref in template.forward_consumer_refs
                          and template.tp_shard_index == rank % plan.spec.tp_degree
                          and rank in template.owner_ranks]
                if len(owners) != 1:
                    raise SchemaError("named dX source GEMM lacks one real weight StateDecl",
                                      path=f"gradient_path[{reverse_ref}].state")
                literals = {item.name: item.literal_value for item in
                            reverse_record.operands if item.literal_value is not None}
                if not all(type(literals.get(field)) is int and
                           literals[field] > 0 for field in ("m", "n", "k")):
                    raise SchemaError("named dX source tile is absent",
                                      path=f"gradient_path[{reverse_ref}].geometry")
                require_source_gemm_dx_geometry(
                    plan, source_forward_ref=source_ref,
                    source_parameter_state_ref=owners[0],
                    m=literals["m"], n=literals["n"], k=literals["k"],
                )
                weight = buffer(reverse, reverse_fid, reverse_idx,
                                SemanticOperandId.COMPUTE_INPUT_ADDRESS)
                upstream = buffer(reverse, reverse_fid, reverse_idx,
                                  SemanticOperandId.COMPUTE_DATA_ADDRESS)
                dx = buffer(reverse, reverse_fid, reverse_idx,
                            SemanticOperandId.COMPUTE_OUTPUT_ADDRESS)
                require_named_gemm_dx_typed_buffers(
                    rank=rank, m=literals["m"], n=literals["n"],
                    k=literals["k"], weight=weight, upstream=upstream,
                    dx=dx, literals=literals,
                    path=f"gradient_path[{reverse_ref}].dX",
                )
                step_load_records = frozenset(
                    (fid, idx) for candidate in dag.actions
                    if candidate.step == step and
                    candidate.logical_core == reverse.logical_core
                    for fid, idx, op in candidate.executable_records
                    if op is RecordOpcode.LSU_LOAD
                )
                load_fid, load_idx = require_named_gemm_dx_state_load(
                    manifest, state_ref=homes[(owners[0], rank)].state_ref,
                    rank=rank,
                    compute_fragment_id=reverse_fid,
                    compute_record_index=reverse_idx, weight=weight,
                    required_state=homes[(owners[0], rank)],
                    allowed_load_records=step_load_records,
                )
                source_loads = [item for item in dag.actions
                                if item.step == step and
                                item.logical_core == reverse.logical_core and
                                (load_fid, load_idx, RecordOpcode.LSU_LOAD)
                                in item.executable_records]
                if len(source_loads) != 1 or not depends_on(reverse, source_loads[0]):
                    raise SchemaError("named dX must depend on its physical parameter LOAD",
                                      path=f"gradient_path[{reverse_ref}].state")
                upstream_producers = []
                for earlier in dag.actions:
                    if earlier.step != step or earlier.logical_core != reverse.logical_core:
                        continue
                    for producer_fid, producer_idx, producer_opcode in earlier.executable_records:
                        output_operands = _DENSE_DX_UPSTREAM_OUTPUTS.get(
                            producer_opcode
                        )
                        if output_operands is None:
                            continue
                        if any(
                            (closure := closures.get((
                                producer_fid, earlier.logical_core,
                                producer_idx, operand,
                            ))) is not None
                            and len(closure.buffer_abi_ids) == 1
                            and same_physical_value(
                                buffers[closure.buffer_abi_ids[0]], upstream)
                            for operand in output_operands
                        ) and depends_on(reverse, earlier):
                            upstream_producers.append(earlier)
                if len(upstream_producers) != 1:
                    # Inverse TP AllGather produces the full dY by one local
                    # copy and exactly one receive from each peer. All these
                    # writes bind the *same* physical output BufferABI and a
                    # subsequent owner/peer barrier gates the GEMM dX.
                    source_ref = upstream.value_id.removesuffix(
                        f".input_gradient::step{step}"
                    )
                    collective_ref = source_ref if source_ref in inverse_collectives else None
                    if (collective_ref is None
                            or inverse_collectives[collective_ref] is not
                            CollectiveKind.ALL_GATHER
                            or upstream.value_id !=
                                f"{collective_ref}.input_gradient::step{step}"
                            or upstream_producers):
                        raise SchemaError(
                            "named dX upstream dY lacks one real earlier derivative producer",
                            path=f"gradient_path[{reverse_ref}].dY",
                        )
                    collective_actions = named[(step, rank, collective_ref)]
                    writers = []
                    barriers = []
                    for candidate in collective_actions:
                        if any(op in (RecordOpcode.EVENT_SET, RecordOpcode.EVENT_WAIT)
                               for _, _, op in candidate.executable_records):
                            barriers.append(candidate)
                        for candidate_fid, candidate_idx, candidate_op in candidate.executable_records:
                            if candidate_op not in (
                                RecordOpcode.DTE_ISSUE, RecordOpcode.DTE_RECV,
                            ):
                                continue
                            dest = buffer(candidate, candidate_fid, candidate_idx,
                                          SemanticOperandId.DESTINATION_ADDRESS)
                            if same_physical_value(dest, upstream):
                                writers.append((candidate, candidate_op))
                    counts = Counter(op for _, op in writers)
                    if (counts != Counter({RecordOpcode.DTE_ISSUE: 1,
                                          RecordOpcode.DTE_RECV: tp - 1})
                            or len(barriers) != 1
                            or not depends_on(reverse, barriers[0])
                            or any(not depends_on(barriers[0], writer)
                                   for writer, _ in writers)):
                        raise SchemaError(
                            "TP inverse AllGather did not assemble source dY before named dX",
                            path=f"gradient_path[{reverse_ref}].dY",
                        )
        derivative_opcode = required_wgrad_opcodes[path.named_wgrad_op_ref]
        wgrad, native, fid, idx = one(step, rank, path.named_wgrad_op_ref,
                                       derivative_opcode)
        produced = buffer(wgrad, fid, idx,
                          SemanticOperandId.COMPUTE_OUTPUT_ADDRESS)
        literals = {item.name: item.literal_value for item in native.operands
                    if item.literal_value is not None}
        if (produced.dtype is not DType.FP32
                or produced.size_bytes != path.gradient_bytes
                or produced.logical_core.die_id != rank
                or literals.get("gradient_datatype") != 3):
            raise SchemaError("WGRAD FP32 physical output/dimensions differ from source parameter",
                              path=f"gradient_path[{path.parameter_state_ref}].wgrad")
        if derivative_opcode is RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING and (
            not all(type(literals.get(field)) is int and literals[field] > 0
                    for field in ("m", "k", "n"))
        ):
            raise SchemaError("native GEMM derivative shape differs from exact source parameter",
                              path=f"gradient_path[{path.parameter_state_ref}].wgrad")
        if derivative_opcode is RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING:
            require_source_gemm_wgrad_geometry(
                plan, path, m=literals["m"], n=literals["n"],
                k=literals["k"],
            )
            require_named_gemm_wgrad_typed_buffers(
                rank=rank, m=literals["m"], n=literals["n"],
                k=literals["k"],
                activation=buffer(wgrad, fid, idx,
                                  SemanticOperandId.COMPUTE_INPUT_ADDRESS),
                upstream=buffer(wgrad, fid, idx,
                                SemanticOperandId.COMPUTE_DATA_ADDRESS),
                gradient=produced, literals=literals,
                path=f"gradient_path[{path.parameter_state_ref}].wgrad",
            )
        if len(path.dp_group_ranks) > 1:
            waves = tuple(wave for wave in plan.gradient_waves
                          if wave.state_ref == path.parameter_state_ref)
            if not waves:
                raise SchemaError("source parameter has no DP gradient tree",
                                  path=f"gradient_path[{path.parameter_state_ref}].dp_sync")
            wave_payloads = {}
            for wave in waves:
                for transfer in wave.transfers:
                    source, destination = transfer.source_rank, transfer.destination_rank
                    edge_matches = [
                        (by_action[src], by_action[dst])
                        for src, dst in dag.transport_edges
                        if by_action[src].step == step and by_action[dst].step == step
                        and by_action[src].logical_core.die_id == source
                        and by_action[dst].logical_core.die_id == destination
                        and by_action[src].operation_ref == path.named_sync_op_ref
                        and by_action[dst].operation_ref == path.named_sync_op_ref
                    ]
                    if len(edge_matches) != 1:
                        raise SchemaError("source P2 DP wave lacks one physical SEND→RECV edge",
                                          path=f"gradient_path[{path.parameter_state_ref}].wave{wave.index}")
                    sending, receiving = edge_matches[0]
                    sends = [(fid, idx) for fid, idx, opcode
                             in sending.executable_records
                             if opcode is RecordOpcode.DTE_SEND]
                    receives = [(fid, idx) for fid, idx, opcode
                                in receiving.executable_records
                                if opcode is RecordOpcode.DTE_RECV]
                    if len(sends) != 1 or len(receives) != 1:
                        raise SchemaError("DP physical wave requires one SEND and one RECV",
                                          path=f"gradient_path[{path.parameter_state_ref}].wave{wave.index}")
                    sfid, sidx = sends[0]
                    rfid, ridx = receives[0]
                    sender = next(stream for stream in fragments[sfid].core_streams
                                  if stream.logical_core == sending.logical_core).records[sidx]
                    receiver = next(stream for stream in fragments[rfid].core_streams
                                    if stream.logical_core == receiving.logical_core).records[ridx]
                    send_buffer = buffer(sending, sfid, sidx,
                                         SemanticOperandId.SOURCE_ADDRESS)
                    recv_buffer = buffer(receiving, rfid, ridx,
                                         SemanticOperandId.DESTINATION_ADDRESS)
                    if any(
                        next((operand.literal_value for operand in record.operands
                              if operand.name == "length_bytes"), None) !=
                            transfer.logical_bytes
                        for record in (sender, receiver)
                    ) or any(abi.dtype is not DType.FP32
                             or abi.size_bytes != transfer.logical_bytes
                             for abi in (send_buffer, recv_buffer)):
                        raise SchemaError("DP P2 wave lacks exact physical FP32 gradient payload",
                                          path=f"gradient_path[{path.parameter_state_ref}].wave{wave.index}")
                    wave_payloads[wave.index] = (
                        sending, receiving, send_buffer, recv_buffer,
                    )
            if len(path.dp_group_ranks) != 2:
                raise SchemaError("multi-rank DP tree needs complete rank-major native source reduction",
                                  path=f"gradient_path[{path.parameter_state_ref}].dp_sync")
            root = path.dp_group_ranks[0]
            reduction, reduced, rid, rindex = one(
                step, root, path.named_sync_op_ref, RecordOpcode.LOCAL_REDUCE,
            )
            reduction_source = buffer(reduction, rid, rindex,
                                      SemanticOperandId.SOURCE_ADDRESS)
            reduction_result = buffer(reduction, rid, rindex,
                                      SemanticOperandId.DESTINATION_ADDRESS)
            rlits = {item.name: item.literal_value for item in reduced.operands
                     if item.literal_value is not None}
            if (rlits.get("input_dtype") != 1 or rlits.get("accumulator_dtype") != 1
                    or rlits.get("output_dtype") != 1 or rlits.get("reduce_op") != 1
                    or rlits.get("input_count") != 2
                    or rlits.get("element_count") != path.gradient_bytes // 4
                    or rlits.get("input_stride_bytes") != path.gradient_bytes
                    or reduction_source.dtype is not DType.FP32
                    or reduction_source.size_bytes != 2 * path.gradient_bytes
                    or reduction_result.dtype is not DType.FP32
                    or reduction_result.size_bytes != path.gradient_bytes):
                raise SchemaError("DP root native rank-major FP32 SUM contract/bytes differ",
                                  path=f"gradient_path[{path.parameter_state_ref}].dp_reduce")
            incoming = [by_action[dst] for src, dst in dag.transport_edges
                        if by_action[src].step == step
                        and by_action[src].operation_ref == path.named_sync_op_ref
                        and by_action[src].logical_core.die_id != root
                        and by_action[dst].logical_core.die_id == root
                        and by_action[dst].operation_ref == path.named_sync_op_ref]
            if len(incoming) != 1 or not depends_on(reduction, incoming[0]):
                raise SchemaError("DP FP32 SUM cannot run before all real RECV payloads",
                                  path=f"gradient_path[{path.parameter_state_ref}].dp_reduce")
            first, second = (wave_payloads[wave.index] for wave in waves)
            reduced_recv = first[3]
            broadcast_source = second[2]
            if (reduced_recv.logical_core.die_id != root
                    or reduced_recv.region_ref != reduction_source.region_ref
                    or reduced_recv.region_offset_bytes !=
                        reduction_source.region_offset_bytes + path.gradient_bytes
                    or broadcast_source.logical_core.die_id != root
                    or not same_physical_value(broadcast_source, reduction_result)):
                raise SchemaError("DP transport fails true rank-major input or reduced-output broadcast binding",
                                  path=f"gradient_path[{path.parameter_state_ref}].dp_payload")
            root_copy_actions = named[(step, root, path.named_sync_op_ref)]
            root_copies = [
                (action, fid, idx, opcode) for action in root_copy_actions
                for fid, idx, opcode in action.executable_records
                if opcode is RecordOpcode.DTE_ISSUE
            ]
            if len(root_copies) != 1:
                raise SchemaError("DP root local rank-major input needs one real WGRAD copy",
                                  path=f"gradient_path[{path.parameter_state_ref}].dp_payload")
            local_action, local_fid, local_idx, _ = root_copies[0]
            copy_source = buffer(local_action, local_fid, local_idx,
                                 SemanticOperandId.SOURCE_ADDRESS)
            copy_dest = buffer(local_action, local_fid, local_idx,
                               SemanticOperandId.DESTINATION_ADDRESS)
            root_wgrad_action, _, root_fid, root_idx = one(
                step, root, path.named_wgrad_op_ref, derivative_opcode,
            )
            root_gradient = buffer(root_wgrad_action, root_fid, root_idx,
                                   SemanticOperandId.COMPUTE_OUTPUT_ADDRESS)
            if (not same_physical_value(copy_source, root_gradient)
                    or copy_dest.region_ref != reduction_source.region_ref
                    or copy_dest.region_offset_bytes !=
                        reduction_source.region_offset_bytes
                    or copy_dest.size_bytes != path.gradient_bytes
                    or not depends_on(local_action, root_wgrad_action)
                    or not depends_on(reduction, local_action)):
                raise SchemaError("DP root local gradient bytes do not populate first rank-major slice",
                                  path=f"gradient_path[{path.parameter_state_ref}].dp_payload")
            if rank != root and not same_physical_value(produced, first[2]):
                raise SchemaError("DP child SEND does not consume this source parameter WGRAD",
                                  path=f"gradient_path[{path.parameter_state_ref}].dp_payload")
            if rank == root:
                sync_result, sync = reduction_result, reduction
            else:
                matches = [by_action[dst] for src, dst in dag.transport_edges
                           if by_action[src].step == step
                           and by_action[src].logical_core.die_id == root
                           and by_action[dst].logical_core.die_id == rank
                           and by_action[src].operation_ref == path.named_sync_op_ref
                           and by_action[dst].operation_ref == path.named_sync_op_ref]
                if len(matches) != 1 or not depends_on(
                    by_action[next(src for src, dst in dag.transport_edges
                                   if dst == matches[0].id)], reduction,
                ):
                    raise SchemaError("DP child cannot consume an unreduced broadcast",
                                      path=f"gradient_path[{path.parameter_state_ref}].dp_broadcast")
                sync = matches[0]
                recv, recv_fid, recv_idx = next(
                    (record, fid, idx) for fid, idx, opcode
                    in sync.executable_records if opcode is RecordOpcode.DTE_RECV
                    for record in (next(stream for stream in
                        fragments[fid].core_streams if stream.logical_core ==
                        sync.logical_core).records[idx],)
                )
                sync_result = buffer(sync, recv_fid, recv_idx,
                                     SemanticOperandId.DESTINATION_ADDRESS)
                if (sync_result.dtype is not DType.FP32 or
                        sync_result.size_bytes != path.gradient_bytes):
                    raise SchemaError("DP broadcast destination lacks same FP32 extent",
                                      path=f"gradient_path[{path.parameter_state_ref}].dp_broadcast")
        else:
            # A singleton DP group has a rank-major SUM of one FP32 term.
            # The source WGRAD BufferABI itself is the exact synchronized
            # value; issuing a DTE copy here would manufacture extra work.
            sync_result, sync = produced, wgrad
        update, sgd, update_fid, update_idx = one(
            step, rank, path.named_optimizer_op_ref, RecordOpcode.SGD_UPDATE,
        )
        consumed = buffer(update, update_fid, update_idx,
                          SemanticOperandId.COMPUTE_DATA_ADDRESS)
        updated_weight = buffer(update, update_fid, update_idx,
                                SemanticOperandId.COMPUTE_OUTPUT_ADDRESS)
        sgd_literals = {item.name: item.literal_value for item in sgd.operands
                        if item.literal_value is not None}
        if (not same_physical_value(sync_result, consumed)
                or sgd_literals.get("gradient_datatype") != 3
                or sgd_literals.get("element_count") != path.weight_bytes // 2
                or updated_weight.dtype is not DType.FP16
                or updated_weight.size_bytes != path.weight_bytes
                or not depends_on(update, sync)):
            raise SchemaError("SGD did not consume synchronized physical gradient",
                              path=f"gradient_path[{path.parameter_state_ref}].sgd")
        store, record, store_fid, store_idx = one(
            step, rank, path.named_store_op_ref, RecordOpcode.LSU_STORE,
        )
        closure = state_uses.get((store_fid, store.logical_core,
                                  store_idx, SemanticOperandId.HBM_ADDRESS))
        store_source = buffer(store, store_fid, store_idx,
                              SemanticOperandId.SOURCE_ADDRESS)
        literals = {item.name: item.literal_value for item in record.operands
                    if item.literal_value is not None}
        if (closure is None or states[closure.state_abi_id] !=
            homes[(path.parameter_state_ref, rank)]
                or literals.get("size_bytes") != path.weight_bytes
                or updated_weight.storage_id != store_source.storage_id
                or updated_weight.logical_core != store_source.logical_core
                or updated_weight.region_ref != store_source.region_ref
                or updated_weight.region_offset_bytes !=
                   store_source.region_offset_bytes
                or updated_weight.size_bytes != store_source.size_bytes
                or updated_weight.dtype is not store_source.dtype
                or updated_weight.tensor_slice.shape !=
                   store_source.tensor_slice.shape
                or not depends_on(store, update)):
            raise SchemaError("SGD must physically store same source StateDecl parameter",
                              path=f"gradient_path[{path.parameter_state_ref}].store")
        next_loads = [
            (by_action[target], fid, idx)
            for source, target in dag.state_version_edges if source == store.id
            and by_action[target].step == step + 1
            and by_action[target].logical_core.die_id == rank
            for fid, idx, opcode in by_action[target].executable_records
            if opcode is RecordOpcode.LSU_LOAD
            and (fid, by_action[target].logical_core, idx,
                 SemanticOperandId.HBM_ADDRESS) in state_uses
            and states[state_uses[(fid, by_action[target].logical_core,
                                   idx, SemanticOperandId.HBM_ADDRESS)]
                       .state_abi_id] == homes[(path.parameter_state_ref, rank)]
        ]
        if step < requirements.steps - 1 and len(next_loads) != 1:
            raise SchemaError("updated parameter needs exact next-step HBM LOAD version",
                              path=f"gradient_path[{path.parameter_state_ref}].version")


__all__ = ["require_exact_dense_parameter_state_inventory",
           "require_source_gemm_wgrad_geometry",
           "require_named_gemm_wgrad_typed_buffers",
           "require_source_gemm_dx_geometry",
           "require_named_gemm_dx_typed_buffers",
           "require_named_gemm_dx_state_load",
           "dense_dx_upstream_output_operands",
           "require_full_dense_physical_gradient_paths"]
