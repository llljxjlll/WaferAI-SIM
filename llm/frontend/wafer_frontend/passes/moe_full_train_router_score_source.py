"""Versioned trainable top1 router requirement; reject the static trace leaf.

The frozen expert choice can act as stop-gradient, while a *dynamic* signed
selected score must participate in weighted combine before gate dW is real.
The existing static gate_weight_f32_bits=1.0 is metadata, not a dScore producer.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..schema.artifact_manifest import OperandKind, RecordOpcode
from ..schema.common import DType, stable_artifact_id
from ..schema.flexible_moe import (
    FlexibleMoeMode, MoeRectActionKind, MoeRectStateRole,
)
from ..schema.moe_compile_sequence import MoeCompileSequence
from .moe_full_train_named_wgrad_operands import (
    _buffer_slice, _one_record, _operand_slice,
)


_VERSION = "wafer_frontend.moe_dynamic_signed_top1_router/v1alpha1"
_FP32_ONE_BITS = 0x3F800000


@dataclass(frozen=True, slots=True)
class MoeSignedRouterRoute:
    token_index: int
    source_rank: int
    selected_expert: int
    expert_home_rank: int
    expert_slot_index: int


@dataclass(frozen=True, slots=True)
class MoeTrainableRouterScorePath:
    step: int
    layer: int
    source_rank: int
    gate_action_ref: str
    weighted_combine_action_ref: str
    combine_backward_action_ref: str
    gate_wgrad_action_ref: str
    gate_parameter_state_ref: str
    gate_gradient_state_ref: str
    routes: tuple[MoeSignedRouterRoute, ...]
    hidden_size: int
    expert_count: int
    score_tape_bytes: int
    dscore_bytes: int
    forward_expert_bytes: int
    backward_dcombined_bytes: int
    fp32_gate_gradient_bytes: int


@dataclass(frozen=True, slots=True)
class MoeTrainableSignedRouterRequirements:
    id: str
    source_sequence_ref: str
    original_static_case_ref: str
    dynamic_score_case_ref: str
    version: str
    score_weight: str
    assignment_derivative: str
    dscore_rule: str
    paths: tuple[MoeTrainableRouterScorePath, ...]

    def validate_against(self, sequence: MoeCompileSequence) -> None:
        expected = build_moe_trainable_signed_router_requirements(sequence)
        if self != expected:
            raise SchemaError("dynamic router source/hardware/route/score contract must be signed from original P2",
                              path="moe_trainable_signed_router")

    def _path(self, step: int, layer: int, source_rank: int):
        matches = [path for path in self.paths
                   if (path.step, path.layer, path.source_rank)
                   == (step, layer, source_rank)]
        if len(matches) != 1:
            raise SchemaError("each step/layer/source rank requires an exact router path",
                              path="moe_trainable_signed_router.paths")
        return matches[0]

    def require_signed_gate_score_tape(self, sequence: MoeCompileSequence) -> None:
        """A real GATE MATMUL must write an independent retained (k,E) score."""
        self.validate_against(sequence)
        for path in self.paths:
            if not path.routes:
                continue
            manifest = _manifest(sequence, path)
            try:
                score = _buffer_slice(manifest, path.source_rank,
                                      ".router_score_tape", DType.FP16,
                                      0, path.score_tape_bytes)
            except SchemaError as exc:
                raise SchemaError("forward GATE has no independently retained signed score tape",
                                  path=f"router.step{path.step}.layer{path.layer}.rank{path.source_rank}") from exc
            _, fragment, stream, index = _one_record(
                manifest, path.source_rank, path.gate_action_ref,
                (RecordOpcode.MATMUL,))
            actual = _operand_slice(manifest, fragment, stream, index,
                                    "output_address", path.score_tape_bytes,
                                    require_exact_view=True)
            if actual != score:
                raise SchemaError("forward GATE MATMUL must write its real score tape, never reused output workspace",
                                  path=f"router.step{path.step}.layer{path.layer}")

    def require_score_weighted_combine(self, sequence: MoeCompileSequence) -> None:
        """The source expert output must be multiplied by its selected score."""
        self.validate_against(sequence)
        for path in self.paths:
            if not path.routes:
                continue
            manifest = _manifest(sequence, path)
            found = _router_record(manifest, path.source_rank,
                                   path.weighted_combine_action_ref)
            if (len(found) != 1
                    or found[0][0].opcode is RecordOpcode.LOCAL_REDUCE
                    or not any(item.name == "score_address"
                               for item in found[0][0].operands)):
                raise SchemaError("weighted combine does not consume genuine router score and expert output",
                                  path=f"router.step{path.step}.layer{path.layer}")
            record, fragment, stream, index = found[0]
            _route_table(record, path)
            for name, suffix, size in (
                ("score_address", ".router_score_tape", path.score_tape_bytes),
                ("expert_output_address", ".expert_return_tape",
                 path.forward_expert_bytes),
                ("output_address", ".weighted_combined",
                 path.forward_expert_bytes),
            ):
                physical = _buffer_slice(manifest, path.source_rank, suffix,
                                         DType.FP16, 0, size)
                if (_operand_slice(manifest, fragment, stream, index, name,
                                   size, require_exact_view=True) != physical):
                    raise SchemaError("router score/returned expert output/combined result must use distinct real SRAM tape",
                                      path=f"router.step{path.step}.layer{path.layer}")

    def require_signed_dscore_producer(self, sequence: MoeCompileSequence) -> None:
        """dS[selected]=dot(dCombined, expert_output); all other dS are 0."""
        self.validate_against(sequence)
        for path in self.paths:
            if not path.routes:
                continue
            manifest = _manifest(sequence, path)
            found = _router_record(manifest, path.source_rank,
                                   path.combine_backward_action_ref)
            if (len(found) != 1
                    or found[0][0].opcode is RecordOpcode.LOCAL_REDUCE
                    or not any(item.name == "dscore_address"
                               for item in found[0][0].operands)):
                raise SchemaError("router backward lacks selected-score derivative from real expert output and dCombined",
                                  path=f"router.step{path.step}.layer{path.layer}")
            record, fragment, stream, index = found[0]
            _route_table(record, path)
            for name, suffix, size in (
                ("expert_output_address", ".expert_return_tape",
                 path.forward_expert_bytes),
                ("upstream_address", ".dcombined_gradient",
                 path.backward_dcombined_bytes),
                ("dscore_address", ".router_score_gradient",
                 path.dscore_bytes),
            ):
                physical = _buffer_slice(manifest, path.source_rank, suffix,
                                         DType.FP16, 0, size)
                if (_operand_slice(manifest, fragment, stream, index, name,
                                   size, require_exact_view=True) != physical):
                    raise SchemaError("router dScore cannot be produced from wrong returned expert/dCombined bytes",
                                      path=f"router.step{path.step}.layer{path.layer}")

    def require_native_fp32_router_wgrad(self, sequence: MoeCompileSequence) -> None:
        """Only X[k,H] and signed dScore[k,E] can feed native 0x25 dW[H,E]."""
        self.validate_against(sequence)
        for path in self.paths:
            if not path.routes:
                continue
            manifest = _manifest(sequence, path)
            record, fragment, stream, index = _one_record(
                manifest, path.source_rank, path.gate_wgrad_action_ref,
                (RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING, RecordOpcode.MATMUL))
            if record.opcode is not RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING:
                raise SchemaError("old router FP16 MATMUL then cast cannot replace signed dScore native 0x25 FP32 gradient",
                                  path=f"router.step{path.step}.layer{path.layer}")
            literals = {item.name: item.literal_value for item in record.operands
                        if item.kind is OperandKind.LITERAL}
            if (tuple(literals.get(name) for name in ("m", "n", "k"))
                    != (path.hidden_size, path.expert_count, len(path.routes))
                    or tuple(literals.get(name) for name in
                             ("activation_datatype", "upstream_datatype",
                              "gradient_datatype")) != (1, 1, 3)):
                raise SchemaError("native router gate WGRAD must preserve P2 H/E/rank rows and physical FP16/FP32 dtypes",
                                  path=f"router.step{path.step}.layer{path.layer}")
            activation = _buffer_slice(
                manifest, path.source_rank, ".activation", DType.FP16,
                0, len(path.routes) * 2 * path.hidden_size)
            dscore = _buffer_slice(
                manifest, path.source_rank, ".router_score_gradient",
                DType.FP16, 0, path.dscore_bytes)
            gate_grad = _buffer_slice(
                manifest, path.source_rank,
                f".state.{path.gate_gradient_state_ref}", DType.FP32,
                0, path.fp32_gate_gradient_bytes)
            if any(_operand_slice(manifest, fragment, stream, index, name,
                                  expected.size_bytes, require_exact_view=True)
                   != expected for name, expected in (
                       ("activation_address", activation),
                       ("upstream_address", dscore),
                       ("gradient_address", gate_grad),
                   )):
                raise SchemaError("native router gradient operands are not the signed score source and FP32 gate matrix",
                                  path=f"router.step{path.step}.layer{path.layer}")


def _manifest(sequence, path):
    matches = [unit for unit in sequence.units
               if (unit.step, unit.layer) == (path.step, path.layer)]
    if len(matches) != 1:
        raise SchemaError("router requires one source production leaf per step/layer",
                          path="moe_trainable_signed_router")
    return matches[0].linked_manifest


def _router_record(manifest, source_rank: int, action_ref: str):
    return [(record, fragment, stream, index)
            for fragment in manifest.fragments for stream in fragment.core_streams
            if stream.logical_core.die_id == source_rank
            for index, record in enumerate(stream.records)
            if record.source_global_action_id == action_ref
            and record.opcode is not RecordOpcode.SRAM_BIND]


def _route_table(record, path: MoeTrainableRouterScorePath) -> None:
    literals = {item.name: item.literal_value for item in record.operands
                if item.kind is OperandKind.LITERAL}
    expected = tuple((row.token_index, row.source_rank,
                      row.selected_expert, row.expert_home_rank,
                      row.expert_slot_index) for row in path.routes)
    actual = literals.get("route_table")
    if type(actual) is not tuple or actual != expected:
        raise SchemaError("router native record must consume exact frozen token/expert/home/slot trace",
                          path=f"router.step{path.step}.layer{path.layer}")


def _build_paths(sequence: MoeCompileSequence) -> tuple[MoeTrainableRouterScorePath, ...]:
    paths = []
    for unit in sequence.units:
        spec, plan = unit.spec, unit.plan
        if spec.mode is not FlexibleMoeMode.TRAIN:
            raise SchemaError("trainable signed router requires TRAIN source",
                              path="moe_trainable_signed_router")
        if any(assignment.gate_weight_f32_bits != _FP32_ONE_BITS
               for assignment in spec.trace.assignments):
            raise SchemaError("new dynamic score case cannot silently ignore externally specified static gate weights",
                              path="moe_trainable_signed_router.trace")
        for rank in range(spec.mesh.rank_count):
            grouped = tuple(assignment for assignment in spec.trace.assignments
                            if assignment.source_rank == rank)
            refs = tuple(f"assignment.{assignment.token_index}"
                         for assignment in grouped)
            by_kind = {
                kind: [action for action in plan.actions
                       if action.kind is kind and action.rank == rank]
                for kind in (MoeRectActionKind.GATE,
                             MoeRectActionKind.WEIGHTED_COMBINE,
                             MoeRectActionKind.COMBINE_BACKWARD,
                             MoeRectActionKind.GATE_WGRAD)
            }
            if any(len(items) != 1 or items[0].assignment_refs != refs
                   for items in by_kind.values()):
                raise SchemaError("router actions must preserve every frozen top1 token choice through backward",
                                  path=f"router.step{unit.step}.layer{unit.layer}.rank{rank}")
            gate, combine, backward, wgrad = (
                by_kind[kind][0] for kind in by_kind)
            states = {
                role: [state for state in plan.state_bindings
                       if state.role is role and state.owner_rank == rank]
                for role in (MoeRectStateRole.GATE_PARAMETER,
                             MoeRectStateRole.GATE_GRADIENT)
            }
            if any(len(items) != 1 for items in states.values()):
                raise SchemaError("router trainable parameter/gradient owner must be source-unique",
                                  path=f"router.step{unit.step}.layer{unit.layer}.rank{rank}")
            weight = states[MoeRectStateRole.GATE_PARAMETER][0]
            gradient = states[MoeRectStateRole.GATE_GRADIENT][0]
            rows, hidden, experts = len(grouped), spec.hidden_size, spec.expert_count
            if (weight.size_bytes != 2 * hidden * experts
                    or gradient.size_bytes != 4 * hidden * experts
                    or gate.flops != 2 * rows * hidden * experts
                    or combine.flops != 2 * rows * hidden
                    or wgrad.flops != 2 * rows * hidden * experts
                    or wgrad.state_refs != (gradient.id,)
                    or combine.id not in backward.deps
                    or backward.id not in wgrad.deps):
                raise SchemaError("router score/weighted-combine/gradient/parameter P2 geometry or lineage differs",
                                  path=f"router.step{unit.step}.layer{unit.layer}.rank{rank}")
            paths.append(MoeTrainableRouterScorePath(
                unit.step, unit.layer, rank, gate.id, combine.id, backward.id,
                wgrad.id, weight.id, gradient.id,
                tuple(MoeSignedRouterRoute(
                    item.token_index, item.source_rank, item.expert_index,
                    item.expert_home_rank, item.slot_index) for item in grouped),
                hidden, experts, 2 * rows * experts, 2 * rows * experts,
                2 * rows * hidden, 2 * rows * hidden, gradient.size_bytes,
            ))
    return tuple(paths)


def build_moe_trainable_signed_router_requirements(
    sequence: MoeCompileSequence,
) -> MoeTrainableSignedRouterRequirements:
    sequence.validate("moe_trainable_signed_router.source")
    original = sequence.materialization.request.case_id
    versioned = stable_artifact_id(
        "moe_dynamic_signed_top1_router_case",
        {"source_case": original, "source_sequence": sequence.id,
         "version": _VERSION, "weight": "selected_raw_signed_score",
         "derivative": "selected_dot_upstream_expert"},
        schema_version=_VERSION,
    )
    semantic = {
        "source_sequence_ref": sequence.id,
        "original_static_case_ref": original,
        "dynamic_score_case_ref": versioned,
        "version": _VERSION,
        "score_weight": "selected_raw_signed_score",
        "assignment_derivative": "stop_gradient_top1_selection",
        "dscore_rule": "selected_dot_upstream_expert_other_scores_zero",
        "paths": _build_paths(sequence),
    }
    return MoeTrainableSignedRouterRequirements(
        stable_artifact_id("moe_trainable_signed_router_source", semantic,
                           schema_version=_VERSION),
        **semantic,
    )


__all__ = ["MoeSignedRouterRoute", "MoeTrainableRouterScorePath",
           "MoeTrainableSignedRouterRequirements",
           "build_moe_trainable_signed_router_requirements"]
