"""Whole-workload overlay and exact replacement carrier for MoE Swizzle V2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .ir0 import FusionPattern
from .swizzle import SwizzleActionKind, SwizzleAlgorithm
from .swizzle_moe import MoePacketWitness, MoeRankProgram, MoeTileWitness


MOE_SWIZZLE_OVERLAY_SCHEMA_VERSION = "wafer_frontend.moe_swizzle_overlay/v1alpha1"


class MoeSwizzleDeploymentMode(str, Enum):
    UNFUSED_TYPED_BASELINE = "unfused_typed_baseline"
    FUSED_AUTO = "fused_auto"
    FUSED_FORCED = "fused_forced"


@dataclass(frozen=True, slots=True)
class MoeSwizzleCandidateAdapter:
    pattern: FusionPattern
    region_ref: str
    decision_ref: str
    candidate_ref: str
    algorithm: SwizzleAlgorithm
    rank_programs: tuple[MoeRankProgram, ...]
    packetization: tuple[MoePacketWitness, ...] = ()
    double_buffer: bool = False

    def validate(self, path: str) -> None:
        if self.pattern not in (
            FusionPattern.MOE_DISPATCH_GEMM,
            FusionPattern.MOE_GEMM_COMBINE,
        ):
            raise SchemaError("requires a supported MoE fusion pattern", path=f"{path}.pattern")
        for name in ("region_ref", "decision_ref", "candidate_ref"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if type(self.algorithm) is not SwizzleAlgorithm:
            raise SchemaError("requires typed swizzle algorithm", path=f"{path}.algorithm")
        if tuple(item.rank for item in self.rank_programs) != tuple(range(4)):
            raise SchemaError("candidate adapter requires canonical rank programs 0..3", path=f"{path}.rank_programs")
        action_ids = set()
        for index, program in enumerate(self.rank_programs):
            program.validate(f"{path}.rank_programs[{index}]")
            for action in program.actions:
                if action.id in action_ids:
                    raise SchemaError("candidate adapter contains duplicate actions", path=f"{path}.rank_programs")
                action_ids.add(action.id)

        packet_ids = set()
        for index, packet in enumerate(self.packetization):
            packet.validate(f"{path}.packetization[{index}]")
            if packet.id in packet_ids:
                raise SchemaError("candidate adapter duplicates packet ids", path=f"{path}.packetization")
            packet_ids.add(packet.id)
        if type(self.double_buffer) is not bool:
            raise SchemaError("double_buffer must be bool", path=f"{path}.double_buffer")


@dataclass(frozen=True, slots=True)
class MoeSwizzleDeploymentSelection:
    region_ref: str
    decision_ref: str
    candidate_ref: str
    mode: MoeSwizzleDeploymentMode

    double_buffer: bool
    def validate(self, path: str) -> None:
        for name in ("region_ref", "decision_ref", "candidate_ref"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if type(self.mode) is not MoeSwizzleDeploymentMode:
            raise SchemaError("requires a typed deployment mode", path=f"{path}.mode")


        if type(self.double_buffer) is not bool:
            raise SchemaError("double_buffer must be bool", path=f"{path}.double_buffer")
@dataclass(frozen=True, slots=True)
class MoeBoundaryValueBinding:
    source_value_ref: str
    replacement_value_ref: str
    producer_action_refs: tuple[str, ...]
    consumer_action_refs: tuple[str, ...]
    terminal: bool

    def validate(self, path: str) -> None:
        validate_nonempty(self.source_value_ref, f"{path}.source_value_ref")
        validate_nonempty(self.replacement_value_ref, f"{path}.replacement_value_ref")
        if len(self.producer_action_refs) != len(set(self.producer_action_refs)):
            raise SchemaError("contains duplicate producers", path=f"{path}.producer_action_refs")
        for index, ref in enumerate(self.producer_action_refs):
            validate_nonempty(ref, f"{path}.producer_action_refs[{index}]")
        if len(self.consumer_action_refs) != len(set(self.consumer_action_refs)):
            raise SchemaError("contains duplicate consumers", path=f"{path}.consumer_action_refs")
        for index, ref in enumerate(self.consumer_action_refs):
            validate_nonempty(ref, f"{path}.consumer_action_refs[{index}]")
        if type(self.terminal) is not bool:
            raise SchemaError("terminal must be bool", path=f"{path}.terminal")


@dataclass(frozen=True, slots=True)
class MoeSwizzleLinkedAction:
    id: str
    rank: int
    die_id: int
    kind: str
    preserved: bool
    source_action_refs: tuple[str, ...]
    deps: tuple[str, ...]
    metadata_action_refs: tuple[str, ...]
    read_value_refs: tuple[str, ...]
    write_value_refs: tuple[str, ...]
    assignment_refs: tuple[str, ...]
    expert_index: int | None
    tile_index: int | None
    n_block_index: int | None
    packet_ref: str | None
    stage: int | None
    pivot_rank: int | None

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        validate_nonempty(self.kind, f"{path}.kind")
        validate_uint64(self.rank, f"{path}.rank")
        validate_uint64(self.die_id, f"{path}.die_id")
        if type(self.preserved) is not bool:
            raise SchemaError("preserved must be bool", path=f"{path}.preserved")
        for name in (
            "source_action_refs", "deps", "metadata_action_refs", "read_value_refs",
            "write_value_refs", "assignment_refs",
        ):
            refs = getattr(self, name)
            if len(refs) != len(set(refs)):
                raise SchemaError("contains duplicate refs", path=f"{path}.{name}")
            for index, ref in enumerate(refs):
                validate_nonempty(ref, f"{path}.{name}[{index}]")
        if self.preserved and len(self.source_action_refs) != 1:
            raise SchemaError(
                "preserved linked action requires exactly one source action",
                path=f"{path}.source_action_refs",
            )
        for name in ("expert_index", "tile_index", "n_block_index", "stage", "pivot_rank"):
            value = getattr(self, name)
            if value is not None:
                validate_uint64(value, f"{path}.{name}")
        if self.packet_ref is not None:
            validate_nonempty(self.packet_ref, f"{path}.packet_ref")


@dataclass(frozen=True, slots=True)
class MoeSwizzleOverlay:
    schema_version: str
    producer_pass: str
    id: str
    source_execution_id: str
    source_forward_execution_id: str
    source_workload_selection_id: str | None
    decision_refs: tuple[str, ...]
    deployment_selections: tuple[MoeSwizzleDeploymentSelection, ...]
    replaced_action_refs: tuple[str, ...]
    replacement_rank_programs: tuple[MoeRankProgram, ...]
    preserved_action_refs: tuple[str, ...]
    boundary_value_bindings: tuple[MoeBoundaryValueBinding, ...]
    linked_actions: tuple[MoeSwizzleLinkedAction, ...]
    terminal_value_refs: tuple[str, ...]
    replacement_packetization: tuple[MoePacketWitness, ...]
    train_forward: bool
    replacement_tile_schedule: tuple[MoeTileWitness, ...]

    @classmethod
    def create(cls, **semantic: object) -> "MoeSwizzleOverlay":
        result = cls(
            MOE_SWIZZLE_OVERLAY_SCHEMA_VERSION,
            "build_moe_swizzle_overlay",
            stable_artifact_id(
                "moe_swizzle_overlay",
                semantic,
                schema_version=MOE_SWIZZLE_OVERLAY_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "source_execution_id", "source_forward_execution_id",
                "source_workload_selection_id", "decision_refs",
                "deployment_selections", "replaced_action_refs",
                "replacement_rank_programs", "preserved_action_refs",
                "boundary_value_bindings", "linked_actions", "terminal_value_refs",
                "replacement_packetization", "replacement_tile_schedule", "train_forward",
            )
        }

    def validate(self, path: str = "moe_swizzle_overlay") -> None:
        if self.schema_version != MOE_SWIZZLE_OVERLAY_SCHEMA_VERSION or self.producer_pass != "build_moe_swizzle_overlay":
            raise SchemaError("unsupported overlay schema/producer", path=path)
        validate_nonempty(self.source_execution_id, f"{path}.source_execution_id")
        validate_nonempty(self.source_forward_execution_id, f"{path}.source_forward_execution_id")
        if self.source_workload_selection_id is not None:
            validate_nonempty(
                self.source_workload_selection_id,
                f"{path}.source_workload_selection_id",
            )
        if type(self.train_forward) is not bool:
            raise SchemaError("train_forward must be bool", path=f"{path}.train_forward")
        if len(self.decision_refs) != 2 or len(set(self.decision_refs)) != 2 or len(self.deployment_selections) != 2:
            raise SchemaError("overlay requires exactly two MoE region decisions", path=path)
        for index, ref in enumerate(self.decision_refs):
            validate_nonempty(ref, f"{path}.decision_refs[{index}]")
        for index, selection in enumerate(self.deployment_selections):
            selection.validate(f"{path}.deployment_selections[{index}]")
            if selection.decision_ref not in set(self.decision_refs):
                raise SchemaError("selection references another decision", path=f"{path}.deployment_selections[{index}]")
        if tuple(item.decision_ref for item in self.deployment_selections) != self.decision_refs:
            raise SchemaError("selection/decision order is not exact", path=f"{path}.deployment_selections")
        fused_modes = {
            item.mode for item in self.deployment_selections
            if item.mode is not MoeSwizzleDeploymentMode.UNFUSED_TYPED_BASELINE
        }
        if self.source_workload_selection_id is None:
            if MoeSwizzleDeploymentMode.FUSED_AUTO in fused_modes:
                raise SchemaError(
                    "region preflight cannot claim joint AUTO deployment",
                    path=f"{path}.source_workload_selection_id",
                )
        elif MoeSwizzleDeploymentMode.FUSED_FORCED in fused_modes:
            raise SchemaError(
                "joint workload selection cannot claim forced region preflight",
                path=f"{path}.source_workload_selection_id",
            )
        replaced, preserved = set(self.replaced_action_refs), set(self.preserved_action_refs)
        if not replaced or not preserved or len(replaced) != len(self.replaced_action_refs) or len(preserved) != len(self.preserved_action_refs) or replaced & preserved:
            raise SchemaError("replaced/preserved refs must be unique nonempty and disjoint", path=path)
        if tuple(item.rank for item in self.replacement_rank_programs) != tuple(range(4)):
            raise SchemaError("replacement programs must cover canonical ranks 0..3", path=f"{path}.replacement_rank_programs")
        replacement_ids = set()
        covered = []
        for index, program in enumerate(self.replacement_rank_programs):
            program.validate(f"{path}.replacement_rank_programs[{index}]")
            for action in program.actions:
                if action.id in replacement_ids:
                    raise SchemaError("duplicate replacement action", path=f"{path}.replacement_rank_programs")
                replacement_ids.add(action.id)
                covered.extend(action.original_action_refs)
        if len(covered) != len(set(covered)) or set(covered) != replaced:
            raise SchemaError("replacement actions must cover replaced refs exactly once", path=f"{path}.replacement_rank_programs")
        linked_ids = set()
        replacement_witnesses = {
            action.id: action
            for program in self.replacement_rank_programs
            for action in program.actions
        }
        seen = set()
        for index, action in enumerate(self.linked_actions):
            action.validate(f"{path}.linked_actions[{index}]")
            if action.id in linked_ids or any(dep not in seen for dep in action.deps) or any(ref not in seen for ref in action.metadata_action_refs):
                raise SchemaError("linked actions must be canonical topological closure", path=f"{path}.linked_actions[{index}]")
            linked_ids.add(action.id)
            seen.add(action.id)
            if not action.preserved:
                witness = replacement_witnesses.get(action.id)
                if witness is None or (
                    action.rank,
                    action.kind,
                    action.source_action_refs,
                    action.assignment_refs,
                    action.expert_index,
                    action.tile_index,
                    action.n_block_index,
                    action.packet_ref,
                    action.stage,
                    action.pivot_rank,
                ) != (
                    witness.rank,
                    f"replacement.{witness.kind.value}",
                    witness.original_action_refs,
                    witness.assignment_refs,
                    witness.expert_index,
                    witness.tile_index,
                    witness.n_block_index,
                    witness.packet_ref,
                    witness.stage,
                    witness.pivot_rank,
                ):
                    raise SchemaError(
                        "linked replacement is not the exact rank-program witness",
                        path=f"{path}.linked_actions[{index}]",
                    )
        packet_ids = set()
        for index, packet in enumerate(self.replacement_packetization):
            packet.validate(f"{path}.replacement_packetization[{index}]")
            if packet.id in packet_ids:
                raise SchemaError("replacement packet ids must be unique", path=f"{path}.replacement_packetization")
            packet_ids.add(packet.id)
        packet_refs = {action.packet_ref for program in self.replacement_rank_programs for action in program.actions if action.packet_ref is not None}
        tile_ids = set()
        for index, tile in enumerate(self.replacement_tile_schedule):
            tile.validate(f"{path}.replacement_tile_schedule[{index}]")
            if tile.id in tile_ids:
                raise SchemaError("replacement tile ids must be unique", path=f"{path}.replacement_tile_schedule")
            tile_ids.add(tile.id)

        unknown_packet_refs = packet_refs - packet_ids
        if packet_ids and unknown_packet_refs:
            # UNFUSED candidates intentionally carry executable rank actions
            # but no fused packetization carrier.  A mixed whole pair may
            # therefore contain both fused packet witnesses and baseline
            # singleton triples.  Admit only the exact typed baseline form;
            # grouped/staged fused packets can never enter through this path.
            action_groups = {
                packet_ref: tuple(
                    action for program in self.replacement_rank_programs
                    for action in program.actions
                    if action.packet_ref == packet_ref
                )
                for packet_ref in unknown_packet_refs
            }
            for packet_ref, group in action_groups.items():
                if (
                    len(group) != 3
                    or {item.kind for item in group} != {
                        SwizzleActionKind.SEND,
                        SwizzleActionKind.RECV,
                        SwizzleActionKind.WAIT,
                    }
                    or len({item.assignment_refs for item in group}) != 1
                    or any(len(item.assignment_refs) != 1 or len(item.original_action_refs) != 1 for item in group)
                    or any(item.stage != 0 or item.pivot_rank is not None for item in group)
                ):
                    raise SchemaError("replacement action references unknown packet witness", path=f"{path}.replacement_packetization")
        if replacement_ids - linked_ids or not linked_ids:
            raise SchemaError("linked workload omits replacement actions", path=f"{path}.linked_actions")
        source_coverage = tuple(
            ref for action in self.linked_actions for ref in action.source_action_refs
        )
        if (
            len(source_coverage) != len(set(source_coverage))
            or set(source_coverage) != replaced | preserved
        ):
            raise SchemaError("linked source provenance must cover replaced/preserved refs exactly once", path=f"{path}.linked_actions")
        for index, binding in enumerate(self.boundary_value_bindings):
            binding.validate(f"{path}.boundary_value_bindings[{index}]")
            if any(ref not in linked_ids for ref in binding.producer_action_refs + binding.consumer_action_refs):
                raise SchemaError("boundary binding references unknown linked action", path=f"{path}.boundary_value_bindings[{index}]")
        if len({item.source_value_ref for item in self.boundary_value_bindings}) != len(self.boundary_value_bindings):
            raise SchemaError("boundary values must be unique", path=f"{path}.boundary_value_bindings")
        if len(self.terminal_value_refs) != len(set(self.terminal_value_refs)) or not self.terminal_value_refs:
            raise SchemaError("terminal values must be unique and nonempty", path=f"{path}.terminal_value_refs")
        terminal_bindings = {item.source_value_ref for item in self.boundary_value_bindings if item.terminal}
        if terminal_bindings != set(self.terminal_value_refs):
            raise SchemaError("terminal boundary coverage is not exact", path=f"{path}.terminal_value_refs")
        expected = stable_artifact_id("moe_swizzle_overlay", self._semantic(), schema_version=MOE_SWIZZLE_OVERLAY_SCHEMA_VERSION)
        if self.id != expected:
            raise SchemaError("unstable overlay id", path=f"{path}.id")


__all__ = [name for name in globals() if name.startswith("Moe") or name.startswith("MOE_SWIZZLE")]
