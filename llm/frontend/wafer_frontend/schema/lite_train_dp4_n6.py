"""Executable N6 carriers for the fixed S2-Lite DP4 tree AllReduce."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from ..lowering.context import LoweringContext
from .artifact_manifest import BufferABI, CommandFragment, LinkedProgramManifest, RecordOpcode
from .common import DType, stable_artifact_id
from .global_action import LogicalCoreRef
from .lite_train_dp4 import (
    S2LiteDp4TreeArGlobalAction,
    TreeArFlowKind,
    TreeArReduceKind,
)


S2_LITE_DP4_TREE_AR_N6_INTENT_SCHEMA_VERSION = (
    "wafer_frontend.s2_lite_dp4_tree_ar_n6_intent/v1alpha1"
)
S2_LITE_DP4_TREE_AR_LOWERED_PROGRAM_SCHEMA_VERSION = (
    "wafer_frontend.s2_lite_dp4_tree_ar_lowered_program/v1alpha1"
)
S2_LITE_DP4_TREE_AR_LINKED_PROGRAM_SCHEMA_VERSION = (
    "wafer_frontend.s2_lite_dp4_tree_ar_linked_program/v1alpha1"
)


class Dp4TreeExecutableKind(str, Enum):
    LOCAL_COPY = "local_copy"
    FLOW_SEND = "flow_send"
    FLOW_RECV = "flow_recv"
    FLOW_WAIT = "flow_wait"
    LOCAL_REDUCE = "local_reduce"


@dataclass(frozen=True, slots=True)
class Dp4TreeExecutableUnit:
    id: str
    kind: Dp4TreeExecutableKind
    source_ref: str
    logical_core: LogicalCoreRef
    input_buffer_abi_refs: tuple[str, ...]
    output_buffer_abi_ref: str | None
    bytes: int
    deps: tuple[str, ...]

    @classmethod
    def create(cls, **semantic: object) -> "Dp4TreeExecutableUnit":
        return cls(
            stable_artifact_id(
                "s2_lite_dp4_tree_executable_unit",
                semantic,
                schema_version=S2_LITE_DP4_TREE_AR_N6_INTENT_SCHEMA_VERSION,
            ),
            **semantic,
        )

    def validate(self, path: str) -> None:
        if type(self.kind) is not Dp4TreeExecutableKind:
            raise SchemaError("must be a DP4 tree executable kind", path=f"{path}.kind")
        self.logical_core.validate(f"{path}.logical_core")
        if self.bytes != 2048:
            raise SchemaError("DP4 tree unit bytes must equal 2048", path=f"{path}.bytes")
        if len(set(self.input_buffer_abi_refs)) != len(self.input_buffer_abi_refs):
            raise SchemaError("input BufferABI refs must be unique", path=f"{path}.input_buffer_abi_refs")
        expected = Dp4TreeExecutableUnit.create(
            kind=self.kind,
            source_ref=self.source_ref,
            logical_core=self.logical_core,
            input_buffer_abi_refs=self.input_buffer_abi_refs,
            output_buffer_abi_ref=self.output_buffer_abi_ref,
            bytes=self.bytes,
            deps=self.deps,
        )
        if self != expected:
            raise SchemaError("unstable DP4 tree executable unit", path=path)


def dp4_tree_lowering_contexts(
    source: S2LiteDp4TreeArGlobalAction,
) -> tuple[LoweringContext, ...]:
    source.validate("source")
    result = tuple(
        LoweringContext(
            ir1=replica.projected.graph,
            fusion_plans=replica.projected.fusion_plans,
            standalone_plans=replica.projected.standalone_plans,
            projection=replica.projected.projection,
            schedule_set=replica.schedule_set,
            global_dag=dag,
        )
        for replica, dag in zip(source.scheduled.replicas, source.local_dags)
    )
    for index, context in enumerate(result):
        context.validate(f"dp4_tree_lowering_contexts[{index}]")
    return result


def _one_action_id(source: S2LiteDp4TreeArGlobalAction, replica: int, marker: str) -> str:
    matches = tuple(
        action.id
        for action in source.local_dags[replica].actions
        if marker in getattr(action.source, "task_id", "")
    )
    if len(matches) != 1:
        raise SchemaError(f"requires one action matching {marker!r}", path="source.local_dags")
    return matches[0]


def dp4_tree_units(
    source: S2LiteDp4TreeArGlobalAction,
    gradients: tuple[BufferABI, ...],
    scratches: tuple[BufferABI, ...],
) -> tuple[Dp4TreeExecutableUnit, ...]:
    """Independently derive the canonical 23-unit tree quotient."""

    if len(gradients) != 4 or len(scratches) != 4:
        raise SchemaError("requires four gradients and four scratch slices", path="buffers")
    cores = tuple(item.logical_core for item in gradients)
    g = tuple(item.id for item in gradients)
    s0, s1, s2, s3 = (item.id for item in scratches)
    units: list[Dp4TreeExecutableUnit] = []
    completion: dict[str, str] = {}

    def add(kind, source_ref, core, inputs=(), output=None, deps=()):
        unit = Dp4TreeExecutableUnit.create(
            kind=kind,
            source_ref=source_ref,
            logical_core=core,
            input_buffer_abi_refs=tuple(inputs),
            output_buffer_abi_ref=output,
            bytes=2048,
            deps=tuple(deps),
        )
        units.append(unit)
        return unit

    wgrad0 = _one_action_id(source, 0, ".lm_head_wgrad")
    wgrad2 = _one_action_id(source, 2, ".lm_head_wgrad")
    copy0 = add(Dp4TreeExecutableKind.LOCAL_COPY, wgrad0, cores[0], (g[0],), s0, (wgrad0,))
    copy2 = add(Dp4TreeExecutableKind.LOCAL_COPY, wgrad2, cores[2], (g[2],), s2, (wgrad2,))
    completion[wgrad0] = copy0.id
    completion[wgrad2] = copy2.id

    flows = {item.kind: item for item in source.tree_flows}
    reduces = {item.kind: item for item in source.tree_reduces}
    flow_buffers = {
        TreeArFlowKind.UPLOAD_1_TO_0: (g[1], s1),
        TreeArFlowKind.UPLOAD_3_TO_2: (g[3], s3),
        TreeArFlowKind.PARTIAL_2_TO_0: (s2, s1),
        TreeArFlowKind.BROADCAST_0_TO_1: (s0, g[1]),
        TreeArFlowKind.BROADCAST_0_TO_2: (s0, s2),
        TreeArFlowKind.BROADCAST_2_TO_3: (s2, g[3]),
    }

    def translated(refs: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(completion.get(ref, ref) for ref in refs)

    def add_flow(kind: TreeArFlowKind) -> None:
        flow = flows[kind]
        source_abi, destination_abi = flow_buffers[kind]
        send = add(
            Dp4TreeExecutableKind.FLOW_SEND,
            flow.id,
            cores[flow.source_replica_index],
            (source_abi,),
            None,
            translated(flow.deps),
        )
        recv = add(
            Dp4TreeExecutableKind.FLOW_RECV,
            flow.id,
            cores[flow.destination_replica_index],
            (),
            destination_abi,
            (),
        )
        wait = add(
            Dp4TreeExecutableKind.FLOW_WAIT,
            flow.id,
            cores[flow.destination_replica_index],
            (),
            None,
            (recv.id,),
        )
        completion[flow.id] = wait.id

    def add_reduce(kind: TreeArReduceKind, inputs: tuple[str, str], output: str) -> None:
        reduce = reduces[kind]
        unit = add(
            Dp4TreeExecutableKind.LOCAL_REDUCE,
            reduce.id,
            cores[reduce.executing_replica_index],
            inputs,
            output,
            translated(reduce.deps),
        )
        completion[reduce.id] = unit.id

    add_flow(TreeArFlowKind.UPLOAD_1_TO_0)
    add_flow(TreeArFlowKind.UPLOAD_3_TO_2)
    add_reduce(TreeArReduceKind.PAIR_01, (s0, s1), s0)
    add_reduce(TreeArReduceKind.PAIR_23, (s2, s3), s2)
    add_flow(TreeArFlowKind.PARTIAL_2_TO_0)
    add_reduce(TreeArReduceKind.GLOBAL_AT_0, (s0, s1), s0)
    add_flow(TreeArFlowKind.BROADCAST_0_TO_1)
    add_flow(TreeArFlowKind.BROADCAST_0_TO_2)
    add_flow(TreeArFlowKind.BROADCAST_2_TO_3)
    return tuple(units)


@dataclass(frozen=True, slots=True)
class S2LiteDp4TreeArN6Intent:
    schema_version: str
    producer_pass: str
    id: str
    source: S2LiteDp4TreeArGlobalAction
    lowering_contexts: tuple[LoweringContext, ...]
    gradient_buffer_abis: tuple[BufferABI, ...]
    scratch_buffer_abis: tuple[BufferABI, ...]
    units: tuple[Dp4TreeExecutableUnit, ...]

    @classmethod
    def create(cls, *, source, lowering_contexts, gradient_buffer_abis, scratch_buffer_abis, units):
        semantic = {
            "source": source,
            "lowering_contexts": lowering_contexts,
            "gradient_buffer_abis": gradient_buffer_abis,
            "scratch_buffer_abis": scratch_buffer_abis,
            "units": units,
        }
        result = cls(
            S2_LITE_DP4_TREE_AR_N6_INTENT_SCHEMA_VERSION,
            "s2_lite_dp4_tree_ar_n6_intent",
            stable_artifact_id(
                "s2_lite_dp4_tree_ar_n6_intent",
                semantic,
                schema_version=S2_LITE_DP4_TREE_AR_N6_INTENT_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self):
        return {
            "source": self.source,
            "lowering_contexts": self.lowering_contexts,
            "gradient_buffer_abis": self.gradient_buffer_abis,
            "scratch_buffer_abis": self.scratch_buffer_abis,
            "units": self.units,
        }

    def validate(self, path: str = "s2_lite_dp4_tree_ar_n6_intent") -> None:
        if self.schema_version != S2_LITE_DP4_TREE_AR_N6_INTENT_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "s2_lite_dp4_tree_ar_n6_intent":
            raise SchemaError("wrong producer pass", path=f"{path}.producer_pass")
        if type(self.source) is not S2LiteDp4TreeArGlobalAction:
            raise SchemaError("must embed typed DP4 tree global action", path=f"{path}.source")
        self.source.validate(f"{path}.source")
        if self.lowering_contexts != dp4_tree_lowering_contexts(self.source):
            raise SchemaError("lowering contexts must preserve all four replicas", path=f"{path}.lowering_contexts")
        if len(self.gradient_buffer_abis) != 4 or len(self.scratch_buffer_abis) != 4:
            raise SchemaError("requires four gradients/four scratch slices", path=path)
        for index, abi in enumerate(self.gradient_buffer_abis):
            abi.validate(f"{path}.gradient_buffer_abis[{index}]")
            if abi.dtype is not DType.FP32 or abi.size_bytes != 2048 or abi.logical_core.die_id != index:
                raise SchemaError("gradient ABI must be FP32/2048B on its replica die", path=f"{path}.gradient_buffer_abis[{index}]")
        for root_index, pair in enumerate((self.scratch_buffer_abis[:2], self.scratch_buffer_abis[2:])):
            die = (0, 2)[root_index]
            first, second = pair
            for index, abi in enumerate(pair):
                abi.validate(f"{path}.scratch_buffer_abis[{root_index * 2 + index}]")
                if abi.dtype is not DType.FP32 or abi.size_bytes != 2048 or abi.logical_core.die_id != die:
                    raise SchemaError("scratch ABI must be FP32/2048B on die0/die2", path=f"{path}.scratch_buffer_abis")
            if (
                first.logical_core != second.logical_core
                or first.region_ref != second.region_ref
                or first.region_offset_bytes + 2048 != second.region_offset_bytes
                or first.alignment_bytes != 64
                or second.alignment_bytes != 64
            ):
                raise SchemaError("scratch slices must form exact contiguous 4096B roots", path=f"{path}.scratch_buffer_abis")
            context = self.lowering_contexts[die]
            lo, hi = first.region_offset_bytes, second.region_offset_bytes + second.size_bytes
            for schedule in context.schedule_set.schedules:
                if schedule.die_id != die:
                    continue
                for binding in schedule.buffer_bindings:
                    if binding.region_ref == first.region_ref and max(lo, binding.region_offset_bytes) < min(hi, binding.region_offset_bytes + binding.size_bytes):
                        raise SchemaError("scratch root overlaps a scheduled buffer", path=f"{path}.scratch_buffer_abis")
        expected_units = dp4_tree_units(self.source, self.gradient_buffer_abis, self.scratch_buffer_abis)
        if self.units != expected_units or len(self.units) != 23:
            raise SchemaError("units must equal the canonical 23-unit tree quotient", path=f"{path}.units")
        for index, unit in enumerate(self.units):
            unit.validate(f"{path}.units[{index}]")
        expected_id = stable_artifact_id(
            "s2_lite_dp4_tree_ar_n6_intent",
            self._semantic(),
            schema_version=S2_LITE_DP4_TREE_AR_N6_INTENT_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class S2LiteDp4TreeArLoweredProgram:
    schema_version: str
    producer_pass: str
    id: str
    intent: S2LiteDp4TreeArN6Intent
    local_fragments: tuple[CommandFragment, ...]
    overlay_fragments: tuple[CommandFragment, ...]

    @classmethod
    def create(cls, *, intent, local_fragments, overlay_fragments):
        semantic = {
            "intent": intent,
            "local_fragments": tuple(sorted(local_fragments, key=lambda item: item.id)),
            "overlay_fragments": tuple(sorted(overlay_fragments, key=lambda item: item.id)),
        }
        result = cls(
            S2_LITE_DP4_TREE_AR_LOWERED_PROGRAM_SCHEMA_VERSION,
            "s2_lite_dp4_tree_ar_lowering",
            stable_artifact_id("s2_lite_dp4_tree_ar_lowered_program", semantic, schema_version=S2_LITE_DP4_TREE_AR_LOWERED_PROGRAM_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self):
        return {"intent": self.intent, "local_fragments": self.local_fragments, "overlay_fragments": self.overlay_fragments}

    def validate(self, path: str = "s2_lite_dp4_tree_ar_lowered_program") -> None:
        if self.schema_version != S2_LITE_DP4_TREE_AR_LOWERED_PROGRAM_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "s2_lite_dp4_tree_ar_lowering":
            raise SchemaError("wrong producer pass", path=f"{path}.producer_pass")
        self.intent.validate(f"{path}.intent")
        if self.local_fragments != tuple(sorted(self.local_fragments, key=lambda item: item.id)) or self.overlay_fragments != tuple(sorted(self.overlay_fragments, key=lambda item: item.id)):
            raise SchemaError("fragments must use canonical id order", path=path)
        for index, fragment in enumerate(self.local_fragments + self.overlay_fragments):
            fragment.validate(f"{path}.fragments[{index}]")
        local_claims = tuple(claim for fragment in self.local_fragments for claim in fragment.claimed_action_ids)
        expected_local = {action.id for dag in self.intent.source.local_dags for action in dag.actions}
        if len(local_claims) != 184 or set(local_claims) != expected_local:
            raise SchemaError("local fragments must claim all 184 local actions once", path=f"{path}.local_fragments")
        overlay_claims = tuple(claim for fragment in self.overlay_fragments for claim in fragment.claimed_action_ids)
        if len(self.overlay_fragments) != 17 or len(overlay_claims) != 23 or set(overlay_claims) != {unit.id for unit in self.intent.units}:
            raise SchemaError("overlay must contain 17 leaves claiming all 23 units once", path=f"{path}.overlay_fragments")
        opcodes = tuple(record.opcode for fragment in self.overlay_fragments for stream in fragment.core_streams for record in stream.records)
        expected = {
            RecordOpcode.SRAM_ALLOC_AT: 4,
            RecordOpcode.SRAM_FREE: 4,
            RecordOpcode.DTE_ISSUE: 2,
            RecordOpcode.DTE_SEND: 6,
            RecordOpcode.DTE_RECV: 6,
            RecordOpcode.DTE_WAIT: 8,
            RecordOpcode.LOCAL_REDUCE: 3,
        }
        if len(opcodes) != 33 or {opcode: opcodes.count(opcode) for opcode in expected} != expected:
            raise SchemaError("overlay record quotient must equal the exact 33-record tree", path=f"{path}.overlay_fragments")
        expected_id = stable_artifact_id("s2_lite_dp4_tree_ar_lowered_program", self._semantic(), schema_version=S2_LITE_DP4_TREE_AR_LOWERED_PROGRAM_SCHEMA_VERSION)
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class S2LiteDp4TreeArLinkedProgram:
    schema_version: str
    producer_pass: str
    id: str
    source: S2LiteDp4TreeArLoweredProgram
    manifest: LinkedProgramManifest

    @classmethod
    def create(cls, *, source, manifest):
        semantic = {"source": source, "manifest": manifest}
        result = cls(
            S2_LITE_DP4_TREE_AR_LINKED_PROGRAM_SCHEMA_VERSION,
            "s2_lite_dp4_tree_ar_manifest_linker",
            stable_artifact_id("s2_lite_dp4_tree_ar_linked_program", semantic, schema_version=S2_LITE_DP4_TREE_AR_LINKED_PROGRAM_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "s2_lite_dp4_tree_ar_linked_program") -> None:
        if self.schema_version != S2_LITE_DP4_TREE_AR_LINKED_PROGRAM_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "s2_lite_dp4_tree_ar_manifest_linker":
            raise SchemaError("wrong producer pass", path=f"{path}.producer_pass")
        self.source.validate(f"{path}.source")
        self.manifest.validate(f"{path}.manifest")
        expected_fragments = tuple(sorted(self.source.local_fragments + self.source.overlay_fragments, key=lambda item: item.id))
        if self.manifest.fragments != expected_fragments:
            raise SchemaError("manifest must cover all 201 local/overlay leaves", path=f"{path}.manifest.fragments")
        from ..lowering.lite_train_dp4_linker import link_s2_lite_dp4_tree_ar_manifest
        if self.manifest != link_s2_lite_dp4_tree_ar_manifest(self.source):
            raise SchemaError("manifest must equal exact DP4 production quotient", path=f"{path}.manifest")
        expected_id = stable_artifact_id("s2_lite_dp4_tree_ar_linked_program", {"source": self.source, "manifest": self.manifest}, schema_version=S2_LITE_DP4_TREE_AR_LINKED_PROGRAM_SCHEMA_VERSION)
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")


__all__ = [
    "Dp4TreeExecutableKind",
    "Dp4TreeExecutableUnit",
    "S2_LITE_DP4_TREE_AR_N6_INTENT_SCHEMA_VERSION",
    "S2_LITE_DP4_TREE_AR_LOWERED_PROGRAM_SCHEMA_VERSION",
    "S2_LITE_DP4_TREE_AR_LINKED_PROGRAM_SCHEMA_VERSION",
    "S2LiteDp4TreeArN6Intent",
    "S2LiteDp4TreeArLoweredProgram",
    "S2LiteDp4TreeArLinkedProgram",
    "dp4_tree_lowering_contexts",
    "dp4_tree_units",
]
