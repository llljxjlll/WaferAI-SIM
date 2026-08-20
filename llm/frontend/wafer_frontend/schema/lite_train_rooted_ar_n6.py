"""Executable N6 intent for the isolated S2-Lite DP2 rooted all-reduce."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from ..lowering.context import LoweringContext
from .artifact_manifest import BufferABI, CommandFragment, LinkedProgramManifest, RecordOpcode
from .common import DType, stable_artifact_id
from .global_action import LogicalCoreRef
from .ir2 import BufferUseRole
from .lite_train_dp2 import (
    RootedArStepKind,
    S2LiteDp2RootedArGlobalAction,
)


S2_LITE_ROOTED_AR_N6_INTENT_SCHEMA_VERSION = (
    "wafer_frontend.s2_lite_rooted_ar_n6_intent/v1alpha1"
)
S2_LITE_ROOTED_AR_LOWERED_PROGRAM_SCHEMA_VERSION = (
    "wafer_frontend.s2_lite_rooted_ar_lowered_program/v1alpha1"
)
S2_LITE_ROOTED_AR_LINKED_PROGRAM_SCHEMA_VERSION = (
    "wafer_frontend.s2_lite_rooted_ar_linked_program/v1alpha1"
)


class RootedArExecutableKind(str, Enum):
    LOCAL_COPY = "local_copy"
    UPLOAD_SEND = "upload_send"
    UPLOAD_RECV = "upload_recv"
    UPLOAD_WAIT = "upload_wait"
    ROOT_REDUCE = "root_reduce"
    DOWNLOAD_SEND = "download_send"
    DOWNLOAD_RECV = "download_recv"
    DOWNLOAD_WAIT = "download_wait"


@dataclass(frozen=True, slots=True)
class RootedArExecutableUnit:
    id: str
    kind: RootedArExecutableKind
    source_step_ref: str
    logical_core: LogicalCoreRef
    input_buffer_abi_refs: tuple[str, ...]
    output_buffer_abi_ref: str | None
    bytes: int
    deps: tuple[str, ...]

    @classmethod
    def create(cls, **semantic: object) -> "RootedArExecutableUnit":
        return cls(
            stable_artifact_id(
                "s2_lite_rooted_ar_executable_unit",
                semantic,
                schema_version=S2_LITE_ROOTED_AR_N6_INTENT_SCHEMA_VERSION,
            ),
            **semantic,
        )

    def validate(self, path: str) -> None:
        if type(self.kind) is not RootedArExecutableKind:
            raise SchemaError("must be a rooted-AR executable kind", path=f"{path}.kind")
        self.logical_core.validate(f"{path}.logical_core")
        if self.bytes != 2048:
            raise SchemaError("rooted-AR executable bytes must equal 2048", path=f"{path}.bytes")
        if len(set(self.input_buffer_abi_refs)) != len(self.input_buffer_abi_refs):
            raise SchemaError("input BufferABI refs must be unique", path=f"{path}.input_buffer_abi_refs")
        expected = stable_artifact_id(
            "s2_lite_rooted_ar_executable_unit",
            {
                "kind": self.kind,
                "source_step_ref": self.source_step_ref,
                "logical_core": self.logical_core,
                "input_buffer_abi_refs": self.input_buffer_abi_refs,
                "output_buffer_abi_ref": self.output_buffer_abi_ref,
                "bytes": self.bytes,
                "deps": self.deps,
            },
            schema_version=S2_LITE_ROOTED_AR_N6_INTENT_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable unit id; expected {expected!r}", path=f"{path}.id")


def rooted_ar_lowering_contexts(
    source: S2LiteDp2RootedArGlobalAction,
) -> tuple[LoweringContext, ...]:
    source.validate("source")
    result = []
    for replica, dag in zip(source.scheduled.replicas, source.local_dags):
        projected = replica.projected
        result.append(LoweringContext(
            ir1=projected.graph,
            fusion_plans=projected.fusion_plans,
            standalone_plans=projected.standalone_plans,
            projection=projected.projection,
            schedule_set=replica.schedule_set,
            global_dag=dag,
        ))
    for index, context in enumerate(result):
        context.validate(f"rooted_ar_lowering_contexts[{index}]")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class S2LiteRootedArN6Intent:
    schema_version: str
    producer_pass: str
    id: str
    source: S2LiteDp2RootedArGlobalAction
    lowering_contexts: tuple[LoweringContext, ...]
    gradient_buffer_abis: tuple[BufferABI, ...]
    scratch_buffer_abis: tuple[BufferABI, ...]
    units: tuple[RootedArExecutableUnit, ...]

    @classmethod
    def create(
        cls,
        *,
        source: S2LiteDp2RootedArGlobalAction,
        lowering_contexts: tuple[LoweringContext, ...],
        gradient_buffer_abis: tuple[BufferABI, ...],
        scratch_buffer_abis: tuple[BufferABI, ...],
        units: tuple[RootedArExecutableUnit, ...],
    ) -> "S2LiteRootedArN6Intent":
        semantic = {
            "source": source,
            "lowering_contexts": lowering_contexts,
            "gradient_buffer_abis": gradient_buffer_abis,
            "scratch_buffer_abis": scratch_buffer_abis,
            "units": units,
        }
        result = cls(
            S2_LITE_ROOTED_AR_N6_INTENT_SCHEMA_VERSION,
            "s2_lite_rooted_ar_n6_intent",
            stable_artifact_id(
                "s2_lite_rooted_ar_n6_intent",
                semantic,
                schema_version=S2_LITE_ROOTED_AR_N6_INTENT_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            "source": self.source,
            "lowering_contexts": self.lowering_contexts,
            "gradient_buffer_abis": self.gradient_buffer_abis,
            "scratch_buffer_abis": self.scratch_buffer_abis,
            "units": self.units,
        }

    def validate(self, path: str = "s2_lite_rooted_ar_n6_intent") -> None:
        if self.schema_version != S2_LITE_ROOTED_AR_N6_INTENT_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "s2_lite_rooted_ar_n6_intent":
            raise SchemaError("wrong producer pass", path=f"{path}.producer_pass")
        self.source.validate(f"{path}.source")
        expected_contexts = rooted_ar_lowering_contexts(self.source)
        if self.lowering_contexts != expected_contexts:
            raise SchemaError("lowering contexts must exactly preserve both replicas", path=f"{path}.lowering_contexts")
        if len(self.gradient_buffer_abis) != 2 or len(self.scratch_buffer_abis) != 2:
            raise SchemaError("requires two gradient and two packed scratch BufferABIs", path=path)
        for name, values in (("gradient_buffer_abis", self.gradient_buffer_abis), ("scratch_buffer_abis", self.scratch_buffer_abis)):
            for index, abi in enumerate(values):
                abi.validate(f"{path}.{name}[{index}]")
                if abi.dtype is not DType.FP32 or abi.size_bytes != 2048:
                    raise SchemaError("rooted-AR buffers must be FP32/2048B", path=f"{path}.{name}[{index}]")
        root_core = self.gradient_buffer_abis[0].logical_core
        leaf_core = self.gradient_buffer_abis[1].logical_core
        if (root_core.die_id, leaf_core.die_id) != (0, 1):
            raise SchemaError("gradient buffers must be rooted on die0/die1", path=f"{path}.gradient_buffer_abis")
        first, second = self.scratch_buffer_abis
        if (
            first.logical_core != root_core
            or second.logical_core != root_core
            or first.region_ref != second.region_ref
            or first.region_offset_bytes + first.size_bytes != second.region_offset_bytes
            or first.alignment_bytes != 64
            or second.alignment_bytes != 64
            or first.binding_id != self.source.ar_contract.root_input_slices[0].id
            or second.binding_id != self.source.ar_contract.root_input_slices[1].id
        ):
            raise SchemaError("scratch BufferABIs must form the exact contiguous rank-major root span", path=f"{path}.scratch_buffer_abis")
        scheduled_root = tuple(
            binding
            for schedule in self.lowering_contexts[0].schedule_set.schedules
            for binding in schedule.buffer_bindings
            if schedule.die_id == 0
        )
        scratch_lo = first.region_offset_bytes
        scratch_hi = second.region_offset_bytes + second.size_bytes
        for binding in scheduled_root:
            if binding.region_ref == first.region_ref and max(scratch_lo, binding.region_offset_bytes) < min(scratch_hi, binding.region_offset_bytes + binding.size_bytes):
                raise SchemaError("root scratch overlaps a scheduled buffer", path=f"{path}.scratch_buffer_abis")
        expected_kinds = tuple(RootedArExecutableKind)
        if tuple(unit.kind for unit in self.units) != expected_kinds:
            raise SchemaError("units must use exact executable rooted-AR order", path=f"{path}.units")
        for index, unit in enumerate(self.units):
            unit.validate(f"{path}.units[{index}]")
        abi_ids = {abi.id for abi in self.gradient_buffer_abis + self.scratch_buffer_abis}
        for index, unit in enumerate(self.units):
            refs = set(unit.input_buffer_abi_refs)
            if unit.output_buffer_abi_ref is not None:
                refs.add(unit.output_buffer_abi_ref)
            if not refs.issubset(abi_ids):
                raise SchemaError("unit references an unknown BufferABI", path=f"{path}.units[{index}]")
        local_copy, up_send, up_recv, up_wait, reduce, down_send, down_recv, down_wait = self.units
        upload, root_reduce, download = self.source.ar_steps
        gates = self.source.sgd_dependencies
        if (
            local_copy.input_buffer_abi_refs != (self.gradient_buffer_abis[0].id,)
            or local_copy.output_buffer_abi_ref != first.id
            or up_send.source_step_ref != upload.id
            or up_send.input_buffer_abi_refs != (self.gradient_buffer_abis[1].id,)
            or up_recv.output_buffer_abi_ref != second.id
            or up_wait.deps != (up_recv.id,)
            or reduce.source_step_ref != root_reduce.id
            or reduce.input_buffer_abi_refs != (first.id, second.id)
            or reduce.output_buffer_abi_ref != first.id
            or set(reduce.deps) != {local_copy.id, up_wait.id}
            or down_send.source_step_ref != download.id
            or down_send.input_buffer_abi_refs != (first.id,)
            or down_recv.output_buffer_abi_ref != self.gradient_buffer_abis[1].id
            or down_wait.deps != (down_recv.id,)
            or tuple(g.depends_on_step_ref for g in gates) != (root_reduce.id, download.id)
        ):
            raise SchemaError("executable units do not close the rooted-AR scratch/dependency contract", path=f"{path}.units")
        expected_id = stable_artifact_id(
            "s2_lite_rooted_ar_n6_intent",
            self._semantic(),
            schema_version=S2_LITE_ROOTED_AR_N6_INTENT_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class S2LiteRootedArLoweredProgram:
    schema_version: str
    producer_pass: str
    id: str
    intent: S2LiteRootedArN6Intent
    local_fragments: tuple[CommandFragment, ...]
    overlay_fragments: tuple[CommandFragment, ...]

    @classmethod
    def create(cls, *, intent: S2LiteRootedArN6Intent, local_fragments: tuple[CommandFragment, ...], overlay_fragments: tuple[CommandFragment, ...]) -> "S2LiteRootedArLoweredProgram":
        semantic = {
            "intent": intent,
            "local_fragments": tuple(sorted(local_fragments, key=lambda item: item.id)),
            "overlay_fragments": tuple(sorted(overlay_fragments, key=lambda item: item.id)),
        }
        result = cls(
            S2_LITE_ROOTED_AR_LOWERED_PROGRAM_SCHEMA_VERSION,
            "s2_lite_rooted_ar_lowering",
            stable_artifact_id("s2_lite_rooted_ar_lowered_program", semantic, schema_version=S2_LITE_ROOTED_AR_LOWERED_PROGRAM_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {"intent": self.intent, "local_fragments": self.local_fragments, "overlay_fragments": self.overlay_fragments}

    def validate(self, path: str = "s2_lite_rooted_ar_lowered_program") -> None:
        if self.schema_version != S2_LITE_ROOTED_AR_LOWERED_PROGRAM_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "s2_lite_rooted_ar_lowering":
            raise SchemaError("wrong producer pass", path=f"{path}.producer_pass")
        self.intent.validate(f"{path}.intent")
        if self.local_fragments != tuple(sorted(self.local_fragments, key=lambda item: item.id)) or self.overlay_fragments != tuple(sorted(self.overlay_fragments, key=lambda item: item.id)):
            raise SchemaError("fragments must use canonical id order", path=path)
        for index, fragment in enumerate(self.local_fragments + self.overlay_fragments):
            fragment.validate(f"{path}.fragments[{index}]")
        local_claims = tuple(action_id for fragment in self.local_fragments for action_id in fragment.claimed_action_ids)
        expected_local = {action.id for dag in self.intent.source.local_dags for action in dag.actions}
        if len(local_claims) != len(expected_local) or set(local_claims) != expected_local:
            raise SchemaError("local fragments must claim all 92 local actions once", path=f"{path}.local_fragments")
        overlay_claims = tuple(action_id for fragment in self.overlay_fragments for action_id in fragment.claimed_action_ids)
        if len(overlay_claims) != 8 or set(overlay_claims) != {unit.id for unit in self.intent.units}:
            raise SchemaError("overlay fragments must claim all 8 executable units once", path=f"{path}.overlay_fragments")
        opcodes = tuple(record.opcode for fragment in self.overlay_fragments for stream in fragment.core_streams for record in stream.records)
        expected_counts = {
            RecordOpcode.SRAM_ALLOC_AT: 2,
            RecordOpcode.DTE_ISSUE: 1,
            RecordOpcode.DTE_SEND: 2,
            RecordOpcode.DTE_RECV: 2,
            RecordOpcode.DTE_WAIT: 3,
            RecordOpcode.LOCAL_REDUCE: 1,
            RecordOpcode.SRAM_FREE: 2,
        }
        if {opcode: opcodes.count(opcode) for opcode in expected_counts} != expected_counts or len(opcodes) != 13:
            raise SchemaError("overlay record quotient must include exact scratch lifecycle and rooted-AR execution", path=f"{path}.overlay_fragments")
        expected_id = stable_artifact_id("s2_lite_rooted_ar_lowered_program", self._semantic(), schema_version=S2_LITE_ROOTED_AR_LOWERED_PROGRAM_SCHEMA_VERSION)
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class S2LiteRootedArLinkedProgram:
    schema_version: str
    producer_pass: str
    id: str
    source: S2LiteRootedArLoweredProgram
    manifest: LinkedProgramManifest

    @classmethod
    def create(cls, *, source: S2LiteRootedArLoweredProgram, manifest: LinkedProgramManifest) -> "S2LiteRootedArLinkedProgram":
        semantic = {"source": source, "manifest": manifest}
        result = cls(
            S2_LITE_ROOTED_AR_LINKED_PROGRAM_SCHEMA_VERSION,
            "s2_lite_rooted_ar_manifest_linker",
            stable_artifact_id("s2_lite_rooted_ar_linked_program", semantic, schema_version=S2_LITE_ROOTED_AR_LINKED_PROGRAM_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "s2_lite_rooted_ar_linked_program") -> None:
        if self.schema_version != S2_LITE_ROOTED_AR_LINKED_PROGRAM_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "s2_lite_rooted_ar_manifest_linker":
            raise SchemaError("wrong producer pass", path=f"{path}.producer_pass")
        self.source.validate(f"{path}.source")
        self.manifest.validate(f"{path}.manifest")
        expected_fragments = tuple(sorted(self.source.local_fragments + self.source.overlay_fragments, key=lambda item: item.id))
        if self.manifest.fragments != expected_fragments:
            raise SchemaError("manifest must exactly cover local and rooted overlay leaves", path=f"{path}.manifest.fragments")
        from ..lowering.lite_train_rooted_ar_linker import (
            link_s2_lite_rooted_ar_manifest,
        )
        if self.manifest != link_s2_lite_rooted_ar_manifest(self.source):
            raise SchemaError(
                "manifest must equal the exact rooted-AR production link quotient",
                path=f"{path}.manifest",
            )
        expected_id = stable_artifact_id(
            "s2_lite_rooted_ar_linked_program",
            {"source": self.source, "manifest": self.manifest},
            schema_version=S2_LITE_ROOTED_AR_LINKED_PROGRAM_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")


__all__ = [
    "RootedArExecutableKind",
    "RootedArExecutableUnit",
    "S2_LITE_ROOTED_AR_N6_INTENT_SCHEMA_VERSION",
    "S2_LITE_ROOTED_AR_LOWERED_PROGRAM_SCHEMA_VERSION",
    "S2_LITE_ROOTED_AR_LINKED_PROGRAM_SCHEMA_VERSION",
    "S2LiteRootedArN6Intent",
    "S2LiteRootedArLoweredProgram",
    "S2LiteRootedArLinkedProgram",
    "rooted_ar_lowering_contexts",
]
