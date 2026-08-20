"""Typed pre-fragment lowering intent for the isolated S3-Lite MoE graph."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .artifact_manifest import BufferABI, CommandFragment, LinkedProgramManifest
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .ir0 import GemmWorkload, SwiGluWorkload
from .lite_moe_execution import (
    LiteMoeGlobalDag,
    LiteMoeProjection,
    LiteMoeScheduled,
    LiteMoeTaskKind,
)
from .lite_moe_n4 import LiteMoeN4IR1


LITE_MOE_N6_INTENT_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_static_moe_n6_intent/v1alpha1"
)
LITE_MOE_LOWERED_PROGRAM_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_lowered_program/v1alpha1"
)
LITE_MOE_LINKED_PROGRAM_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_linked_program/v1alpha1"
)


@dataclass(frozen=True, slots=True)
class LiteMoeBufferOperand:
    buffer_abi_ref: str
    offset_bytes: int
    size_bytes: int

    def validate(self, path: str) -> None:
        validate_nonempty(self.buffer_abi_ref, f"{path}.buffer_abi_ref")
        validate_uint64(self.offset_bytes, f"{path}.offset_bytes")
        validate_uint64(self.size_bytes, f"{path}.size_bytes")
        if self.size_bytes == 0:
            raise SchemaError("operand size must be non-zero", path=f"{path}.size_bytes")


@dataclass(frozen=True, slots=True)
class LiteMoeComputeUnit:
    action_ref: str
    node_ref: str
    kind: LiteMoeTaskKind
    workload: GemmWorkload | SwiGluWorkload
    inputs: tuple[LiteMoeBufferOperand, ...]
    output: LiteMoeBufferOperand

    def validate(self, path: str) -> None:
        validate_nonempty(self.action_ref, f"{path}.action_ref")
        validate_nonempty(self.node_ref, f"{path}.node_ref")
        if self.kind not in (LiteMoeTaskKind.GEMM, LiteMoeTaskKind.SWIGLU):
            raise SchemaError("must be GEMM or SWIGLU", path=f"{path}.kind")
        expected = GemmWorkload if self.kind is LiteMoeTaskKind.GEMM else SwiGluWorkload
        if type(self.workload) is not expected:
            raise SchemaError("compute workload kind mismatch", path=f"{path}.workload")
        self.workload.validate(f"{path}.workload")
        if len(self.inputs) != (2 if self.kind is LiteMoeTaskKind.GEMM else 1):
            raise SchemaError("compute input arity mismatch", path=f"{path}.inputs")
        for index, operand in enumerate(self.inputs):
            operand.validate(f"{path}.inputs[{index}]")
        self.output.validate(f"{path}.output")


@dataclass(frozen=True, slots=True)
class LiteMoeStateLoadUnit:
    action_ref: str
    state_ref: str
    hbm_binding_ref: str
    destination_buffer_abi_ref: str
    bytes: int

    def validate(self, path: str) -> None:
        for name in (
            "action_ref", "state_ref", "hbm_binding_ref",
            "destination_buffer_abi_ref",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        validate_uint64(self.bytes, f"{path}.bytes")
        if self.bytes == 0:
            raise SchemaError("state load bytes must be non-zero", path=f"{path}.bytes")


@dataclass(frozen=True, slots=True)
class LiteMoeDteUnit:
    flow_ref: str
    p2p_binding_ref: str
    pair_route_ref: str
    source_die_id: int
    destination_die_id: int
    send_action_ref: str
    recv_action_ref: str
    wait_action_ref: str
    source_buffer_abi_ref: str
    destination_buffer_abi_ref: str
    channel_symbol: str
    token_symbol: str
    bytes: int

    def validate(self, path: str) -> None:
        for name in (
            "flow_ref", "p2p_binding_ref", "pair_route_ref",
            "send_action_ref", "recv_action_ref", "wait_action_ref",
            "source_buffer_abi_ref", "destination_buffer_abi_ref",
            "channel_symbol", "token_symbol",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if (
            self.source_die_id not in (0, 1)
            or self.destination_die_id not in (0, 1)
            or self.source_die_id == self.destination_die_id
            or self.bytes != 32
        ):
            raise SchemaError("invalid exact DTE unit geometry", path=path)


@dataclass(frozen=True, slots=True)
class LiteMoeN6Intent:
    schema_version: str
    producer_pass: str
    id: str
    source_global_id: str
    source_schedule_id: str
    source_projection_id: str
    source_n4_id: str
    buffer_abis: tuple[BufferABI, ...]
    state_loads: tuple[LiteMoeStateLoadUnit, ...]
    compute_units: tuple[LiteMoeComputeUnit, ...]
    dte_units: tuple[LiteMoeDteUnit, ...]

    @classmethod
    def create(cls, **semantic_key: object) -> "LiteMoeN6Intent":
        result = cls(
            schema_version=LITE_MOE_N6_INTENT_SCHEMA_VERSION,
            producer_pass="lite_moe_n6_intent",
            id=stable_artifact_id(
                "s3_lite_static_moe_n6_intent",
                semantic_key,
                schema_version=LITE_MOE_N6_INTENT_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    def validate(self, path: str = "lite_moe_n6_intent") -> None:
        if (
            self.schema_version != LITE_MOE_N6_INTENT_SCHEMA_VERSION
            or self.producer_pass != "lite_moe_n6_intent"
        ):
            raise SchemaError("unsupported N6 intent schema/producer", path=path)
        for name in (
            "source_global_id", "source_schedule_id", "source_projection_id",
            "source_n4_id",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if len(self.buffer_abis) != 64:
            raise SchemaError("must carry exactly 64 BufferABI values", path=f"{path}.buffer_abis")
        buffer_ids: set[str] = set()
        for index, abi in enumerate(self.buffer_abis):
            if type(abi) is not BufferABI:
                raise SchemaError("must be a BufferABI", path=f"{path}.buffer_abis[{index}]")
            abi.validate(f"{path}.buffer_abis[{index}]")
            if abi.id in buffer_ids:
                raise SchemaError("duplicate BufferABI id", path=f"{path}.buffer_abis[{index}]")
            buffer_ids.add(abi.id)
        if (len(self.state_loads), len(self.compute_units), len(self.dte_units)) != (24, 32, 8):
            raise SchemaError("must contain exact 24 state/32 compute/8 DTE units", path=path)
        claimed_actions: list[str] = []
        for index, unit in enumerate(self.state_loads):
            unit.validate(f"{path}.state_loads[{index}]")
            claimed_actions.append(unit.action_ref)
            if unit.destination_buffer_abi_ref not in buffer_ids:
                raise SchemaError("state unit references unknown BufferABI", path=f"{path}.state_loads[{index}]")
        for index, unit in enumerate(self.compute_units):
            unit.validate(f"{path}.compute_units[{index}]")
            claimed_actions.append(unit.action_ref)
            if any(item.buffer_abi_ref not in buffer_ids for item in (*unit.inputs, unit.output)):
                raise SchemaError("compute unit references unknown BufferABI", path=f"{path}.compute_units[{index}]")
        flow_refs: set[str] = set()
        token_symbols: set[str] = set()
        channel_symbols: set[str] = set()
        for index, unit in enumerate(self.dte_units):
            unit.validate(f"{path}.dte_units[{index}]")
            claimed_actions.extend((unit.send_action_ref, unit.recv_action_ref, unit.wait_action_ref))
            if (
                unit.flow_ref in flow_refs
                or unit.token_symbol in token_symbols
                or unit.channel_symbol in channel_symbols
            ):
                raise SchemaError("DTE flow/channel/token identities must be unique", path=f"{path}.dte_units[{index}]")
            flow_refs.add(unit.flow_ref); token_symbols.add(unit.token_symbol); channel_symbols.add(unit.channel_symbol)
            if unit.source_buffer_abi_ref not in buffer_ids or unit.destination_buffer_abi_ref not in buffer_ids:
                raise SchemaError("DTE unit references unknown BufferABI", path=f"{path}.dte_units[{index}]")
        if len(claimed_actions) != 80 or len(set(claimed_actions)) != 80:
            raise SchemaError("N6 units must claim exactly 80 actions once", path=path)
        expected_id = stable_artifact_id(
            "s3_lite_static_moe_n6_intent",
            self._semantic_key(),
            schema_version=LITE_MOE_N6_INTENT_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError("unstable artifact id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class LiteMoeLoweredProgram:
    schema_version: str
    producer_pass: str
    id: str
    n4: LiteMoeN4IR1
    projection: LiteMoeProjection
    schedule: LiteMoeScheduled
    global_dag: LiteMoeGlobalDag
    intent: LiteMoeN6Intent
    fragments: tuple[CommandFragment, ...]

    @classmethod
    def create(cls, **semantic_key: object) -> "LiteMoeLoweredProgram":
        semantic_key["fragments"] = tuple(
            sorted(semantic_key["fragments"], key=lambda item: item.id)  # type: ignore[index,union-attr]
        )
        result = cls(
            schema_version=LITE_MOE_LOWERED_PROGRAM_SCHEMA_VERSION,
            producer_pass="lite_moe_lowering",
            id=stable_artifact_id(
                "s3_lite_moe_lowered_program",
                semantic_key,
                schema_version=LITE_MOE_LOWERED_PROGRAM_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "n4": self.n4,
            "projection": self.projection,
            "schedule": self.schedule,
            "global_dag": self.global_dag,
            "intent": self.intent,
            "fragments": self.fragments,
        }

    def validate(self, path: str = "lite_moe_lowered_program") -> None:
        if (
            self.schema_version != LITE_MOE_LOWERED_PROGRAM_SCHEMA_VERSION
            or self.producer_pass != "lite_moe_lowering"
        ):
            raise SchemaError("unsupported lowered schema/producer", path=path)
        if type(self.n4) is not LiteMoeN4IR1:
            raise SchemaError("must embed LiteMoeN4IR1", path=f"{path}.n4")
        if type(self.projection) is not LiteMoeProjection:
            raise SchemaError("must embed LiteMoeProjection", path=f"{path}.projection")
        if type(self.schedule) is not LiteMoeScheduled:
            raise SchemaError("must embed LiteMoeScheduled", path=f"{path}.schedule")
        if type(self.global_dag) is not LiteMoeGlobalDag:
            raise SchemaError("must embed LiteMoeGlobalDag", path=f"{path}.global_dag")
        if type(self.intent) is not LiteMoeN6Intent:
            raise SchemaError("must embed LiteMoeN6Intent", path=f"{path}.intent")
        from ..passes.lite_moe_execution import (
            validate_lite_moe_global,
            validate_lite_moe_projection,
            validate_lite_moe_schedule,
        )
        from ..passes.lite_moe_n6 import validate_lite_moe_n6_intent
        from ..lowering.lite_moe import lower_lite_moe_n6_intent

        validate_lite_moe_projection(self.projection, self.n4)
        validate_lite_moe_schedule(self.schedule, self.projection, self.n4)
        validate_lite_moe_global(
            self.global_dag, self.schedule, self.projection, self.n4
        )
        validate_lite_moe_n6_intent(
            self.intent,
            self.global_dag,
            self.schedule,
            self.projection,
            self.n4,
        )
        expected = lower_lite_moe_n6_intent(
            self.intent, self.global_dag, self.schedule, self.n4
        )
        if self.fragments != expected:
            raise SchemaError(
                "fragments must equal the production MoE lowering",
                path=f"{path}.fragments",
            )
        if len(self.fragments) != 72 or sum(
            len(fragment.claimed_action_ids) for fragment in self.fragments
        ) != 80:
            raise SchemaError("must carry exact 72 leaves/80 actions", path=f"{path}.fragments")
        expected_id = stable_artifact_id(
            "s3_lite_moe_lowered_program",
            self._semantic_key(),
            schema_version=LITE_MOE_LOWERED_PROGRAM_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError("unstable artifact id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class LiteMoeLinkedProgram:
    schema_version: str
    producer_pass: str
    id: str
    source: LiteMoeLoweredProgram
    manifest: LinkedProgramManifest

    @classmethod
    def create(
        cls, *, source: LiteMoeLoweredProgram, manifest: LinkedProgramManifest
    ) -> "LiteMoeLinkedProgram":
        semantic_key = {"source": source, "manifest": manifest}
        result = cls(
            schema_version=LITE_MOE_LINKED_PROGRAM_SCHEMA_VERSION,
            producer_pass="lite_moe_manifest_linker",
            id=stable_artifact_id(
                "s3_lite_moe_linked_program",
                semantic_key,
                schema_version=LITE_MOE_LINKED_PROGRAM_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {"source": self.source, "manifest": self.manifest}

    def validate(self, path: str = "lite_moe_linked_program") -> None:
        if (
            self.schema_version != LITE_MOE_LINKED_PROGRAM_SCHEMA_VERSION
            or self.producer_pass != "lite_moe_manifest_linker"
        ):
            raise SchemaError("unsupported linked schema/producer", path=path)
        if type(self.source) is not LiteMoeLoweredProgram:
            raise SchemaError("must embed LiteMoeLoweredProgram", path=f"{path}.source")
        self.source.validate(f"{path}.source")
        if type(self.manifest) is not LinkedProgramManifest:
            raise SchemaError("must embed LinkedProgramManifest", path=f"{path}.manifest")
        self.manifest.validate(f"{path}.manifest")
        from ..lowering.linker import NaiveManifestLinker

        expected = NaiveManifestLinker().link_lite_moe(self.source)
        if self.manifest != expected:
            raise SchemaError("manifest is not the production MoE quotient", path=f"{path}.manifest")
        expected_id = stable_artifact_id(
            "s3_lite_moe_linked_program",
            self._semantic_key(),
            schema_version=LITE_MOE_LINKED_PROGRAM_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError("unstable artifact id", path=f"{path}.id")


__all__ = [
    "LITE_MOE_LINKED_PROGRAM_SCHEMA_VERSION",
    "LITE_MOE_LOWERED_PROGRAM_SCHEMA_VERSION",
    "LITE_MOE_N6_INTENT_SCHEMA_VERSION", "LiteMoeBufferOperand",
    "LiteMoeComputeUnit", "LiteMoeDteUnit", "LiteMoeLinkedProgram",
    "LiteMoeLoweredProgram", "LiteMoeN6Intent", "LiteMoeStateLoadUnit",
]
