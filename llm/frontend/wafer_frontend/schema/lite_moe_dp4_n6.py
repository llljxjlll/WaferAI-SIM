"""Typed N6 carriers for the fixed four-die S3-Lite MoE programs."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .artifact_manifest import BufferABI, CommandFragment, LinkedProgramManifest
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .ir0 import GemmWorkload, SwiGluWorkload
from .lite_moe_dp4_execution import (
    LiteMoeDp4ExecutionCase,
    LiteMoeDp4TaskKind,
)
from .lite_moe_dp4_backward import LiteMoeDp4Backward
from .lite_moe_dp4_train_forward import LiteMoeDp4TrainForward
from .lite_moe_n6 import LiteMoeBufferOperand, LiteMoeStateLoadUnit


LITE_MOE_DP4_INFER_INTENT_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_dp4_infer_n6_intent/v1alpha1"
)
LITE_MOE_DP4_INFER_LOWERED_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_dp4_infer_lowered_program/v1alpha1"
)
LITE_MOE_DP4_INFER_LINKED_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_dp4_infer_linked_program/v1alpha1"
)
LITE_MOE_DP4_TRAIN_FORWARD_LOWERED_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_dp4_train_forward_lowered_program/v1alpha1"
)
LITE_MOE_DP4_TRAIN_FORWARD_LINKED_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_dp4_train_forward_linked_program/v1alpha1"
)
LITE_MOE_DP4_BACKWARD_LOWERED_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_dp4_backward_lowered_program/v1alpha1"
)
LITE_MOE_DP4_BACKWARD_LINKED_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_dp4_backward_linked_program/v1alpha1"
)


def _semantic(instance: object) -> dict[str, object]:
    return {
        name: getattr(instance, name)
        for name in instance.__dataclass_fields__
        if name not in ("schema_version", "producer_pass", "id")
    }


def _stable(instance: object, prefix: str, version: str, path: str) -> None:
    expected = stable_artifact_id(prefix, _semantic(instance), schema_version=version)
    if getattr(instance, "id") != expected:
        raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class LiteMoeDp4ComputeUnit:
    action_ref: str
    node_ref: str
    kind: LiteMoeDp4TaskKind
    workload: GemmWorkload | SwiGluWorkload
    inputs: tuple[LiteMoeBufferOperand, ...]
    output: LiteMoeBufferOperand

    def validate(self, path: str) -> None:
        validate_nonempty(self.action_ref, f"{path}.action_ref")
        validate_nonempty(self.node_ref, f"{path}.node_ref")
        if self.kind not in (LiteMoeDp4TaskKind.GEMM, LiteMoeDp4TaskKind.SWIGLU):
            raise SchemaError("must be GEMM or SWIGLU", path=f"{path}.kind")
        expected = GemmWorkload if self.kind is LiteMoeDp4TaskKind.GEMM else SwiGluWorkload
        if type(self.workload) is not expected:
            raise SchemaError("compute workload kind mismatch", path=f"{path}.workload")
        self.workload.validate(f"{path}.workload")
        if len(self.inputs) != (2 if self.kind is LiteMoeDp4TaskKind.GEMM else 1):
            raise SchemaError("compute input arity mismatch", path=f"{path}.inputs")
        for index, operand in enumerate(self.inputs):
            operand.validate(f"{path}.inputs[{index}]")
        self.output.validate(f"{path}.output")


@dataclass(frozen=True, slots=True)
class LiteMoeDp4DteUnit:
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
            self.source_die_id not in (0, 1, 2, 3)
            or self.destination_die_id not in (0, 1, 2, 3)
            or self.source_die_id == self.destination_die_id
            or self.bytes != 32
        ):
            raise SchemaError("invalid exact DP4 DTE geometry", path=path)


@dataclass(frozen=True, slots=True)
class LiteMoeDp4InferN6Intent:
    schema_version: str
    producer_pass: str
    id: str
    source_case_id: str
    source_global_id: str
    source_schedule_id: str
    source_projection_id: str
    source_n4_id: str
    buffer_abis: tuple[BufferABI, ...]
    state_loads: tuple[LiteMoeStateLoadUnit, ...]
    compute_units: tuple[LiteMoeDp4ComputeUnit, ...]
    dte_units: tuple[LiteMoeDp4DteUnit, ...]

    @classmethod
    def create(cls, **semantic: object) -> "LiteMoeDp4InferN6Intent":
        result = cls(
            LITE_MOE_DP4_INFER_INTENT_SCHEMA_VERSION,
            "lite_moe_dp4_infer_n6_intent",
            stable_artifact_id(
                "s3_lite_moe_dp4_infer_n6_intent",
                semantic,
                schema_version=LITE_MOE_DP4_INFER_INTENT_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "lite_moe_dp4_infer_n6_intent") -> None:
        if (
            self.schema_version != LITE_MOE_DP4_INFER_INTENT_SCHEMA_VERSION
            or self.producer_pass != "lite_moe_dp4_infer_n6_intent"
        ):
            raise SchemaError("unsupported DP4 infer intent schema/producer", path=path)
        for name in (
            "source_case_id", "source_global_id", "source_schedule_id",
            "source_projection_id", "source_n4_id",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        abi_ids: set[str] = set()
        for index, abi in enumerate(self.buffer_abis):
            if type(abi) is not BufferABI:
                raise SchemaError("must be a BufferABI", path=f"{path}.buffer_abis[{index}]")
            abi.validate(f"{path}.buffer_abis[{index}]")
            if abi.id in abi_ids:
                raise SchemaError("duplicate BufferABI id", path=f"{path}.buffer_abis[{index}]")
            abi_ids.add(abi.id)
        if (len(self.state_loads), len(self.compute_units), len(self.dte_units)) != (24, 32, 12):
            raise SchemaError("requires exact 24 state/32 compute/12 DTE units", path=path)
        claims: list[str] = []
        for index, unit in enumerate(self.state_loads):
            unit.validate(f"{path}.state_loads[{index}]")
            claims.append(unit.action_ref)
            if unit.destination_buffer_abi_ref not in abi_ids:
                raise SchemaError("state unit references unknown BufferABI", path=f"{path}.state_loads[{index}]")
        for index, unit in enumerate(self.compute_units):
            unit.validate(f"{path}.compute_units[{index}]")
            claims.append(unit.action_ref)
            if any(item.buffer_abi_ref not in abi_ids for item in (*unit.inputs, unit.output)):
                raise SchemaError("compute unit references unknown BufferABI", path=f"{path}.compute_units[{index}]")
        flow_refs: set[str] = set()
        channels: set[str] = set()
        tokens: set[str] = set()
        for index, unit in enumerate(self.dte_units):
            unit.validate(f"{path}.dte_units[{index}]")
            claims.extend((unit.send_action_ref, unit.recv_action_ref, unit.wait_action_ref))
            if (
                unit.flow_ref in flow_refs
                or unit.channel_symbol in channels
                or unit.token_symbol in tokens
            ):
                raise SchemaError("DTE identities must be unique", path=f"{path}.dte_units[{index}]")
            flow_refs.add(unit.flow_ref); channels.add(unit.channel_symbol); tokens.add(unit.token_symbol)
            if unit.source_buffer_abi_ref not in abi_ids or unit.destination_buffer_abi_ref not in abi_ids:
                raise SchemaError("DTE unit references unknown BufferABI", path=f"{path}.dte_units[{index}]")
        if len(claims) != 92 or len(set(claims)) != 92:
            raise SchemaError("infer intent must claim exactly 92 actions once", path=path)
        _stable(
            self,
            "s3_lite_moe_dp4_infer_n6_intent",
            LITE_MOE_DP4_INFER_INTENT_SCHEMA_VERSION,
            path,
        )


@dataclass(frozen=True, slots=True)
class LiteMoeDp4InferLoweredProgram:
    schema_version: str
    producer_pass: str
    id: str
    source: LiteMoeDp4ExecutionCase
    intent: LiteMoeDp4InferN6Intent
    fragments: tuple[CommandFragment, ...]

    @classmethod
    def create(cls, **semantic: object) -> "LiteMoeDp4InferLoweredProgram":
        result = cls(
            LITE_MOE_DP4_INFER_LOWERED_SCHEMA_VERSION,
            "lite_moe_dp4_infer_lower_program",
            stable_artifact_id(
                "s3_lite_moe_dp4_infer_lowered_program",
                semantic,
                schema_version=LITE_MOE_DP4_INFER_LOWERED_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "lite_moe_dp4_infer_lowered_program") -> None:
        if (
            self.schema_version != LITE_MOE_DP4_INFER_LOWERED_SCHEMA_VERSION
            or self.producer_pass != "lite_moe_dp4_infer_lower_program"
            or type(self.source) is not LiteMoeDp4ExecutionCase
            or type(self.intent) is not LiteMoeDp4InferN6Intent
        ):
            raise SchemaError("unsupported DP4 infer lowered schema/producer", path=path)
        self.source.validate(f"{path}.source")
        self.intent.validate(f"{path}.intent")
        if len(self.fragments) != 80:
            raise SchemaError("infer lowering requires exactly 80 leaves", path=f"{path}.fragments")
        for index, fragment in enumerate(self.fragments):
            fragment.validate(f"{path}.fragments[{index}]")
        _stable(
            self,
            "s3_lite_moe_dp4_infer_lowered_program",
            LITE_MOE_DP4_INFER_LOWERED_SCHEMA_VERSION,
            path,
        )


@dataclass(frozen=True, slots=True)
class LiteMoeDp4InferLinkedProgram:
    schema_version: str
    producer_pass: str
    id: str
    source: LiteMoeDp4InferLoweredProgram
    manifest: LinkedProgramManifest

    @classmethod
    def create(cls, **semantic: object) -> "LiteMoeDp4InferLinkedProgram":
        result = cls(
            LITE_MOE_DP4_INFER_LINKED_SCHEMA_VERSION,
            "lite_moe_dp4_infer_link_program",
            stable_artifact_id(
                "s3_lite_moe_dp4_infer_linked_program",
                semantic,
                schema_version=LITE_MOE_DP4_INFER_LINKED_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "lite_moe_dp4_infer_linked_program") -> None:
        if (
            self.schema_version != LITE_MOE_DP4_INFER_LINKED_SCHEMA_VERSION
            or self.producer_pass != "lite_moe_dp4_infer_link_program"
            or type(self.source) is not LiteMoeDp4InferLoweredProgram
        ):
            raise SchemaError("unsupported DP4 infer linked schema/producer", path=path)
        self.source.validate(f"{path}.source")
        self.manifest.validate(f"{path}.manifest")
        from ..lowering.lite_moe_dp4_linker import link_lite_moe_dp4_infer_manifest
        expected = link_lite_moe_dp4_infer_manifest(self.source)
        if self.manifest != expected:
            raise SchemaError("manifest is not the exact DP4 infer link", path=f"{path}.manifest")
        _stable(
            self,
            "s3_lite_moe_dp4_infer_linked_program",
            LITE_MOE_DP4_INFER_LINKED_SCHEMA_VERSION,
            path,
        )


@dataclass(frozen=True, slots=True)
class LiteMoeDp4TrainForwardLoweredProgram:
    schema_version: str
    producer_pass: str
    id: str
    source: LiteMoeDp4TrainForward
    forward: LiteMoeDp4InferLoweredProgram
    tape_buffer_abis: tuple[BufferABI, ...]
    tape_fragments: tuple[CommandFragment, ...]
    fragments: tuple[CommandFragment, ...]

    @classmethod
    def create(cls, **semantic: object) -> "LiteMoeDp4TrainForwardLoweredProgram":
        result = cls(
            LITE_MOE_DP4_TRAIN_FORWARD_LOWERED_SCHEMA_VERSION,
            "lite_moe_dp4_train_forward_lower_program",
            stable_artifact_id(
                "s3_lite_moe_dp4_train_forward_lowered_program",
                semantic,
                schema_version=LITE_MOE_DP4_TRAIN_FORWARD_LOWERED_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "lite_moe_dp4_train_forward_lowered_program") -> None:
        if (
            self.schema_version != LITE_MOE_DP4_TRAIN_FORWARD_LOWERED_SCHEMA_VERSION
            or self.producer_pass != "lite_moe_dp4_train_forward_lower_program"
            or type(self.source) is not LiteMoeDp4TrainForward
            or type(self.forward) is not LiteMoeDp4InferLoweredProgram
        ):
            raise SchemaError("unsupported DP4 train-forward lowered schema/producer", path=path)
        self.source.validate(f"{path}.source")
        self.forward.validate(f"{path}.forward")
        if self.forward.source != self.source.forward:
            raise SchemaError("train-forward changed embedded infer source", path=f"{path}.forward")
        if len(self.tape_buffer_abis) != 8 or len(self.tape_fragments) != 8:
            raise SchemaError("train-forward requires eight tape ABI/leaves", path=path)
        for index, abi in enumerate(self.tape_buffer_abis):
            abi.validate(f"{path}.tape_buffer_abis[{index}]")
        for index, fragment in enumerate(self.tape_fragments):
            fragment.validate(f"{path}.tape_fragments[{index}]")
        from ..lowering.lite_moe_dp4 import rebase_lite_moe_dp4_infer_fragments
        expected = tuple(sorted((
            *rebase_lite_moe_dp4_infer_fragments(self.forward.fragments, self.source.id),
            *self.tape_fragments,
        ), key=lambda item: item.id))
        if self.fragments != expected or len(self.fragments) != 88:
            raise SchemaError("train-forward must contain exact infer+tape leaves", path=f"{path}.fragments")
        _stable(
            self,
            "s3_lite_moe_dp4_train_forward_lowered_program",
            LITE_MOE_DP4_TRAIN_FORWARD_LOWERED_SCHEMA_VERSION,
            path,
        )


@dataclass(frozen=True, slots=True)
class LiteMoeDp4TrainForwardLinkedProgram:
    schema_version: str
    producer_pass: str
    id: str
    source: LiteMoeDp4TrainForwardLoweredProgram
    manifest: LinkedProgramManifest

    @classmethod
    def create(cls, **semantic: object) -> "LiteMoeDp4TrainForwardLinkedProgram":
        result = cls(
            LITE_MOE_DP4_TRAIN_FORWARD_LINKED_SCHEMA_VERSION,
            "lite_moe_dp4_train_forward_link_program",
            stable_artifact_id(
                "s3_lite_moe_dp4_train_forward_linked_program",
                semantic,
                schema_version=LITE_MOE_DP4_TRAIN_FORWARD_LINKED_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "lite_moe_dp4_train_forward_linked_program") -> None:
        if (
            self.schema_version != LITE_MOE_DP4_TRAIN_FORWARD_LINKED_SCHEMA_VERSION
            or self.producer_pass != "lite_moe_dp4_train_forward_link_program"
            or type(self.source) is not LiteMoeDp4TrainForwardLoweredProgram
        ):
            raise SchemaError("unsupported DP4 train-forward linked schema/producer", path=path)
        self.source.validate(f"{path}.source")
        self.manifest.validate(f"{path}.manifest")
        from ..lowering.lite_moe_dp4_linker import link_lite_moe_dp4_train_forward_manifest
        if self.manifest != link_lite_moe_dp4_train_forward_manifest(self.source):
            raise SchemaError("manifest is not exact DP4 train-forward link", path=f"{path}.manifest")
        _stable(
            self,
            "s3_lite_moe_dp4_train_forward_linked_program",
            LITE_MOE_DP4_TRAIN_FORWARD_LINKED_SCHEMA_VERSION,
            path,
        )


@dataclass(frozen=True, slots=True)
class LiteMoeDp4BackwardLoweredProgram:
    schema_version: str
    producer_pass: str
    id: str
    source: LiteMoeDp4Backward
    intent: LiteMoeDp4InferN6Intent
    fragments: tuple[CommandFragment, ...]

    @classmethod
    def create(cls, **semantic: object) -> "LiteMoeDp4BackwardLoweredProgram":
        result = cls(
            LITE_MOE_DP4_BACKWARD_LOWERED_SCHEMA_VERSION,
            "lite_moe_dp4_backward_lower_program",
            stable_artifact_id(
                "s3_lite_moe_dp4_backward_lowered_program",
                semantic,
                schema_version=LITE_MOE_DP4_BACKWARD_LOWERED_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "lite_moe_dp4_backward_lowered_program") -> None:
        if (
            self.schema_version != LITE_MOE_DP4_BACKWARD_LOWERED_SCHEMA_VERSION
            or self.producer_pass != "lite_moe_dp4_backward_lower_program"
            or type(self.source) is not LiteMoeDp4Backward
            or type(self.intent) is not LiteMoeDp4InferN6Intent
        ):
            raise SchemaError("unsupported DP4 backward lowered schema/producer", path=path)
        self.source.validate(f"{path}.source")
        self.intent.validate(f"{path}.intent")
        formal = self.source.train_forward.forward
        if (
            self.intent.source_case_id != formal.id
            or self.intent.source_global_id != formal.global_dag.id
            or len(self.fragments) != 32
        ):
            raise SchemaError("backward intent/source/count provenance changed", path=path)
        from ..lowering.lite_moe_dp4_backward import lower_lite_moe_dp4_backward
        expected = lower_lite_moe_dp4_backward(
            self.source,
            formal.n4,
            formal.projection,
            formal.schedule,
            formal.global_dag,
            self.intent,
            formal.adapter.spec.trace,
        )
        if self.fragments != expected:
            raise SchemaError("fragments are not exact DP4 backward lowering", path=f"{path}.fragments")
        _stable(
            self,
            "s3_lite_moe_dp4_backward_lowered_program",
            LITE_MOE_DP4_BACKWARD_LOWERED_SCHEMA_VERSION,
            path,
        )


@dataclass(frozen=True, slots=True)
class LiteMoeDp4BackwardLinkedProgram:
    schema_version: str
    producer_pass: str
    id: str
    source: LiteMoeDp4BackwardLoweredProgram
    manifest: LinkedProgramManifest

    @classmethod
    def create(cls, **semantic: object) -> "LiteMoeDp4BackwardLinkedProgram":
        result = cls(
            LITE_MOE_DP4_BACKWARD_LINKED_SCHEMA_VERSION,
            "lite_moe_dp4_backward_link_program",
            stable_artifact_id(
                "s3_lite_moe_dp4_backward_linked_program",
                semantic,
                schema_version=LITE_MOE_DP4_BACKWARD_LINKED_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "lite_moe_dp4_backward_linked_program") -> None:
        if (
            self.schema_version != LITE_MOE_DP4_BACKWARD_LINKED_SCHEMA_VERSION
            or self.producer_pass != "lite_moe_dp4_backward_link_program"
            or type(self.source) is not LiteMoeDp4BackwardLoweredProgram
        ):
            raise SchemaError("unsupported DP4 backward linked schema/producer", path=path)
        self.source.validate(f"{path}.source")
        self.manifest.validate(f"{path}.manifest")
        from ..lowering.lite_moe_dp4_backward_linker import (
            link_lite_moe_dp4_backward_manifest,
        )
        if self.manifest != link_lite_moe_dp4_backward_manifest(self.source):
            raise SchemaError("manifest is not exact DP4 backward link", path=f"{path}.manifest")
        _stable(
            self,
            "s3_lite_moe_dp4_backward_linked_program",
            LITE_MOE_DP4_BACKWARD_LINKED_SCHEMA_VERSION,
            path,
        )


__all__ = [
    "LITE_MOE_DP4_BACKWARD_LINKED_SCHEMA_VERSION",
    "LITE_MOE_DP4_BACKWARD_LOWERED_SCHEMA_VERSION",
    "LITE_MOE_DP4_INFER_INTENT_SCHEMA_VERSION",
    "LITE_MOE_DP4_INFER_LINKED_SCHEMA_VERSION",
    "LITE_MOE_DP4_INFER_LOWERED_SCHEMA_VERSION",
    "LITE_MOE_DP4_TRAIN_FORWARD_LINKED_SCHEMA_VERSION",
    "LITE_MOE_DP4_TRAIN_FORWARD_LOWERED_SCHEMA_VERSION",
    "LiteMoeDp4ComputeUnit",
    "LiteMoeDp4DteUnit",
    "LiteMoeDp4BackwardLinkedProgram",
    "LiteMoeDp4BackwardLoweredProgram",
    "LiteMoeDp4InferLinkedProgram",
    "LiteMoeDp4InferLoweredProgram",
    "LiteMoeDp4InferN6Intent",
    "LiteMoeDp4TrainForwardLinkedProgram",
    "LiteMoeDp4TrainForwardLoweredProgram",
]
