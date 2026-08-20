"""Self-contained production cases for the fixed S3-Lite DP4 MoE matrix."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.lite_moe_dp4 import (
    build_lite_moe_dp4_ir0_adapter,
    build_lite_moe_dp4_oracle,
    build_lite_moe_dp4_spec,
    build_lite_moe_dp4_topology,
    validate_lite_moe_dp4_ir0_adapter,
)
from llm.frontend.wafer_frontend.passes.lite_moe_dp4_backward import (
    build_lite_moe_dp4_backward,
    validate_lite_moe_dp4_backward,
)
from llm.frontend.wafer_frontend.passes.lite_moe_dp4_execution import (
    build_lite_moe_dp4_execution_case,
    validate_lite_moe_dp4_execution_case,
)
from llm.frontend.wafer_frontend.passes.lite_moe_dp4_link_program import (
    link_lite_moe_dp4_backward_program,
    link_lite_moe_dp4_infer_program,
    link_lite_moe_dp4_train_forward_program,
)
from llm.frontend.wafer_frontend.passes.lite_moe_dp4_lower_program import (
    lower_lite_moe_dp4_backward_program,
    lower_lite_moe_dp4_infer_program,
    lower_lite_moe_dp4_train_forward_program,
)
from llm.frontend.wafer_frontend.passes.lite_moe_dp4_train_forward import (
    build_lite_moe_dp4_train_forward,
    validate_lite_moe_dp4_train_forward,
)
from llm.frontend.wafer_frontend.passes.load_fabric import (
    hbm_address_spaces_from_data,
    physical_fabric_from_data,
    validate_identity_mapping_text,
)
from llm.frontend.wafer_frontend.passes.program_io import (
    build_deterministic_timing_state_overrides,
    build_timing_program_io,
)
from llm.frontend.wafer_frontend.policies.registry import production_registry
from llm.frontend.wafer_frontend.schema.lite_moe_dp4 import (
    S3_LITE_MOE_DP4_DOWN_WGRAD_CASE_ID,
    S3_LITE_MOE_DP4_INFER_CASE_ID,
    S3_LITE_MOE_DP4_TRAIN_FORWARD_CASE_ID,
    LiteMoeDp4IR0Adapter,
    LiteMoeDp4Oracle,
    LiteMoeDp4Spec,
    LiteMoeDp4Topology,
)
from llm.frontend.wafer_frontend.schema.lite_moe_dp4_backward import LiteMoeDp4Backward
from llm.frontend.wafer_frontend.schema.lite_moe_dp4_execution import LiteMoeDp4ExecutionCase
from llm.frontend.wafer_frontend.schema.lite_moe_dp4_train_forward import LiteMoeDp4TrainForward
from llm.frontend.wafer_frontend.schema.lite_moe_dp4_n6 import (
    LiteMoeDp4BackwardLinkedProgram,
    LiteMoeDp4BackwardLoweredProgram,
    LiteMoeDp4InferLinkedProgram,
    LiteMoeDp4InferLoweredProgram,
    LiteMoeDp4InferN6Intent,
    LiteMoeDp4TrainForwardLinkedProgram,
    LiteMoeDp4TrainForwardLoweredProgram,
)
from llm.frontend.wafer_frontend.schema.n4 import FusionPartitionContext, InterDiePlanningContext
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.policy import RegistryKind
from llm.frontend.wafer_frontend.schema.program_io import ProgramIoContract
from llm.frontend.wafer_frontend.schema.program_io import (
    ProgramHbmTarget,
    ProgramSramTarget,
)
from llm.test.frontend.integration.lite_moe_cases import (
    LiteMoeSourceCase,
    build_lite_moe_source_case,
)


_ROOT = Path(__file__).resolve().parents[4]
_HARDWARE = _ROOT / "notes/frontend/examples/hardware_2x2.json"
_MAPPING = _ROOT / "llm/test/default/mapping.spec"
_PRODUCER = "lite_moe_dp4_cases"


class LiteMoeDp4Mode(str, Enum):
    INFER = "infer"
    TRAIN_FORWARD = "train_forward"
    DOWN_WGRAD = "down_wgrad"


_CASE_IDS = {
    LiteMoeDp4Mode.INFER: S3_LITE_MOE_DP4_INFER_CASE_ID,
    LiteMoeDp4Mode.TRAIN_FORWARD: S3_LITE_MOE_DP4_TRAIN_FORWARD_CASE_ID,
    LiteMoeDp4Mode.DOWN_WGRAD: S3_LITE_MOE_DP4_DOWN_WGRAD_CASE_ID,
}


@dataclass(frozen=True, slots=True)
class LiteMoeDp4Case:
    """One exact mode over a shared, fully production pre-N6 carrier chain."""

    mode: LiteMoeDp4Mode
    case_id: str
    source: LiteMoeSourceCase
    hardware_path: Path
    mapping_path: Path
    hardware_json: str
    mapping_text: str
    spec: LiteMoeDp4Spec
    topology: LiteMoeDp4Topology
    oracle: LiteMoeDp4Oracle
    adapter: LiteMoeDp4IR0Adapter
    forward: LiteMoeDp4ExecutionCase
    train_forward: LiteMoeDp4TrainForward | None
    backward: LiteMoeDp4Backward | None

    @property
    def production(self) -> LiteMoeDp4ExecutionCase | LiteMoeDp4TrainForward | LiteMoeDp4Backward:
        if self.mode is LiteMoeDp4Mode.INFER:
            return self.forward
        if self.mode is LiteMoeDp4Mode.TRAIN_FORWARD:
            assert self.train_forward is not None
            return self.train_forward
        assert self.backward is not None
        return self.backward

    def validate(self, path: str = "lite_moe_dp4_case") -> None:
        if type(self.mode) is not LiteMoeDp4Mode or self.case_id != _CASE_IDS.get(self.mode):
            raise SchemaError("case id/mode changed", path=path)
        self.source.validate(f"{path}.source")
        if (
            self.hardware_path != _HARDWARE
            or self.mapping_path != _MAPPING
            or self.hardware_json != _HARDWARE.read_text(encoding="utf-8")
            or self.mapping_text != _MAPPING.read_text(encoding="utf-8")
        ):
            raise SchemaError("runtime hardware/mapping inputs drifted", path=path)
        hardware = json.loads(self.hardware_json)
        fabric = physical_fabric_from_data(hardware, path=f"{path}.hardware")
        if fabric.die_grid != (2, 2) or tuple(die.id for die in fabric.dies) != (0, 1, 2, 3):
            raise SchemaError("requires exact usable 2x2 fabric", path=f"{path}.hardware")
        validate_identity_mapping_text(
            self.mapping_text,
            total_cores=sum(len(die.cores) for die in fabric.dies),
            path=f"{path}.mapping_text",
        )
        self.spec.validate(f"{path}.spec")
        self.topology.validate_against(self.spec, f"{path}.topology")
        self.oracle.validate_against(self.spec, self.topology, f"{path}.oracle")
        validate_lite_moe_dp4_ir0_adapter(self.adapter, self.source.spec)
        validate_lite_moe_dp4_execution_case(self.forward)
        if (
            self.adapter.spec != self.spec
            or self.adapter.topology != self.topology
            or self.adapter.oracle != self.oracle
            or self.forward.experiment != self.source.spec
            or self.forward.adapter != self.adapter
            or self.forward.placement_context.fabric != fabric
        ):
            raise SchemaError("source-to-forward provenance changed", path=path)
        if (
            len(self.adapter.graph.nodes),
            len(self.adapter.graph.values),
            len(self.adapter.graph.edges),
            len(self.adapter.p2p_bindings),
            len(self.forward.projection.flows),
            len(self.forward.schedule.placements),
            len(self.forward.schedule.buffers),
            len(self.forward.global_dag.actions),
        ) != (44, 64, 42, 12, 12, 92, 68, 92):
            raise SchemaError("forward production quotient changed", path=path)
        if (
            sum(flow.bytes for flow in self.forward.projection.flows) != 384
            or self.oracle.logical_p2p_bytes != 384
            or self.oracle.data_packets != 24
            or self.oracle.total_expert_gemm_flops != 24576
        ):
            raise SchemaError("forward work/traffic oracle changed", path=path)
        if self.mode is LiteMoeDp4Mode.INFER:
            if self.train_forward is not None or self.backward is not None or self.spec.case_id != self.case_id:
                raise SchemaError("infer carrier identity changed", path=path)
            return
        if self.train_forward is None:
            raise SchemaError("training case lacks typed train-forward", path=path)
        validate_lite_moe_dp4_train_forward(self.train_forward, self.forward)
        if (
            self.train_forward.case_id != S3_LITE_MOE_DP4_TRAIN_FORWARD_CASE_ID
            or len(self.train_forward.tape_buffers) != 8
            or len(self.train_forward.tape_copies) != 8
            or self.train_forward.total_tape_bytes != 512
        ):
            raise SchemaError("train-forward tape quotient changed", path=path)
        if self.mode is LiteMoeDp4Mode.TRAIN_FORWARD:
            if self.backward is not None or self.case_id != self.train_forward.case_id:
                raise SchemaError("train-forward carrier identity changed", path=path)
            return
        if self.backward is None:
            raise SchemaError("backward case lacks typed backward carrier", path=path)
        validate_lite_moe_dp4_backward(self.backward, self.train_forward)
        if (
            self.case_id != self.backward.case_id
            or tuple(map(len, (
                self.backward.remote_gradients,
                self.backward.trainable_down_states,
                self.backward.token_wgrads,
                self.backward.expert_reduces,
                self.backward.sgd_stores,
            ))) != (6, 4, 8, 4, 4)
            or sum(item.bytes for item in self.backward.remote_gradients) != 192
            or sum(item.state_store_bytes for item in self.backward.sgd_stores) != 4096
        ):
            raise SchemaError("backward production quotient changed", path=path)


@dataclass(frozen=True, slots=True)
class LiteMoeDp4InferProgramIoCase:
    """Exact I-mode lower/link/ProgramIo quotient for an actual artifact SHA."""

    case: LiteMoeDp4Case
    intent: LiteMoeDp4InferN6Intent
    lowered: LiteMoeDp4InferLoweredProgram
    linked: LiteMoeDp4InferLinkedProgram
    program_io: ProgramIoContract

    def validate(self, path: str = "lite_moe_dp4_infer_program_io_case") -> None:
        self.case.validate(f"{path}.case")
        if self.case.mode is not LiteMoeDp4Mode.INFER:
            raise SchemaError("requires the infer production case", path=f"{path}.case")
        self.intent.validate(f"{path}.intent")
        self.lowered.validate(f"{path}.lowered")
        self.linked.validate(f"{path}.linked")
        self.program_io.validate_against(self.linked.manifest)
        if (
            self.lowered.source != self.case.forward
            or self.lowered.intent != self.intent
            or self.linked.source != self.lowered
            or (
                len(self.linked.manifest.fragments),
                sum(
                    len(stream.records)
                    for fragment in self.linked.manifest.fragments
                    for stream in (
                        fragment.fragment.core_streams
                        if hasattr(fragment, "fragment")
                        else fragment.core_streams
                    )
                ),
                len(self.linked.manifest.input_digests),
                len(self.linked.manifest.core_streams),
                len(self.linked.manifest.address_operand_bindings),
                len(self.linked.manifest.state_operand_bindings),
                len(self.linked.manifest.runtime_symbol_definitions),
                len(self.linked.manifest.program_symbol_definitions),
            )
            != (80, 260, 85, 4, 404, 24, 52, 149)
        ):
            raise SchemaError("infer lower/link quotient changed", path=path)
        if (
            len(self.program_io.blobs),
            len(self.program_io.initializations),
            len(self.program_io.output_probes),
            sum(item.length_bytes for item in self.program_io.initializations),
            sum(item.length_bytes for item in self.program_io.output_probes),
        ) != (16, 80, 8, 39296, 256):
            raise SchemaError("infer ProgramIo quotient changed", path=path)


@dataclass(frozen=True, slots=True)
class LiteMoeDp4TrainForwardProgramIoCase:
    """Exact TF lower/link/ProgramIo quotient for an actual artifact SHA."""

    case: LiteMoeDp4Case
    lowered: LiteMoeDp4TrainForwardLoweredProgram
    linked: LiteMoeDp4TrainForwardLinkedProgram
    program_io: ProgramIoContract

    def validate(self, path: str = "lite_moe_dp4_train_forward_program_io_case") -> None:
        self.case.validate(f"{path}.case")
        if (
            self.case.mode is not LiteMoeDp4Mode.TRAIN_FORWARD
            or self.case.train_forward is None
        ):
            raise SchemaError("requires the train-forward production case", path=f"{path}.case")
        self.lowered.validate(f"{path}.lowered")
        self.linked.validate(f"{path}.linked")
        self.program_io.validate_against(self.linked.manifest)
        manifest = self.linked.manifest
        if (
            self.lowered.source != self.case.train_forward
            or self.linked.source != self.lowered
            or (
                len(manifest.fragments),
                sum(
                    len(stream.records)
                    for fragment in manifest.fragments
                    for stream in (
                        fragment.fragment.core_streams
                        if hasattr(fragment, "fragment")
                        else fragment.core_streams
                    )
                ),
                len(manifest.input_digests),
                len(manifest.core_streams),
                len(manifest.address_operand_bindings),
                len(manifest.state_operand_bindings),
                len(manifest.runtime_symbol_definitions),
                len(manifest.program_symbol_definitions),
            )
            != (88, 284, 93, 4, 436, 24, 60, 165)
        ):
            raise SchemaError("train-forward lower/link quotient changed", path=path)
        probe_sizes = tuple(sorted(item.length_bytes for item in self.program_io.output_probes))
        if (
            len(self.program_io.blobs),
            len(self.program_io.initializations),
            len(self.program_io.output_probes),
            sum(item.length_bytes for item in self.program_io.initializations),
            sum(item.length_bytes for item in self.program_io.output_probes),
        ) != (16, 88, 16, 39808, 768) or probe_sizes != (32,) * 8 + (64,) * 8:
            raise SchemaError(
                "train-forward ProgramIo must probe eight combined and eight tape roots",
                path=path,
            )


@dataclass(frozen=True, slots=True)
class LiteMoeDp4BackwardProgramIoCase:
    """Exact TB lower/link/ProgramIo quotient for an actual artifact SHA."""

    case: LiteMoeDp4Case
    lowered: LiteMoeDp4BackwardLoweredProgram
    linked: LiteMoeDp4BackwardLinkedProgram
    program_io: ProgramIoContract

    def validate(self, path: str = "lite_moe_dp4_backward_program_io_case") -> None:
        self.case.validate(f"{path}.case")
        if (
            self.case.mode is not LiteMoeDp4Mode.DOWN_WGRAD
            or self.case.backward is None
        ):
            raise SchemaError("requires the backward production case", path=f"{path}.case")
        self.lowered.validate(f"{path}.lowered")
        self.linked.validate(f"{path}.linked")
        self.program_io.validate_against(self.linked.manifest)
        manifest = self.linked.manifest
        if (
            self.lowered.source != self.case.backward
            or self.linked.source != self.lowered
            or (
                len(manifest.fragments),
                sum(
                    len(stream.records)
                    for fragment in manifest.fragments
                    for stream in (
                        fragment.fragment.core_streams
                        if hasattr(fragment, "fragment")
                        else fragment.core_streams
                    )
                ),
                len(manifest.input_digests),
                len(manifest.core_streams),
                len(manifest.address_operand_bindings),
                len(manifest.state_operand_bindings),
                len(manifest.runtime_symbol_definitions),
                len(manifest.program_symbol_definitions),
            )
            != (32, 114, 37, 4, 190, 8, 28, 73)
        ):
            raise SchemaError("backward lower/link quotient changed", path=path)
        sram_initializations = tuple(
            item for item in self.program_io.initializations
            if type(item.target) is ProgramSramTarget
        )
        hbm_initializations = tuple(
            item for item in self.program_io.initializations
            if type(item.target) is ProgramHbmTarget
        )
        probes = self.program_io.output_probes
        if (
            len(self.program_io.blobs),
            len(self.program_io.initializations),
            len(sram_initializations),
            len(hbm_initializations),
            len(probes),
            sum(item.length_bytes for item in self.program_io.initializations),
            sum(item.length_bytes for item in probes),
        ) != (8, 34, 30, 4, 4, 25536, 4096) or any(
            type(item.target) is not ProgramHbmTarget
            or item.length_bytes != 1024
            for item in probes
        ):
            raise SchemaError(
                "backward ProgramIo must probe only four updated HBM weights",
                path=path,
            )


def _build_common() -> tuple[
    LiteMoeSourceCase, str, str, LiteMoeDp4Spec, LiteMoeDp4Topology,
    LiteMoeDp4Oracle, LiteMoeDp4IR0Adapter, LiteMoeDp4ExecutionCase,
]:
    source = build_lite_moe_source_case()
    spec = build_lite_moe_dp4_spec(source.moe_spec.trace)
    topology = build_lite_moe_dp4_topology(spec)
    oracle = build_lite_moe_dp4_oracle(spec, topology)
    adapter = build_lite_moe_dp4_ir0_adapter(source.spec, spec, topology, oracle)
    hardware_json = _HARDWARE.read_text(encoding="utf-8")
    mapping_text = _MAPPING.read_text(encoding="utf-8")
    hardware = json.loads(hardware_json)
    placement = PlacementContext.create(
        producer_pass=_PRODUCER,
        fabric=physical_fabric_from_data(hardware),
        placement=source.spec.placement,
        hbm_address_spaces=hbm_address_spaces_from_data(hardware),
    )
    registry = production_registry()
    partition = FusionPartitionContext.create(producer_pass=_PRODUCER)
    planning = InterDiePlanningContext.create(
        producer_pass=_PRODUCER,
        fused_policy=registry.instantiate(RegistryKind.INTER_DIE, "naive").selection,
        standalone_policy=registry.instantiate(
            RegistryKind.STANDALONE_COLLECTIVE, "direct_all_gather"
        ).selection,
    )
    forward = build_lite_moe_dp4_execution_case(
        source.spec, adapter, placement, partition, planning
    )
    return source, hardware_json, mapping_text, spec, topology, oracle, adapter, forward


def build_lite_moe_dp4_case(mode: LiteMoeDp4Mode) -> LiteMoeDp4Case:
    """Build one exact mode exclusively through public production passes."""

    if type(mode) is not LiteMoeDp4Mode:
        raise SchemaError("mode must be typed", path="lite_moe_dp4_case.mode")
    source, hardware_json, mapping_text, spec, topology, oracle, adapter, forward = _build_common()
    train_forward = None
    backward = None
    if mode in (LiteMoeDp4Mode.TRAIN_FORWARD, LiteMoeDp4Mode.DOWN_WGRAD):
        train_forward = build_lite_moe_dp4_train_forward(forward)
    if mode is LiteMoeDp4Mode.DOWN_WGRAD:
        assert train_forward is not None
        backward = build_lite_moe_dp4_backward(train_forward)
    result = LiteMoeDp4Case(
        mode, _CASE_IDS[mode], source, _HARDWARE, _MAPPING, hardware_json,
        mapping_text, spec, topology, oracle, adapter, forward, train_forward, backward,
    )
    result.validate()
    return result


def build_lite_moe_dp4_cases() -> tuple[LiteMoeDp4Case, ...]:
    """Return the canonical I/TF/TB matrix in fixed order."""

    return tuple(build_lite_moe_dp4_case(mode) for mode in LiteMoeDp4Mode)


def build_lite_moe_dp4_infer_program_io_case(
    program_artifact_sha256: str,
) -> LiteMoeDp4InferProgramIoCase:
    """Build I mode through production lower/link and actual-SHA ProgramIo."""

    case = build_lite_moe_dp4_case(LiteMoeDp4Mode.INFER)
    lowered = lower_lite_moe_dp4_infer_program(case.forward)
    linked = link_lite_moe_dp4_infer_program(lowered)
    seeds, expected = build_deterministic_timing_state_overrides(linked)
    program_io = build_timing_program_io(
        linked,
        program_artifact_sha256,
        state_seed_overrides=seeds,
        state_expected_overrides=expected,
    )
    result = LiteMoeDp4InferProgramIoCase(
        case, lowered.intent, lowered, linked, program_io
    )
    result.validate()
    return result


def build_lite_moe_dp4_train_forward_program_io_case(
    program_artifact_sha256: str,
) -> LiteMoeDp4TrainForwardProgramIoCase:
    """Build TF through production lower/link and actual-SHA ProgramIo."""

    case = build_lite_moe_dp4_case(LiteMoeDp4Mode.TRAIN_FORWARD)
    assert case.train_forward is not None
    lowered = lower_lite_moe_dp4_train_forward_program(case.train_forward)
    linked = link_lite_moe_dp4_train_forward_program(lowered)
    seeds, expected = build_deterministic_timing_state_overrides(linked)
    program_io = build_timing_program_io(
        linked,
        program_artifact_sha256,
        state_seed_overrides=seeds,
        state_expected_overrides=expected,
    )
    result = LiteMoeDp4TrainForwardProgramIoCase(
        case, lowered, linked, program_io
    )
    result.validate()
    return result


def build_lite_moe_dp4_backward_program_io_case(
    program_artifact_sha256: str,
) -> LiteMoeDp4BackwardProgramIoCase:
    """Build TB through production lower/link and actual-SHA ProgramIo."""

    case = build_lite_moe_dp4_case(LiteMoeDp4Mode.DOWN_WGRAD)
    assert case.backward is not None
    lowered = lower_lite_moe_dp4_backward_program(case.backward)
    linked = link_lite_moe_dp4_backward_program(lowered)
    seeds, expected = build_deterministic_timing_state_overrides(linked)
    program_io = build_timing_program_io(
        linked,
        program_artifact_sha256,
        state_seed_overrides=seeds,
        state_expected_overrides=expected,
    )
    result = LiteMoeDp4BackwardProgramIoCase(
        case, lowered, linked, program_io
    )
    result.validate()
    return result


__all__ = [
    "LiteMoeDp4Case",
    "LiteMoeDp4Mode",
    "LiteMoeDp4InferProgramIoCase",
    "LiteMoeDp4BackwardProgramIoCase",
    "LiteMoeDp4TrainForwardProgramIoCase",
    "build_lite_moe_dp4_case",
    "build_lite_moe_dp4_cases",
    "build_lite_moe_dp4_infer_program_io_case",
    "build_lite_moe_dp4_backward_program_io_case",
    "build_lite_moe_dp4_train_forward_program_io_case",
]
