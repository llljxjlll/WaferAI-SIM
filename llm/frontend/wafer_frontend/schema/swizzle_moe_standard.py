"""Strict standard-artifact wrapper for the MoE Swizzle lowering path."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .artifact_manifest import CommandFragment, LinkedProgramManifest
from .common import stable_artifact_id
from .ir1 import IR1
from .program_io import ProgramIoContract
from .swizzle_moe import (
    MoeHardwareFacts, MoeSwizzleDecision, MoeSwizzleWorkloadSelection,
)
from .swizzle_moe_abi import MoeSwizzleCoreAddressABI
from .swizzle_moe_execution import MoeScaleExecution
from .swizzle_moe_ir2 import MoeSwizzleIr2Projection
from .swizzle_moe_operand_abi import MoeSwizzleOperandABI
from .swizzle_moe_plan import MoeSwizzleOverlay
from .swizzle_moe_scale import MoeSwizzleScaleOracle, MoeSwizzleScaleSpec
from .swizzle_moe_state import MoeSwizzleWorkloadStateABI
from .swizzle_moe_workload import MoeSwizzleWorkloadProjection
from .swizzle_moe_workload_abi import MoeSwizzleWorkloadABI
from .swizzle_moe_workload_bridge import MoeSwizzleWorkloadValueBridge


MOE_SWIZZLE_STANDARD_LINKED_PROGRAM_SCHEMA_VERSION = (
    "wafer_frontend.moe_swizzle_standard_linked_program/v1alpha1"
)


@dataclass(frozen=True, slots=True)
class MoeSwizzleStandardLinkedProgram:
    schema_version: str
    producer_pass: str
    id: str
    ir1: IR1
    spec: MoeSwizzleScaleSpec
    oracle: MoeSwizzleScaleOracle
    execution: MoeScaleExecution
    decisions: tuple[MoeSwizzleDecision, MoeSwizzleDecision]
    selection: MoeSwizzleWorkloadSelection
    overlay: MoeSwizzleOverlay
    workload: MoeSwizzleWorkloadProjection
    projection: MoeSwizzleIr2Projection
    core_abi: MoeSwizzleCoreAddressABI
    operand_abi: MoeSwizzleOperandABI
    state_abi: MoeSwizzleWorkloadStateABI
    value_bridge: MoeSwizzleWorkloadValueBridge
    workload_abi: MoeSwizzleWorkloadABI
    hardware_facts: MoeHardwareFacts
    fragment: CommandFragment
    manifest: LinkedProgramManifest
    program_io: ProgramIoContract | None

    @classmethod
    def create(cls, **semantic: object) -> "MoeSwizzleStandardLinkedProgram":
        result = cls(
            schema_version=MOE_SWIZZLE_STANDARD_LINKED_PROGRAM_SCHEMA_VERSION,
            producer_pass="moe_swizzle_standard_linker",
            id=stable_artifact_id(
                "moe_swizzle_standard_linked_program",
                semantic,
                schema_version=MOE_SWIZZLE_STANDARD_LINKED_PROGRAM_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "ir1", "spec", "oracle", "execution", "decisions", "selection",
                "overlay", "workload", "projection", "core_abi", "operand_abi",
                "state_abi", "value_bridge", "workload_abi", "hardware_facts",
                "fragment", "manifest", "program_io",
            )
        }

    def validate(self, path: str = "moe_swizzle_standard_linked_program") -> None:
        if (
            self.schema_version != MOE_SWIZZLE_STANDARD_LINKED_PROGRAM_SCHEMA_VERSION
            or self.producer_pass != "moe_swizzle_standard_linker"
        ):
            raise SchemaError("unsupported MoE standard wrapper schema/producer", path=path)
        self.ir1.validate(f"{path}.ir1")
        self.execution.validate_against(self.spec, self.oracle, f"{path}.execution")
        if len(self.decisions) != 2 or len({item.id for item in self.decisions}) != 2:
            raise SchemaError("wrapper requires exactly two decisions", path=f"{path}.decisions")
        for index, decision in enumerate(self.decisions):
            decision.validate(f"{path}.decisions[{index}]")
            if decision.problem.source_execution_id != self.execution.id:
                raise SchemaError("decision belongs to another execution", path=f"{path}.decisions[{index}]")
        self.selection.validate(f"{path}.selection")
        self.overlay.validate(f"{path}.overlay")
        self.workload.validate(f"{path}.workload")
        self.projection.validate(f"{path}.projection")
        self.core_abi.validate(f"{path}.core_abi")
        self.operand_abi.validate(f"{path}.operand_abi")
        self.state_abi.validate(f"{path}.state_abi")
        self.value_bridge.validate(f"{path}.value_bridge")
        self.workload_abi.validate(f"{path}.workload_abi")
        self.hardware_facts.validate(f"{path}.hardware_facts")
        self.fragment.validate(f"{path}.fragment")
        self.manifest.validate(f"{path}.manifest")
        if self.program_io is not None:
            self.program_io.validate(f"{path}.program_io")
        projected_actions = tuple(sorted(item.id for item in self.workload.actions))
        if (
            self.overlay.source_execution_id != self.execution.id
            or self.overlay.source_workload_selection_id != self.selection.id
            or {
                self.selection.source_dispatch_decision_id,
                self.selection.source_combine_decision_id,
            } != {item.id for item in self.decisions}
            or self.workload.source_overlay_id != self.overlay.id
            or self.workload.replacement_projection_id != self.projection.id
            or self.projection.source_overlay_id != self.overlay.id
            or self.core_abi.source_projection_id != self.projection.id
            or self.core_abi.source_workload_projection_id != self.workload.id
            or self.operand_abi.source_projection_id != self.projection.id
            or self.fragment.source_global_dag_id != self.workload.id
            or self.fragment.claimed_action_ids != projected_actions
            or self.manifest.source_ir1_id != self.ir1.id
            or self.manifest.source_projection_id != self.projection.id
            or self.manifest.source_schedule_set_id != self.core_abi.id
            or self.manifest.source_global_dag_id != self.workload.id
            or len(self.manifest.fragments) != 1
            or self.manifest.fragments[0].id != self.fragment.id
        ):
            raise SchemaError("standard fragment/manifest provenance is not exact", path=path)
        if (
            self.program_io is not None
            and self.program_io.source_linked_manifest_id != self.manifest.id
        ):
            raise SchemaError("ProgramIo belongs to another linked manifest", path=f"{path}.program_io")
        originals = tuple(ref for item in self.workload.actions for ref in item.source_action_refs)
        if (
            len(originals) != len(set(originals))
            or set(originals) != {item.id for item in self.execution.actions}
        ):
            raise SchemaError("whole standard wrapper does not exactly cover execution actions", path=path)
        from ..passes.build_moe_swizzle_program_io import (
            validate_moe_swizzle_program_io_against,
        )
        from ..lowering.moe_swizzle_abi import allocate_moe_swizzle_core_address_abi
        from ..lowering.moe_swizzle_workload_abi import build_moe_swizzle_workload_abi
        from ..lowering.moe_swizzle_workload_linker import link_moe_swizzle_workload_manifest
        from ..lowering.moe_swizzle_workload_standard import lower_moe_swizzle_workload_fragment
        from ..passes.build_moe_scale_swizzle_overlay import build_moe_scale_swizzle_overlay
        from ..passes.build_moe_swizzle_workload_state_abi import build_moe_swizzle_workload_state_abi
        from ..passes.build_moe_swizzle_workload_value_bridge import build_moe_swizzle_workload_value_bridge
        from ..passes.project_moe_scale_swizzle_ir2 import project_moe_scale_swizzle_ir2
        from ..passes.project_moe_swizzle_whole_workload import project_moe_swizzle_whole_workload
        from ..passes.schedule_moe_swizzle_workload_endpoints import schedule_moe_swizzle_workload_endpoints
        from ..passes.schedule_moe_swizzle_workload_storage_reuse import schedule_moe_swizzle_workload_storage_reuse
        from .swizzle_moe_placement import build_moe_swizzle_workload_placement
        from .swizzle_moe_operand_abi import build_moe_swizzle_operand_abi
        expected_overlay = build_moe_scale_swizzle_overlay(
            self.execution, self.decisions, self.selection,
        )
        expected_projection = project_moe_scale_swizzle_ir2(
            expected_overlay, self.execution, self.spec, self.decisions,
            self.selection,
            endpoint_session_capacity=self.projection.endpoint_session_capacity,
        )
        expected_state = build_moe_swizzle_workload_state_abi(
            self.ir1, self.execution, self.spec, self.oracle,
        )
        expected_workload = project_moe_swizzle_whole_workload(
            expected_overlay, self.execution, expected_projection, expected_state,
        )
        expected_placement = build_moe_swizzle_workload_placement(
            self.ir1, expected_workload, expected_projection,
            self.hardware_facts,
        )
        expected_workload = schedule_moe_swizzle_workload_endpoints(
            expected_workload, expected_projection, expected_placement,
            capacity_per_core=expected_projection.endpoint_session_capacity,
        )
        expected_placement = build_moe_swizzle_workload_placement(
            self.ir1, expected_workload, expected_projection,
            self.hardware_facts,
        )
        expected_bridge = build_moe_swizzle_workload_value_bridge(
            self.execution, expected_workload, expected_projection,
        )
        expected_workload = schedule_moe_swizzle_workload_storage_reuse(
            expected_workload, expected_projection, expected_bridge,
            expected_placement,
        )
        expected_bridge = build_moe_swizzle_workload_value_bridge(
            self.execution, expected_workload, expected_projection,
        )
        expected_core = allocate_moe_swizzle_core_address_abi(
            self.ir1, expected_projection, hardware_facts=self.hardware_facts,
            workload_projection=expected_workload,
        )
        expected_operand = build_moe_swizzle_operand_abi(expected_projection)
        expected_workload_abi = build_moe_swizzle_workload_abi(
            self.ir1, expected_workload, expected_projection, expected_state,
            expected_bridge, self.hardware_facts,
        )
        expected_fragment = lower_moe_swizzle_workload_fragment(
            self.ir1, self.execution, expected_workload, expected_projection,
            expected_core, expected_operand, expected_state, expected_bridge,
            expected_workload_abi, self.hardware_facts,
        )
        if (
            (self.overlay, self.workload, self.projection, self.core_abi,
             self.operand_abi, self.state_abi, self.value_bridge,
             self.workload_abi, self.fragment)
            != (expected_overlay, expected_workload, expected_projection,
                expected_core, expected_operand, expected_state,
                expected_bridge, expected_workload_abi, expected_fragment)
            or self.manifest != link_moe_swizzle_workload_manifest(
                self.ir1, self.spec, self.oracle, self.execution, self.decisions,
                self.selection, expected_overlay, expected_workload,
                expected_projection, expected_core, expected_operand,
                expected_state, expected_bridge, expected_workload_abi,
                self.hardware_facts, expected_fragment,
            )
        ):
            raise SchemaError("whole standard wrapper is not the deterministic rebuild", path=path)
        if self.program_io is not None:
            validate_moe_swizzle_program_io_against(
                self.program_io, self, f"{path}.program_io",
            )
        expected = stable_artifact_id(
            "moe_swizzle_standard_linked_program",
            self._semantic(),
            schema_version=MOE_SWIZZLE_STANDARD_LINKED_PROGRAM_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable MoE standard wrapper id", path=f"{path}.id")


__all__ = [
    "MOE_SWIZZLE_STANDARD_LINKED_PROGRAM_SCHEMA_VERSION",
    "MoeSwizzleStandardLinkedProgram",
]
