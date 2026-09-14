from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.flexible_moe_standard import (
    build_flexible_moe_standard_program_io_plan,
    plan_flexible_moe_standard_mapping,
)
from llm.frontend.wafer_frontend.passes.flexible_moe import (
    build_round_robin_flexible_moe_spec,
    compile_flexible_moe_baseline,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import RecordOpcode
from llm.frontend.wafer_frontend.schema.flexible_moe import (
    FlexibleMoeMode,
    MoeRectActionKind,
    MoeRectFlowStage,
    MoeRectStateRole,
)
from llm.frontend.wafer_frontend.schema.flexible_moe_standard import (
    FlexibleMoeIoRole,
    FlexibleMoeStateAccess,
)
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec


def _case(rows: int, columns: int, mode: FlexibleMoeMode):
    ranks = rows * columns
    spec = build_round_robin_flexible_moe_spec(
        RectMeshSpec(rows, columns),
        mode,
        routing_shift=0 if ranks == 1 else 1,
    )
    plan = compile_flexible_moe_baseline(spec)
    mapping = plan_flexible_moe_standard_mapping(plan, spec)
    program_io = build_flexible_moe_standard_program_io_plan(
        mapping, plan, spec, "0" * 64
    )
    return spec, plan, mapping, program_io


class FlexibleMoeStandardTest(unittest.TestCase):
    def test_representative_infer_train_mapping_and_program_io_plans(self) -> None:
        for mode in FlexibleMoeMode:
            for rows, columns in ((1, 1), (1, 2), (2, 2)):
                with self.subTest(mode=mode.value, mesh=(rows, columns)):
                    spec, plan, mapping, program_io = _case(rows, columns, mode)
                    self.assertEqual(
                        tuple(item.runtime_core_id for item in mapping.core_streams),
                        tuple(range(spec.mesh.rank_count)),
                    )
                    self.assertEqual(mapping.record_count, len(plan.actions))
                    self.assertEqual(len(mapping.endpoint_abi), len(plan.flows))
                    self.assertEqual(len(mapping.state_abi), len(plan.state_bindings))
                    self.assertTrue(mapping.standard_mapping_verified)
                    self.assertFalse(mapping.lower_link_verified)
                    self.assertFalse(mapping.runtime_verified)
                    self.assertTrue(program_io.timing_execution)
                    self.assertFalse(program_io.runtime_verified)
                    mapping.validate_against(plan, spec)
                    program_io.validate_against(mapping, plan, spec)

    def test_every_flow_has_public_send_recv_wait_records_and_dag_deps(self) -> None:
        spec, plan, mapping, _ = _case(2, 2, FlexibleMoeMode.TRAIN)
        records = tuple(
            item for stream in mapping.core_streams for item in stream.records
        )
        by_action = {item.source_action_ref: item for item in records}
        action_index = {item.id: item for item in plan.actions}
        for action in plan.actions:
            self.assertEqual(
                by_action[action.id].dependency_record_refs,
                tuple(by_action[ref].id for ref in action.deps),
            )
        self.assertEqual(set(item.stage for item in plan.flows), set(MoeRectFlowStage))
        for endpoint in mapping.endpoint_abi:
            flow_records = {
                item.opcode: item for item in records if item.flow_ref == endpoint.flow_ref
            }
            self.assertEqual(
                set(flow_records),
                {RecordOpcode.DTE_SEND, RecordOpcode.DTE_RECV, RecordOpcode.DTE_WAIT},
            )
            self.assertEqual(
                (
                    endpoint.send_record_ref,
                    endpoint.recv_record_ref,
                    endpoint.wait_record_ref,
                ),
                (
                    flow_records[RecordOpcode.DTE_SEND].id,
                    flow_records[RecordOpcode.DTE_RECV].id,
                    flow_records[RecordOpcode.DTE_WAIT].id,
                ),
            )
        for action in plan.actions:
            if action.kind is MoeRectActionKind.STATE_STORE:
                self.assertEqual(len(action.state_refs), 1)
                self.assertEqual(len(action.deps), 1)
                self.assertIn(
                    action_index[action.deps[0]].kind,
                    {MoeRectActionKind.EXPERT_SGD, MoeRectActionKind.GATE_SGD},
                )

    def test_state_abi_and_program_io_cover_parameters_without_aliasing(self) -> None:
        spec, plan, mapping, program_io = _case(1, 2, FlexibleMoeMode.TRAIN)
        role_by_ref = {item.id: item.role for item in plan.state_bindings}
        for state in mapping.state_abi:
            role = role_by_ref[state.state_ref]
            expected = (
                FlexibleMoeStateAccess.SCRATCH
                if role in (MoeRectStateRole.EXPERT_GRADIENT, MoeRectStateRole.GATE_GRADIENT)
                else FlexibleMoeStateAccess.READ_WRITE
            )
            self.assertIs(state.access, expected)
            self.assertEqual(state.address % 64, 0)
        initialized = {
            item.state_ref for item in program_io.entries
            if item.role is FlexibleMoeIoRole.STATE_INITIALIZATION
        }
        updated = {
            item.state_ref for item in program_io.entries
            if item.role is FlexibleMoeIoRole.UPDATED_STATE_PROBE
        }
        persistent = {item.state_ref for item in mapping.state_abi if item.persistent}
        self.assertEqual(initialized, persistent)
        self.assertEqual(updated, persistent)

    def test_mapping_cannot_claim_link_or_runtime_and_is_repeatable(self) -> None:
        spec, plan, first, first_io = _case(1, 2, FlexibleMoeMode.INFERENCE)
        second = plan_flexible_moe_standard_mapping(plan, spec)
        second_io = build_flexible_moe_standard_program_io_plan(
            second, plan, spec, "0" * 64
        )
        self.assertEqual(first, second)
        self.assertEqual(first_io, second_io)
        with self.assertRaisesRegex(SchemaError, "cannot claim lower/link or runtime"):
            replace(first, lower_link_verified=True).validate_against(plan, spec)
        missing = replace(first_io, entries=first_io.entries[1:])
        with self.assertRaisesRegex(SchemaError, "initialize every persistent state"):
            missing.validate_against(first, plan, spec)


if __name__ == "__main__":
    unittest.main()
