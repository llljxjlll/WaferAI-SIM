from __future__ import annotations

from collections import Counter
from dataclasses import replace
from functools import lru_cache
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.flexible_dense_backward import (
    build_flexible_dense_backward_program_io,
    materialize_flexible_dense_backward,
)
from llm.frontend.wafer_frontend.passes.flexible_dense_backward_multi import (
    _gradient_carrier_bytes,
)
from llm.frontend.wafer_frontend.passes.flexible_dense_train import (
    materialize_flexible_dense_train_forward,
)
from llm.frontend.wafer_frontend.passes.load_fabric import (
    hbm_address_spaces_from_data,
    physical_fabric_from_data,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    ManifestInputKind,
    ProgramSymbolKind,
    RecordOpcode,
)
from llm.frontend.wafer_frontend.schema.flexible_dense_backward import (
    FlexibleDenseBackwardLinkedProgram,
)
from llm.frontend.wafer_frontend.schema.flexible_dense_train import (
    FlexibleDenseTrainActionKind,
    FlexibleDenseTrainGradientSyncRole,
)
from llm.frontend.wafer_frontend.schema.persistent_state import (
    PersistentStateAccess,
    StateKind,
)
from llm.frontend.wafer_frontend.schema.program_io import (
    ProgramIoTargetKind,
)
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    loads_dataclass,
)
from llm.test.frontend.unit.test_flexible_dense_train import _hardware, _spec


_ZERO_SHA = "0" * 64


@lru_cache(maxsize=1)
def _linked() -> FlexibleDenseBackwardLinkedProgram:
    raw = _hardware(1, 1)
    fabric = physical_fabric_from_data(raw)
    spaces = hbm_address_spaces_from_data(raw)
    forward = materialize_flexible_dense_train_forward(
        _spec(1, 1), RectMeshSpec(1, 1), fabric, spaces
    )
    return materialize_flexible_dense_backward(
        forward,
        fabric,
        spaces,
    )


@lru_cache(maxsize=8)
def _linked_shape(rows: int, columns: int) -> FlexibleDenseBackwardLinkedProgram:
    raw = _hardware(rows, columns)
    fabric = physical_fabric_from_data(raw)
    spaces = hbm_address_spaces_from_data(raw)
    forward = materialize_flexible_dense_train_forward(
        _spec(rows, columns), RectMeshSpec(rows, columns), fabric, spaces
    )
    return materialize_flexible_dense_backward(forward, fabric, spaces)


class FlexibleDenseBackwardTest(unittest.TestCase):
    def test_one_by_one_real_manifest_covers_every_parameter(self) -> None:
        linked = _linked()
        linked.validate()
        records = tuple(
            record
            for fragment in linked.manifest.fragments
            for stream in fragment.core_streams
            for record in stream.records
        )
        counts = Counter(item.opcode for item in records)
        states = tuple(
            item
            for fragment in linked.manifest.fragments
            for item in fragment.state_abi
        )
        parameter_count = len(linked.plan.parameter_templates)
        backward_count = len(linked.plan.tape_bindings)
        self.assertEqual(linked.record_count, len(records))
        self.assertEqual(len(states), parameter_count)
        self.assertTrue(all(
            item.kind is StateKind.TRAINABLE_PARAMETER
            and item.access is PersistentStateAccess.READ_WRITE
            for item in states
        ))
        self.assertEqual(counts[RecordOpcode.SRAM_ALLOC_AT], 2 * parameter_count)
        self.assertEqual(
            counts[RecordOpcode.SRAM_BIND],
            backward_count + 2 * parameter_count,
        )
        self.assertEqual(counts[RecordOpcode.SRAM_FREE], 2 * parameter_count)
        self.assertEqual(counts[RecordOpcode.LSU_LOAD], parameter_count)
        self.assertEqual(
            counts[RecordOpcode.MATMUL], backward_count + parameter_count
        )
        self.assertEqual(counts[RecordOpcode.SGD_UPDATE], parameter_count)
        self.assertEqual(counts[RecordOpcode.LSU_STORE], parameter_count)
        region_names = {
            item.name
            for item in linked.manifest.program_symbol_definitions
            if item.symbol.kind is ProgramSymbolKind.SRAM_REGION
        }
        self.assertEqual(
            region_names,
            {linked.fabric.sram_profiles[0].regions[0].name},
        )

    def test_backward_four_stage_lineage_is_exact_but_does_not_overclaim(self) -> None:
        linked = _linked()
        backward_inputs = tuple(
            item
            for item in linked.manifest.input_digests
            if item.kind is not ManifestInputKind.COMMAND_FRAGMENT
        )
        self.assertEqual(
            {item.kind for item in backward_inputs},
            {
                ManifestInputKind.FLEXIBLE_DENSE_BACKWARD_IR,
                ManifestInputKind.FLEXIBLE_DENSE_BACKWARD_PROJECTION,
                ManifestInputKind.FLEXIBLE_DENSE_BACKWARD_SCHEDULE,
                ManifestInputKind.FLEXIBLE_DENSE_BACKWARD_GLOBAL,
            },
        )
        self.assertEqual(linked.manifest.source_ir1_id, linked.backward_ir.id)
        self.assertEqual(
            linked.manifest.source_projection_id,
            linked.backward_projection.id,
        )
        self.assertEqual(
            linked.manifest.source_schedule_set_id,
            linked.backward_schedule.id,
        )
        self.assertEqual(
            linked.manifest.source_global_dag_id,
            linked.backward_global_dag.id,
        )
        self.assertFalse(linked.runtime_verified)

    def test_program_io_initializes_and_probes_every_updated_state(self) -> None:
        linked = _linked()
        program_io = build_flexible_dense_backward_program_io(
            linked, _ZERO_SHA
        )
        program_io.validate_against(linked.manifest)
        self.assertEqual(
            len(program_io.initializations),
            len(linked.plan.parameter_templates),
        )
        self.assertEqual(
            len(program_io.output_probes),
            len(linked.plan.parameter_templates),
        )

    def test_stable_round_trip_and_runtime_overclaim_fail_closed(self) -> None:
        linked = _linked()
        self.assertEqual(_linked().id, linked.id)
        self.assertEqual(
            loads_dataclass(
                FlexibleDenseBackwardLinkedProgram,
                canonical_json(linked),
            ),
            linked,
        )
        with self.assertRaisesRegex(SchemaError, "not runtime evidence"):
            replace(linked, runtime_verified=True).validate()

    def test_gradient_carrier_covers_logical_payload_and_fails_closed(self) -> None:
        self.assertEqual(_gradient_carrier_bytes(128, 1), 128)
        self.assertEqual(_gradient_carrier_bytes(128, 2), 4096)
        self.assertEqual(_gradient_carrier_bytes(12800, 10), 12800)
        for logical_bytes, dp_degree in ((0, 1), (-1, 1), (128, 0)):
            with self.subTest(logical_bytes=logical_bytes, dp_degree=dp_degree):
                with self.assertRaises(SchemaError):
                    _gradient_carrier_bytes(logical_bytes, dp_degree)

    def test_representative_rectangles_close_rank_state_and_dp_transport(self) -> None:
        for rows, columns in ((1, 2), (2, 1), (2, 2), (2, 3), (3, 2)):
            with self.subTest(rows=rows, columns=columns):
                linked = _linked_shape(rows, columns)
                ranks = rows * columns
                fragment = linked.manifest.fragments[0]
                self.assertEqual(len(linked.manifest.core_bindings), ranks)
                self.assertEqual(len(linked.manifest.core_streams), ranks)
                self.assertEqual(len(fragment.core_streams), ranks)
                expected_states = sum(
                    len(template.owner_ranks)
                    for template in linked.plan.parameter_templates
                )
                self.assertEqual(len(fragment.state_abi), expected_states)
                sync_count = sum(
                    action.kind is FlexibleDenseTrainActionKind.GRADIENT_SYNC
                    for action in linked.plan.rank_actions
                )
                counts = Counter(
                    record.opcode
                    for stream in fragment.core_streams
                    for record in stream.records
                )
                send_count = sum(
                    action.gradient_sync_role in (
                        FlexibleDenseTrainGradientSyncRole.REDUCE_SEND,
                        FlexibleDenseTrainGradientSyncRole.BROADCAST_SEND,
                    )
                    for action in linked.plan.rank_actions
                )
                receive_count = sum(
                    action.gradient_sync_role in (
                        FlexibleDenseTrainGradientSyncRole.REDUCE_RECEIVE,
                        FlexibleDenseTrainGradientSyncRole.BROADCAST_RECEIVE,
                    )
                    for action in linked.plan.rank_actions
                )
                reduce_count = sum(
                    action.gradient_sync_role
                    is FlexibleDenseTrainGradientSyncRole.REDUCE_RECEIVE
                    for action in linked.plan.rank_actions
                )
                self.assertEqual(counts[RecordOpcode.DTE_SEND], send_count)
                self.assertEqual(counts[RecordOpcode.DTE_RECV], receive_count)
                self.assertEqual(counts[RecordOpcode.DTE_WAIT], receive_count)
                self.assertEqual(counts[RecordOpcode.LOCAL_REDUCE], reduce_count)
                self.assertEqual(sync_count, send_count + receive_count)
                if rows == 1:
                    self.assertEqual(sync_count, 0)
                    self.assertEqual(linked.manifest.runtime_symbol_definitions, ())
                else:
                    self.assertGreater(sync_count, 0)
                program_io = build_flexible_dense_backward_program_io(
                    linked, _ZERO_SHA
                )
                program_io.validate_against(linked.manifest)
                hbm_initializations = tuple(
                    item for item in program_io.initializations
                    if item.target.kind is ProgramIoTargetKind.HBM
                )
                sram_initializations = tuple(
                    item for item in program_io.initializations
                    if item.target.kind is ProgramIoTargetKind.SRAM
                )
                self.assertEqual(len(hbm_initializations), expected_states)
                self.assertEqual(len(program_io.output_probes), expected_states)
                self.assertEqual(
                    len(sram_initializations),
                    0 if rows == 1 else expected_states,
                )


if __name__ == "__main__":
    unittest.main()
