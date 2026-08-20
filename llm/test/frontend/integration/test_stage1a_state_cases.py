from __future__ import annotations

from collections import Counter
from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.artifact_manifest import RecordOpcode
from llm.frontend.wafer_frontend.schema.global_action import LogicalCoreRef
from llm.frontend.wafer_frontend.schema.ir0 import StateAccessMode
from llm.frontend.wafer_frontend.schema.ir2 import (
    BufferOwnership,
    SemanticTaskKind,
    StateIoOrigin,
    StateTransferOrigin,
)
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind
from llm.frontend.wafer_frontend.schema.program_io import (
    ProgramHbmTarget,
    ProgramIoMode,
    ProgramIoPurpose,
    ProgramSramTarget,
)

from stage1a_state_cases import (
    build_cross_action_kv_foundation_case,
    build_parameter_foundation_case,
    build_pd1_case,
)


class Stage1aParameterFoundationCaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = build_parameter_foundation_case()

    def test_dma_feeds_matmul_on_one_core_without_d2d(self) -> None:
        case = self.case
        self.assertEqual(len(case.projection.dags), 1)
        dag = case.projection.dags[0]
        self.assertEqual(
            Counter(task.kind for task in dag.tasks),
            Counter(
                {
                    SemanticTaskKind.DMA_IN: 1,
                    SemanticTaskKind.COMP: 1,
                }
            ),
        )
        dma = next(
            task for task in dag.tasks if task.kind is SemanticTaskKind.DMA_IN
        )
        comp = next(
            task for task in dag.tasks if task.kind is SemanticTaskKind.COMP
        )
        self.assertEqual(comp.deps, (dma.id,))
        self.assertEqual(len(dag.state_staging_values), 1)
        staging = dag.state_staging_values[0]
        assert dma.dma is not None
        assert comp.compute is not None
        self.assertEqual(dma.dma.local_value_ref, staging.id)
        self.assertEqual(
            tuple(operand.value_id for operand in comp.compute.inputs),
            ("p1.x", staging.id),
        )
        self.assertEqual(
            tuple(
                flow
                for local_dag in case.projection.dags
                for flow in local_dag.flows
            ),
            (),
        )

        schedule = case.schedule_set.schedules[0]
        self.assertEqual(
            {placement.task_id: placement.core_id for placement in schedule.placements},
            {dma.id: 0, comp.id: 0},
        )
        dma_action = next(
            action
            for action in case.global_dag.actions
            if action.task_kind is SemanticTaskKind.DMA_IN
        )
        comp_action = next(
            action
            for action in case.global_dag.actions
            if action.task_kind is SemanticTaskKind.COMP
        )
        self.assertEqual(comp_action.deps, (dma_action.id,))
        self.assertEqual(
            (dma_action.logical_core, comp_action.logical_core),
            (LogicalCoreRef(0, 0), LogicalCoreRef(0, 0)),
        )
        self.assertEqual(dma_action.bytes, 128)
        self.assertEqual(case.gemm_flops, 128)
        self.assertEqual(case.hbm_capacity_floor_cycles, 8)

    def test_linked_manifest_is_finalizer_input_with_exact_state_records(self) -> None:
        case = self.case
        records = tuple(
            record
            for fragment in case.manifest.fragments
            for stream in fragment.core_streams
            for record in stream.records
        )
        self.assertEqual(
            Counter(record.opcode for record in records)[RecordOpcode.LSU_LOAD],
            1,
        )
        self.assertEqual(
            Counter(record.opcode for record in records)[RecordOpcode.LSU_STORE],
            0,
        )
        self.assertEqual(
            Counter(record.opcode for record in records)[RecordOpcode.MATMUL],
            1,
        )
        matmul = next(
            record for record in records if record.opcode is RecordOpcode.MATMUL
        )
        self.assertEqual(matmul.operands[-1].literal_value, (1, 1, 8, 8))
        state_abis = tuple(
            abi
            for fragment in case.manifest.fragments
            for abi in fragment.state_abi
        )
        self.assertEqual(len(state_abis), 1)
        self.assertEqual(
            (
                state_abis[0].state_ref,
                state_abis[0].hbm_binding_ref,
                state_abis[0].address,
                state_abis[0].size_bytes,
            ),
            (case.state_ref, case.hbm_binding_ref, 0x1000, 128),
        )
        self.assertEqual(len(case.manifest.state_operand_bindings), 1)
        state_manifest = case.graph.persistent_state_manifest
        assert state_manifest is not None
        self.assertEqual(state_manifest.address_spaces[0].alignment_bytes, 64)
        self.assertEqual(state_manifest.bindings[0].address % 64, 0)
        case.manifest.validate_against(
            case.graph,
            (),
            (),
            case.projection,
            case.schedule_set,
            case.global_dag,
            case.manifest.fragments,
        )

    def test_timing_sidecar_contains_one_exact_hbm_seed(self) -> None:
        case = self.case
        self.assertIs(case.program_io.mode, ProgramIoMode.TIMING)
        hbm_initializations = tuple(
            entry
            for entry in case.program_io.initializations
            if type(entry.target) is ProgramHbmTarget
        )
        self.assertEqual(len(hbm_initializations), 1)
        initialization = hbm_initializations[0]
        assert type(initialization.target) is ProgramHbmTarget
        self.assertEqual(
            (
                initialization.target.state_ref,
                initialization.target.hbm_binding_ref,
                initialization.offset_bytes,
                initialization.length_bytes,
                initialization.purpose,
            ),
            (
                case.state_ref,
                case.hbm_binding_ref,
                0,
                128,
                ProgramIoPurpose.STATE,
            ),
        )
        blobs = {blob.id: blob.payload() for blob in case.program_io.blobs}
        self.assertEqual(blobs[initialization.blob_ref], bytes(range(128)))
        self.assertEqual(case.seed, bytes(range(128)))
        self.assertFalse(
            any(
                type(probe.target) is ProgramHbmTarget
                for probe in case.program_io.output_probes
            )
        )
        self.assertTrue(
            all(
                type(probe.target) is ProgramSramTarget
                for probe in case.program_io.output_probes
            )
        )
        case.program_io.validate_against(case.manifest)

    def test_builder_is_deterministic(self) -> None:
        self.assertEqual(build_parameter_foundation_case(), self.case)

class Stage1aCrossActionKvFoundationCaseTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.case = build_cross_action_kv_foundation_case()

    def test_four_states_store_then_load_on_one_core_across_actions(self) -> None:
        case = self.case
        manifest = case.graph.persistent_state_manifest
        assert manifest is not None
        self.assertEqual(len(manifest.declarations), 4)
        self.assertEqual(len(manifest.bindings), 4)
        self.assertEqual(
            tuple(len(payload) for _state_ref, payload in case.state_payloads),
            (32, 32, 32, 32),
        )
        self.assertEqual(
            {payload[0] for _state_ref, payload in case.state_payloads},
            {0x11, 0x22, 0x33, 0x44},
        )

        access_by_id = {access.id: access for access in case.graph.state_accesses}
        self.assertEqual(
            {access_by_id[item].mode for item in case.write_access_ids},
            {StateAccessMode.WRITE},
        )
        self.assertEqual(
            {access_by_id[item].mode for item in case.read_access_ids},
            {StateAccessMode.READ},
        )
        self.assertEqual(
            {access_by_id[item].state_ref for item in case.write_access_ids},
            {state_ref for state_ref, _payload in case.state_payloads},
        )
        self.assertEqual(
            {access_by_id[item].state_ref for item in case.read_access_ids},
            {state_ref for state_ref, _payload in case.state_payloads},
        )

        write_actions = tuple(
            action
            for action in case.global_dag.actions
            if action.task_kind is SemanticTaskKind.DMA_OUT
            and getattr(action.origin_ref, "state_access_ref", None)
            in case.write_access_ids
        )
        read_actions = tuple(
            action
            for action in case.global_dag.actions
            if action.task_kind is SemanticTaskKind.DMA_IN
            and getattr(action.origin_ref, "state_access_ref", None)
            in case.read_access_ids
        )
        self.assertEqual((len(write_actions), len(read_actions)), (4, 4))
        self.assertEqual(
            tuple(action.bytes for action in write_actions),
            (32, 32, 32, 32),
        )
        self.assertEqual(
            tuple(action.bytes for action in read_actions),
            (32, 32, 32, 32),
        )
        self.assertEqual(
            {action.logical_core for action in (*write_actions, *read_actions)},
            {LogicalCoreRef(0, 0)},
        )
        write_by_state = {
            access_by_id[action.origin_ref.state_access_ref].state_ref: action
            for action in write_actions
        }
        read_by_state = {
            access_by_id[action.origin_ref.state_access_ref].state_ref: action
            for action in read_actions
        }
        for state_ref, _payload in case.state_payloads:
            write_index = write_by_state[state_ref].core_order_index
            read_index = read_by_state[state_ref].core_order_index
            assert write_index is not None and read_index is not None
            self.assertLess(write_index, read_index)
        write_end = max(
            action.core_order_index
            for action in write_actions
            if action.core_order_index is not None
        )
        read_start = min(
            action.core_order_index
            for action in read_actions
            if action.core_order_index is not None
        )
        middle = tuple(
            action
            for action in case.global_dag.actions
            if action.core_order_index is not None
            and write_end < action.core_order_index < read_start
        )
        self.assertEqual(len(middle), 10)
        self.assertTrue(
            all(action.task_kind is SemanticTaskKind.COMP for action in middle)
        )
        self.assertEqual(case.middle_action_count, 10)

    def test_exact_state_records_bytes_and_capacity_floor(self) -> None:
        case = self.case
        records = tuple(
            record
            for fragment in case.manifest.fragments
            for stream in fragment.core_streams
            for record in stream.records
        )
        counts = Counter(record.opcode for record in records)
        self.assertEqual(
            counts,
            Counter(
                {
                    RecordOpcode.MATMUL: 9,
                    RecordOpcode.SWIGLU: 2,
                    RecordOpcode.RESIDUAL: 4,
                    RecordOpcode.RMSNORM: 5,
                    RecordOpcode.ROPE_QK_EXACT: 2,
                    RecordOpcode.ATTENTION_EXACT: 2,
                    RecordOpcode.EMBEDDING_LOOKUP: 1,
                    RecordOpcode.LSU_LOAD: 4,
                    RecordOpcode.LSU_STORE: 4,
                    RecordOpcode.SRAM_BIND: 25,
                    RecordOpcode.SRAM_FREE: 49,
                    RecordOpcode.SRAM_ALLOC_AT: 49,
                }
            ),
        )
        state_abis = {
            abi.id: abi
            for fragment in case.manifest.fragments
            for abi in fragment.state_abi
        }
        self.assertEqual(len(state_abis), 4)
        self.assertEqual(
            {abi.state_ref for abi in state_abis.values()},
            {state_ref for state_ref, _payload in case.state_payloads},
        )
        self.assertEqual(len(case.manifest.state_operand_bindings), 8)
        self.assertEqual(
            (
                case.hbm_write_bytes,
                case.hbm_read_bytes,
                case.d2d_bytes,
                case.hbm_capacity_floor_cycles,
            ),
            (128, 128, 0, 16),
        )
        case.manifest.validate_against(
            case.graph,
            (),
            (),
            case.projection,
            case.schedule_set,
            case.global_dag,
            tuple(
                fragment.fragment
                if hasattr(fragment, "fragment")
                else fragment
                for fragment in case.manifest.fragments
            ),
        )

    def test_program_io_carries_exact_sram_store_load_and_hbm_probes(self) -> None:
        case = self.case
        self.assertIs(case.program_io.mode, ProgramIoMode.TIMING)
        self.assertEqual(
            (
                len(case.program_io.initializations),
                len(case.program_io.output_probes),
            ),
            (49, 9),
        )
        blobs = {blob.id: blob.payload() for blob in case.program_io.blobs}
        hbm_initializations = tuple(
            entry
            for entry in case.program_io.initializations
            if type(entry.target) is ProgramHbmTarget
        )
        self.assertEqual(hbm_initializations, ())
        hbm_probes = tuple(
            probe
            for probe in case.program_io.output_probes
            if type(probe.target) is ProgramHbmTarget
        )
        self.assertEqual(len(hbm_probes), 4)
        self.assertEqual(
            {
                probe.target.state_ref: blobs[probe.blob_ref]
                for probe in hbm_probes
            },
            dict(case.state_payloads),
        )
        write_initializations = tuple(
            entry
            for entry in case.program_io.initializations
            if type(entry.target) is ProgramSramTarget
            and entry.target.value_id in case.write_staging_ids
        )
        read_probes = tuple(
            probe
            for probe in case.program_io.output_probes
            if type(probe.target) is ProgramSramTarget
            and probe.target.value_id in case.read_staging_ids
        )
        self.assertEqual((len(write_initializations), len(read_probes)), (4, 4))
        access_by_id = {access.id: access for access in case.graph.state_accesses}
        write_state_by_value = {
            staging.id: access_by_id[staging.state_access_ref].state_ref
            for dag in case.projection.dags
            for staging in dag.state_staging_values
            if staging.id in case.write_staging_ids
        }
        read_state_by_value = {
            staging.id: access_by_id[staging.state_access_ref].state_ref
            for dag in case.projection.dags
            for staging in dag.state_staging_values
            if staging.id in case.read_staging_ids
        }
        payload_by_state = dict(case.state_payloads)
        self.assertEqual(
            {
                entry.target.value_id: blobs[entry.blob_ref]
                for entry in write_initializations
            },
            {
                value_id: payload_by_state[state_ref]
                for value_id, state_ref in write_state_by_value.items()
            },
        )
        self.assertEqual(
            {
                probe.target.value_id: blobs[probe.blob_ref]
                for probe in read_probes
            },
            {
                value_id: payload_by_state[state_ref]
                for value_id, state_ref in read_state_by_value.items()
            },
        )
        buffer_abis = {
            abi.id: abi
            for fragment in case.manifest.fragments
            for abi in fragment.buffer_abi
        }
        self.assertTrue(
            all(
                buffer_abis[entry.target.buffer_abi_id].ownership
                is BufferOwnership.BORROWED
                for entry in write_initializations
            )
        )
        self.assertTrue(
            all(
                buffer_abis[probe.target.buffer_abi_id].ownership
                is BufferOwnership.OWNED
                for probe in read_probes
            )
        )
        case.program_io.validate_against(case.manifest)

    def test_builder_is_deterministic(self) -> None:
        self.assertEqual(build_cross_action_kv_foundation_case(), self.case)


class Stage1aPd1FoundationCaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = build_pd1_case()

    @staticmethod
    def _is_ancestor(actions, ancestor_id: str, target_id: str) -> bool:
        by_id = {action.id: action for action in actions}
        pending = list(by_id[target_id].deps)
        seen: set[str] = set()
        while pending:
            current = pending.pop()
            if current == ancestor_id:
                return True
            if current not in seen:
                seen.add(current)
                pending.extend(by_id[current].deps)
        return False

    def test_two_lineage_pairs_and_exact_bridge_completion(self) -> None:
        case = self.case
        manifest = case.graph.persistent_state_manifest
        assert manifest is not None
        self.assertEqual(
            (len(manifest.declarations), len(manifest.bindings)),
            (4, 4),
        )
        access_index = {access.id: access for access in case.graph.state_accesses}
        declaration_index = {
            declaration.id: declaration
            for declaration in manifest.declarations
        }
        self.assertEqual(
            {access_index[item].mode for item in case.source_access_ids},
            {StateAccessMode.READ},
        )
        self.assertEqual(
            {access_index[item].rank for item in case.source_access_ids},
            {0},
        )
        self.assertEqual(
            {access_index[item].mode for item in case.destination_access_ids},
            {StateAccessMode.WRITE},
        )
        self.assertEqual(
            {access_index[item].rank for item in case.destination_access_ids},
            {1},
        )
        lineage_pairs = []
        for contract in case.contracts:
            source = access_index[contract.source_state_access_ref]
            destination = access_index[
                contract.destination_state_access_ref
            ]
            source_decl = declaration_index[source.state_ref]
            destination_decl = declaration_index[destination.state_ref]
            self.assertNotEqual(source.state_ref, destination.state_ref)
            self.assertEqual(
                (
                    source_decl.identity.kind,
                    source_decl.identity.request_ref,
                    source_decl.identity.layer_index,
                    source_decl.identity.generation,
                    source_decl.shape,
                    source_decl.tensor_bytes,
                ),
                (
                    destination_decl.identity.kind,
                    destination_decl.identity.request_ref,
                    destination_decl.identity.layer_index,
                    destination_decl.identity.generation,
                    destination_decl.shape,
                    destination_decl.tensor_bytes,
                ),
            )
            lineage_pairs.append(source_decl.identity.kind)
        self.assertEqual(
            set(lineage_pairs),
            {StateKind.KV_KEY, StateKind.KV_VALUE},
        )
        self.assertEqual(len(lineage_pairs), 2)
        self.assertEqual(len(case.global_dag.actions), 12)
        self.assertEqual(len(case.bridge_action_ids), 12)
        self.assertEqual(
            set(case.bridge_action_ids),
            {action.id for action in case.global_dag.actions},
        )

        task_index = {
            task.id: task
            for dag in case.projection.dags
            for task in dag.tasks
        }
        action_by_task = {
            action.source.task_id: action for action in case.global_dag.actions
        }
        for contract in case.contracts:
            source_dma = next(
                task
                for task in task_index.values()
                if isinstance(task.origin_ref, StateIoOrigin)
                and task.origin_ref.state_access_ref
                == contract.source_state_access_ref
                and task.kind is SemanticTaskKind.DMA_IN
            )
            destination_dma = next(
                task
                for task in task_index.values()
                if isinstance(task.origin_ref, StateIoOrigin)
                and task.origin_ref.state_access_ref
                == contract.destination_state_access_ref
                and task.kind is SemanticTaskKind.DMA_OUT
            )
            send = next(
                task
                for task in task_index.values()
                if isinstance(task.origin_ref, StateTransferOrigin)
                and task.origin_ref.state_transfer_ref == contract.id
                and task.kind is SemanticTaskKind.SEND
            )
            recv = next(
                task
                for task in task_index.values()
                if isinstance(task.origin_ref, StateTransferOrigin)
                and task.origin_ref.state_transfer_ref == contract.id
                and task.kind is SemanticTaskKind.RECV
            )
            wait = next(
                task
                for task in task_index.values()
                if isinstance(task.origin_ref, StateTransferOrigin)
                and task.origin_ref.state_transfer_ref == contract.id
                and task.kind is SemanticTaskKind.WAIT
            )
            assert source_dma.dma is not None
            assert destination_dma.dma is not None
            source_target = task_index[source_dma.dma.access_task_refs[0]]
            destination_target = task_index[
                destination_dma.dma.access_task_refs[0]
            ]
            for ancestor, target in (
                (source_dma, source_target),
                (source_target, send),
                (recv, wait),
                (wait, destination_target),
                (destination_target, destination_dma),
            ):
                self.assertTrue(
                    self._is_ancestor(
                        case.global_dag.actions,
                        action_by_task[ancestor.id].id,
                        action_by_task[target.id].id,
                    )
                )

    def test_exact_hbm_d2d_and_linked_record_counts(self) -> None:
        case = self.case
        records = tuple(
            record
            for fragment in case.manifest.fragments
            for stream in fragment.core_streams
            for record in stream.records
        )
        counts = Counter(record.opcode for record in records)
        self.assertEqual(
            {
                opcode: counts[opcode]
                for opcode in (
                    RecordOpcode.LSU_LOAD,
                    RecordOpcode.LSU_STORE,
                    RecordOpcode.DTE_SEND,
                    RecordOpcode.DTE_RECV,
                    RecordOpcode.DTE_WAIT,
                    RecordOpcode.ATTENTION_EXACT,
                )
            },
            {
                RecordOpcode.LSU_LOAD: 2,
                RecordOpcode.LSU_STORE: 2,
                RecordOpcode.DTE_SEND: 2,
                RecordOpcode.DTE_RECV: 2,
                RecordOpcode.DTE_WAIT: 2,
                RecordOpcode.ATTENTION_EXACT: 2,
            },
        )
        source_refs = set(case.source_access_ids)
        destination_refs = set(case.destination_access_ids)
        self.assertEqual(
            sum(
                action.bytes
                for action in case.global_dag.actions
                if action.task_kind is SemanticTaskKind.DMA_IN
                and isinstance(action.origin_ref, StateIoOrigin)
                and action.origin_ref.state_access_ref in source_refs
            ),
            64,
        )
        self.assertEqual(
            sum(
                action.bytes
                for action in case.global_dag.actions
                if action.task_kind is SemanticTaskKind.DMA_OUT
                and isinstance(action.origin_ref, StateIoOrigin)
                and action.origin_ref.state_access_ref in destination_refs
            ),
            64,
        )
        self.assertEqual(case.d2d_bytes, 64)
        flow_ids = {
            task.flow_id
            for dag in case.projection.dags
            for task in dag.tasks
            if isinstance(task.origin_ref, StateTransferOrigin)
            and task.kind is SemanticTaskKind.SEND
        }
        replicas = tuple(
            flow
            for dag in case.projection.dags
            for flow in dag.flows
            if flow.id in flow_ids
        )
        self.assertEqual((len(flow_ids), len(replicas)), (2, 4))
        self.assertEqual({flow.bytes for flow in replicas}, {32})
        case.manifest.validate_against(
            case.graph,
            (),
            (),
            case.projection,
            case.schedule_set,
            case.global_dag,
            case.manifest.fragments,
        )

    def test_program_io_has_two_source_seeds_and_two_destination_probes(self) -> None:
        case = self.case
        blobs = {blob.id: blob.payload() for blob in case.program_io.blobs}
        hbm_initializations = tuple(
            entry
            for entry in case.program_io.initializations
            if type(entry.target) is ProgramHbmTarget
        )
        hbm_probes = tuple(
            probe
            for probe in case.program_io.output_probes
            if type(probe.target) is ProgramHbmTarget
        )
        self.assertEqual((len(hbm_initializations), len(hbm_probes)), (2, 2))
        self.assertEqual(
            {
                entry.target.state_ref: blobs[entry.blob_ref]
                for entry in hbm_initializations
            },
            dict(case.source_payloads),
        )
        self.assertEqual(
            {
                probe.target.state_ref: blobs[probe.blob_ref]
                for probe in hbm_probes
            },
            dict(case.destination_expected),
        )
        self.assertEqual(
            {payload[0] for _state_ref, payload in case.source_payloads},
            {0x5A, 0xA5},
        )
        self.assertTrue(
            all(
                entry.purpose is ProgramIoPurpose.STATE
                and entry.offset_bytes == 0
                and entry.length_bytes == 32
                for entry in hbm_initializations
            )
        )
        self.assertTrue(
            all(
                probe.offset_bytes == 0 and probe.length_bytes == 32
                for probe in hbm_probes
            )
        )
        case.program_io.validate_against(case.manifest)

    def test_builder_is_deterministic_and_contract_tamper_fails_closed(self) -> None:
        case = self.case
        self.assertEqual(build_pd1_case(), case)
        with self.assertRaises(SchemaError):
            replace(
                case.contracts[0],
                source_ir1_id="forged.ir1",
            ).validate_against(case.graph)


if __name__ == "__main__":
    unittest.main()
