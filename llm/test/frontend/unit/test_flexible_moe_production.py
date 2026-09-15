import unittest
from dataclasses import replace

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.flexible_moe_production import (
    FlexibleMoeProductionArtifacts,
    FlexibleMoeRuntimeEvidence,
    build_flexible_moe_production_program_io,
    lower_link_flexible_moe_production,
    observe_flexible_moe_production_runtime,
)
from llm.frontend.wafer_frontend.lowering.flexible_moe_multi_production import (
    _validate_fan_in_event_refs,
)
from llm.frontend.wafer_frontend.passes.flexible_moe import (
    build_round_robin_flexible_moe_spec,
    compile_flexible_moe_baseline,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    STATE_TRANSFER_WAVE_RUNTIME_SYMBOL_SCHEMA_VERSION,
    CommandFragment,
    LinkedProgramManifest,
    ManifestInputKind,
    ProgramSymbolKind,
    RecordOpcode,
    RuntimeSymbolKind,
)
from llm.frontend.wafer_frontend.schema.common import DType, stable_artifact_id
from llm.frontend.wafer_frontend.schema.flexible_moe import (
    FlexibleMoeLimits,
    FlexibleMoeMode,
    FlexibleMoeSpec,
    MoeRectActionKind,
    MoeRectStaticTrace,
    MoeRectTraceAssignment,
)
from llm.frontend.wafer_frontend.schema.program_io import ProgramIoContract
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.frontend.wafer_frontend.schema.serde import canonical_digest


def _case(mode: FlexibleMoeMode):
    spec = build_round_robin_flexible_moe_spec(
        RectMeshSpec(1, 1), mode, routing_shift=0,
    )
    return spec, compile_flexible_moe_baseline(spec)


def _hot_case(rows: int, columns: int):
    mesh = RectMeshSpec(rows, columns)
    trace = MoeRectStaticTrace.create(
        token_count=mesh.rank_count,
        expert_count=mesh.rank_count,
        capacity_per_expert=mesh.rank_count,
        assignments=tuple(
            MoeRectTraceAssignment(rank, rank, 0, 0, rank)
            for rank in range(mesh.rank_count)
        ),
    )
    spec = FlexibleMoeSpec.create(
        mesh=mesh,
        mode=FlexibleMoeMode.INFERENCE,
        hidden_size=16,
        intermediate_size=32,
        expert_count=mesh.rank_count,
        expert_parallel_degree=mesh.rank_count,
        top_k=1,
        trace_mode="static",
        trace=trace,
        limits=FlexibleMoeLimits(),
        expert_dtype=DType.FP16,
        combine_dtype=DType.FP32,
        token_drop=False,
    )
    return spec, compile_flexible_moe_baseline(spec)


class FlexibleMoeProductionTest(unittest.TestCase):
    def test_hot_fan_in_materializes_exact_ordered_event_fences(self) -> None:
        spec, plan = _hot_case(1, 10)
        result = lower_link_flexible_moe_production(
            plan, spec, physical_region_name="dense_release",
        )
        result.validate_against(plan, spec)
        self.assertEqual(len(plan.flows), 18)
        self.assertLessEqual(plan.max_sessions_per_rank_wave, 3)

        records_by_core = {}
        opcodes = []
        for fragment in result.fragments:
            for stream in fragment.core_streams:
                records_by_core.setdefault(stream.logical_core.die_id, []).extend(
                    stream.records
                )
                opcodes.extend(record.opcode for record in stream.records)
        for opcode in (
            RecordOpcode.DTE_SEND,
            RecordOpcode.DTE_RECV,
            RecordOpcode.DTE_WAIT,
        ):
            self.assertEqual(opcodes.count(opcode), len(plan.flows))
        self.assertEqual(opcodes.count(RecordOpcode.EVENT_SET), 16)
        self.assertEqual(opcodes.count(RecordOpcode.EVENT_WAIT), 16)

        action_by_id = {action.id: action for action in plan.actions}
        event_definitions = tuple(
            definition for definition in result.manifest.runtime_symbol_definitions
            if definition.symbol.kind is RuntimeSymbolKind.EVENT_TAG
        )
        self.assertEqual(len(event_definitions), 16)
        self.assertEqual(len({item.symbol.id for item in event_definitions}), 16)
        fenced_edges = set()
        for definition in event_definitions:
            source_action = action_by_id[definition.source_action_id]
            destination_action = action_by_id[definition.destination_action_id]
            self.assertEqual(source_action.kind, MoeRectActionKind.WAIT)
            self.assertEqual(destination_action.kind, MoeRectActionKind.SEND)
            fenced_edges.add((source_action.rank, destination_action.rank))
            binding_ref = stable_artifact_id(
                "state_transfer_wave_binding",
                {
                    "source_global_dag_id": plan.id,
                    "source_action_id": source_action.id,
                    "destination_action_id": destination_action.id,
                    "capacity": 3,
                },
                schema_version=STATE_TRANSFER_WAVE_RUNTIME_SYMBOL_SCHEMA_VERSION,
            )
            self.assertEqual(definition.symbol.source_ref, binding_ref)
            self.assertEqual(
                definition.symbol.id,
                stable_artifact_id(
                    "state_transfer_wave_event",
                    {
                        "binding_ref": binding_ref,
                        "source_action_id": source_action.id,
                        "destination_action_id": destination_action.id,
                    },
                    schema_version=STATE_TRANSFER_WAVE_RUNTIME_SYMBOL_SCHEMA_VERSION,
                ),
            )
            self.assertEqual(
                tuple(core.die_id for core in definition.logical_cores),
                tuple(sorted((source_action.rank, destination_action.rank))),
            )

            source_records = records_by_core[source_action.rank]
            destination_records = records_by_core[destination_action.rank]
            source_wait_index = next(
                index for index, record in enumerate(source_records)
                if record.opcode is RecordOpcode.DTE_WAIT
                and record.source_global_action_id == source_action.id
            )
            set_index = next(
                index for index, record in enumerate(source_records)
                if record.opcode is RecordOpcode.EVENT_SET
                and record.source_global_action_id == source_action.id
                and record.operands[2].symbol_ref == definition.symbol.id
            )
            wait_index = next(
                index for index, record in enumerate(destination_records)
                if record.opcode is RecordOpcode.EVENT_WAIT
                and record.source_global_action_id == destination_action.id
                and record.operands[2].symbol_ref == definition.symbol.id
            )
            send_index = next(
                index for index, record in enumerate(destination_records)
                if record.opcode is RecordOpcode.DTE_SEND
                and record.source_global_action_id == destination_action.id
            )
            self.assertEqual(set_index, source_wait_index + 1)
            self.assertEqual(send_index, wait_index + 1)
        self.assertEqual(
            fenced_edges,
            {(0, rank) for rank in range(1, 9)}
            | {(rank, 0) for rank in range(1, 9)},
        )

    def test_hot_fan_in_event_closure_rejects_missing_and_duplicate_pairs(self) -> None:
        _validate_fan_in_event_refs(("event.a", "event.b"), ("event.b", "event.a"))
        with self.assertRaisesRegex(SchemaError, "close exactly"):
            _validate_fan_in_event_refs(("event.a",), ("event.a", "event.b"))
        with self.assertRaisesRegex(SchemaError, "EVENT_WAIT tags must be unique"):
            _validate_fan_in_event_refs(
                ("event.a", "event.b"), ("event.a", "event.a"),
            )

    def test_hot_fan_in_large_representatives_materialize_with_bounded_sessions(self) -> None:
        for rows, columns in ((2, 10), (10, 2)):
            with self.subTest(rows=rows, columns=columns):
                spec, plan = _hot_case(rows, columns)
                result = lower_link_flexible_moe_production(
                    plan, spec, physical_region_name="dense_release",
                )
                result.validate_against(plan, spec)
                self.assertEqual(len(plan.flows), 2 * (spec.mesh.rank_count - 1))
                self.assertLessEqual(plan.max_sessions_per_rank_wave, 3)
                opcodes = tuple(
                    record.opcode
                    for fragment in result.fragments
                    for stream in fragment.core_streams
                    for record in stream.records
                )
                self.assertEqual(opcodes.count(RecordOpcode.EVENT_SET), 36)
                self.assertEqual(opcodes.count(RecordOpcode.EVENT_WAIT), 36)

    def test_exact_linked_manifest_file_limit_fails_closed(self) -> None:
        spec = build_round_robin_flexible_moe_spec(
            RectMeshSpec(1, 2), FlexibleMoeMode.TRAIN,
        )
        baseline = compile_flexible_moe_baseline(spec)
        constrained = FlexibleMoeSpec.create(**(
            spec._semantic()
            | {"limits": replace(
                spec.limits,
                max_artifact_file_bytes=baseline.symbolic_file_bytes + 1024,
            )}
        ))
        plan = compile_flexible_moe_baseline(constrained)
        self.assertLessEqual(
            plan.symbolic_file_bytes,
            constrained.limits.max_artifact_file_bytes,
        )
        with self.assertRaisesRegex(SchemaError, "linked manifest file capacity"):
            lower_link_flexible_moe_production(plan, constrained)

    def test_1x1_inference_is_real_single_manifest_and_program_io(self) -> None:
        spec, plan = _case(FlexibleMoeMode.INFERENCE)
        result = lower_link_flexible_moe_production(plan, spec)
        self.assertIs(type(result), FlexibleMoeProductionArtifacts)
        self.assertIs(type(result.manifest), LinkedProgramManifest)
        self.assertTrue(all(type(item) is CommandFragment for item in result.fragments))
        self.assertTrue(result.lower_link_verified)
        self.assertFalse(result.runtime_verified)
        result.validate_against(plan, spec)

        opcodes = {
            record.opcode
            for fragment in result.fragments
            for stream in fragment.core_streams
            for record in stream.records
        }
        self.assertTrue({
            RecordOpcode.SRAM_ALLOC_AT,
            RecordOpcode.LSU_LOAD,
            RecordOpcode.MATMUL,
            RecordOpcode.LOCAL_REDUCE,
        }.issubset(opcodes))
        program_io = build_flexible_moe_production_program_io(
            result, plan, spec, "a" * 64,
        )
        self.assertIs(type(program_io), ProgramIoContract)
        program_io.validate_against(result.manifest)
        self.assertEqual(len(program_io.output_probes), 1)

    def test_1x1_train_closes_sgd_store_and_updated_state_program_io(self) -> None:
        spec, plan = _case(FlexibleMoeMode.TRAIN)
        result = lower_link_flexible_moe_production(plan, spec)
        opcodes = tuple(
            record.opcode
            for fragment in result.fragments
            for stream in fragment.core_streams
            for record in stream.records
        )
        self.assertEqual(opcodes.count(RecordOpcode.SGD_UPDATE), 2)
        self.assertEqual(opcodes.count(RecordOpcode.LSU_STORE), 2)
        program_io = build_flexible_moe_production_program_io(
            result, plan, spec, "b" * 64,
        )
        program_io.validate_against(result.manifest)
        self.assertEqual(len(program_io.output_probes), 2)

    def test_is_deterministic(self) -> None:
        spec, plan = _case(FlexibleMoeMode.INFERENCE)
        first = lower_link_flexible_moe_production(plan, spec)
        second = lower_link_flexible_moe_production(plan, spec)
        self.assertEqual(first, second)

    def test_program_io_rejects_placeholder_artifact_sha(self) -> None:
        spec, plan = _case(FlexibleMoeMode.INFERENCE)
        result = lower_link_flexible_moe_production(plan, spec)
        with self.assertRaisesRegex(SchemaError, "actual non-placeholder"):
            build_flexible_moe_production_program_io(
                result, plan, spec, "0" * 64,
            )

    def test_runtime_evidence_closes_markers_and_fails_on_residual(self) -> None:
        spec, plan = _case(FlexibleMoeMode.INFERENCE)
        artifacts = lower_link_flexible_moe_production(plan, spec)
        program_io = build_flexible_moe_production_program_io(
            artifacts, plan, spec, "a" * 64,
        )
        report = {
            "artifact_sha256": program_io.program_artifact_sha256,
            "linked_manifest_id": artifacts.manifest.id,
            "linked_manifest_digest": canonical_digest(artifacts.manifest),
        }
        resolver = "ProgramIo resolved initializations=3 probes=1"
        output = "\n".join((
            "[SIM_RESULT] makespan_cycles=1538",
            "[PROGRAM_MEMORY] core=0 lsu_issued=3 lsu_completed=3 "
            "lsu_hbm_read_bytes=1 lsu_residual=0 dte_residual=0",
            "[PROGRAM_IO_PROBE] id=p0 valid=1 exact=1 pass=1",
            "[PROGRAM_IO] phase=verify mode=timing initializations=3 probes=1 pass=1",
            "[P5 P2P TIMING DRAIN] residual=0",
            "[DRAIN] router_residual=0",
            "[DRAIN] d2d_link_residual=0",
        ))
        evidence = observe_flexible_moe_production_runtime(
            artifacts,
            spec,
            program_io,
            finalizer_report=report,
            resolver_output=resolver,
            npusim_output=output,
            finalizer_exit_code=0,
            resolver_exit_code=0,
            npusim_exit_code=0,
        )
        self.assertIs(type(evidence), FlexibleMoeRuntimeEvidence)
        self.assertTrue(evidence.runtime_verified)
        self.assertFalse(evidence.functional_execution)
        with self.assertRaisesRegex(SchemaError, "drain"):
            observe_flexible_moe_production_runtime(
                artifacts,
                spec,
                program_io,
                finalizer_report=report,
                resolver_output=resolver,
                npusim_output=output.replace("router_residual=0", "router_residual=1"),
                finalizer_exit_code=0,
                resolver_exit_code=0,
                npusim_exit_code=0,
            )
        with self.assertRaisesRegex(SchemaError, "exit code"):
            observe_flexible_moe_production_runtime(
                artifacts,
                spec,
                program_io,
                finalizer_report=report,
                resolver_output=resolver,
                npusim_output=output,
                finalizer_exit_code=0,
                resolver_exit_code=0,
                npusim_exit_code=7,
            )

    def test_wrapper_rejects_forged_plan_digest_and_source(self) -> None:
        spec, plan = _case(FlexibleMoeMode.INFERENCE)
        result = lower_link_flexible_moe_production(plan, spec)
        forged_inputs = tuple(
            replace(digest, digest="f" * 64)
            if digest.kind is ManifestInputKind.FLEXIBLE_MOE_PLAN
            else digest
            for digest in result.manifest.input_digests
        )
        forged_manifest = LinkedProgramManifest.create(
            producer_pass=result.manifest.producer_pass,
            **{
                **result.manifest._semantic_key(),
                "input_digests": forged_inputs,
            },
        )
        with self.assertRaisesRegex(SchemaError, "typed source"):
            replace(result, manifest=forged_manifest).validate_against(plan, spec)

        forged_manifest = LinkedProgramManifest.create(
            producer_pass=result.manifest.producer_pass,
            **{
                **result.manifest._semantic_key(),
                "source_ir1_id": "forged.spec",
            },
        )
        with self.assertRaisesRegex(SchemaError, "source lineage"):
            replace(result, manifest=forged_manifest).validate_against(plan, spec)

        forged_inputs = tuple(
            replace(digest, digest="e" * 64)
            if digest.kind is ManifestInputKind.COMMAND_FRAGMENT
            else digest
            for digest in result.manifest.input_digests
        )
        forged_manifest = LinkedProgramManifest.create(
            producer_pass=result.manifest.producer_pass,
            **{
                **result.manifest._semantic_key(),
                "input_digests": forged_inputs,
            },
        )
        with self.assertRaisesRegex(SchemaError, "fragment digest"):
            replace(result, manifest=forged_manifest).validate_against(plan, spec)

    def test_representative_multi_die_shapes_materialize_exact_dte_closure(self) -> None:
        for rows, columns in ((1, 2), (2, 1), (2, 2), (2, 3), (3, 2)):
            for mode in FlexibleMoeMode:
                with self.subTest(rows=rows, columns=columns, mode=mode):
                    spec = build_round_robin_flexible_moe_spec(
                        RectMeshSpec(rows, columns), mode,
                    )
                    plan = compile_flexible_moe_baseline(spec)
                    result = lower_link_flexible_moe_production(plan, spec)
                    repeated = lower_link_flexible_moe_production(plan, spec)
                    self.assertEqual(result, repeated)
                    result.validate_against(plan, spec)
                    self.assertEqual(
                        tuple(item.logical_core.die_id for item in result.manifest.core_streams),
                        tuple(range(spec.mesh.rank_count)),
                    )
                    self.assertEqual(
                        tuple(item.runtime_core_id for item in result.manifest.core_streams),
                        tuple(rank * 16 for rank in range(spec.mesh.rank_count)),
                    )
                    output_buffers = tuple(
                        abi for fragment in result.fragments for abi in fragment.buffer_abi
                        if abi.value_id.endswith(".output")
                    )
                    self.assertEqual(
                        {item.region_ref for item in output_buffers},
                        {"flexible_moe.sram.comm"},
                    )
                    opcodes = tuple(
                        record.opcode
                        for fragment in result.fragments
                        for stream in fragment.core_streams
                        for record in stream.records
                    )
                    for opcode in (
                        RecordOpcode.DTE_SEND,
                        RecordOpcode.DTE_RECV,
                        RecordOpcode.DTE_WAIT,
                    ):
                        self.assertEqual(opcodes.count(opcode), len(plan.flows))
                    definitions = result.manifest.runtime_symbol_definitions
                    self.assertEqual(
                        sum(item.symbol.kind.value == "dte_fsm" for item in definitions),
                        len(plan.flows),
                    )
                    self.assertEqual(
                        sum(item.symbol.kind.value == "dte_token" for item in definitions),
                        len(plan.flows),
                    )
                    program_io = build_flexible_moe_production_program_io(
                        result, plan, spec, "c" * 64,
                    )
                    program_io.validate_against(result.manifest)
                    self.assertEqual(len(program_io.initializations), 3 * spec.mesh.rank_count)
                    self.assertEqual(
                        len(program_io.output_probes),
                        spec.mesh.rank_count if mode is FlexibleMoeMode.INFERENCE
                        else 2 * spec.mesh.rank_count,
                    )

    def test_balanced_2x3_manifest_canonical_is_unchanged_by_hot_fan_in(self) -> None:
        expected = {
            FlexibleMoeMode.INFERENCE: (
                "linked_program_manifest_d716424efc14a6b7",
                "9d3e769dd10269e2d4bc865734de52b11cd550f010b0daee2a42b9ab5ddb2411",
            ),
            FlexibleMoeMode.TRAIN: (
                "linked_program_manifest_1932a864547a01b5",
                "9d936cdb470bfd56715bd5b8271239fa0451e0e753a670edb1af299c0d336bdf",
            ),
        }
        for mode in FlexibleMoeMode:
            spec = build_round_robin_flexible_moe_spec(
                RectMeshSpec(2, 3), mode, routing_shift=1,
            )
            plan = compile_flexible_moe_baseline(spec)
            result = lower_link_flexible_moe_production(plan, spec)
            self.assertEqual(
                (result.manifest.id, canonical_digest(result.manifest)),
                expected[mode],
            )

    def test_full_model_dataflow_empty_expert_keeps_state_without_phantom_compute(self) -> None:
        spec, plan = _hot_case(1, 2)
        artifacts = lower_link_flexible_moe_production(
            plan, spec, full_model_dataflow=True,
        )
        expert_by_rank = {
            action.rank: action
            for action in plan.actions if action.kind is MoeRectActionKind.EXPERT_FORWARD
        }
        self.assertEqual(len(expert_by_rank[0].assignment_refs), 2)
        self.assertEqual(expert_by_rank[1].flops, 0)
        physical = {
            rank: [record for fragment in artifacts.manifest.fragments
                   for stream in fragment.core_streams
                   if stream.logical_core.die_id == rank
                   for record in stream.records]
            for rank in (0, 1)
        }
        self.assertFalse(any(
            record.source_global_action_id == expert_by_rank[1].id
            and record.opcode in (RecordOpcode.MATMUL, RecordOpcode.SWIGLU)
            for record in physical[1]
        ))
        self.assertEqual(sum(
            record.opcode is RecordOpcode.MATMUL and
            record.source_global_action_id == expert_by_rank[0].id
            for record in physical[0]
        ), 1)
        self.assertTrue(any(
            item.kind.value == "trainable_parameter" and item.die_id == 1
            for fragment in artifacts.manifest.fragments
            for item in fragment.state_abi
        ))

    def test_release_physical_region_does_not_change_frozen_default(self) -> None:
        spec = build_round_robin_flexible_moe_spec(
            RectMeshSpec(1, 2), FlexibleMoeMode.INFERENCE,
        )
        plan = compile_flexible_moe_baseline(spec)
        default = lower_link_flexible_moe_production(plan, spec)
        self.assertEqual(
            default.manifest.id,
            "linked_program_manifest_94ab147f7b90418b",
        )
        default_regions = tuple(
            (item.name, item.value, item.size_bytes)
            for item in default.manifest.program_symbol_definitions
            if item.symbol.kind is ProgramSymbolKind.SRAM_REGION
        )
        self.assertEqual(
            set(default_regions),
            {("input", 4096, 36864), ("comm", 40960, 36864)},
        )
        self.assertEqual(default, lower_link_flexible_moe_production(plan, spec))

        release = lower_link_flexible_moe_production(
            plan, spec, physical_region_name="dense_release",
        )
        release_regions = tuple(
            (item.name, item.value, item.size_bytes)
            for item in release.manifest.program_symbol_definitions
            if item.symbol.kind is ProgramSymbolKind.SRAM_REGION
        )
        self.assertEqual(release_regions, (("dense_release", 0, 1 << 20),))
        buffers = {
            item.id: item
            for fragment in release.fragments
            for item in fragment.buffer_abi
        }
        self.assertEqual(
            {item.region_ref for item in buffers.values()},
            {"flexible_moe.sram.core0"},
        )
        for rank in range(spec.mesh.rank_count):
            spans = sorted(
                (item.region_offset_bytes, item.region_offset_bytes + item.size_bytes)
                for item in buffers.values()
                if item.logical_core.die_id == rank
            )
            self.assertLessEqual(spans[-1][1], 1 << 20)
            self.assertTrue(all(
                left[1] <= right[0] for left, right in zip(spans, spans[1:])
            ))

    def test_multi_die_all_local_trace_emits_no_transport(self) -> None:
        spec = build_round_robin_flexible_moe_spec(
            RectMeshSpec(2, 3), FlexibleMoeMode.INFERENCE, routing_shift=0,
        )
        plan = compile_flexible_moe_baseline(spec)
        self.assertEqual(plan.flows, ())
        result = lower_link_flexible_moe_production(plan, spec)
        opcodes = {
            record.opcode
            for fragment in result.fragments
            for stream in fragment.core_streams
            for record in stream.records
        }
        self.assertTrue({
            RecordOpcode.DTE_SEND,
            RecordOpcode.DTE_RECV,
            RecordOpcode.DTE_WAIT,
        }.isdisjoint(opcodes))
        self.assertFalse(any(
            item.symbol.kind.value.startswith("dte_")
            for item in result.manifest.runtime_symbol_definitions
        ))


if __name__ == "__main__":
    unittest.main()
