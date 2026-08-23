from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.swizzle import (
    lower_swizzle_projection,
)
from llm.frontend.wafer_frontend.lowering.swizzle_linker import (
    link_swizzle_manifest,
)
from llm.frontend.wafer_frontend.lowering.swizzle_abi import (
    allocate_swizzle_core_address_abi,
    build_swizzle_core_address_abi,
)
from llm.frontend.wafer_frontend.lowering.swizzle_standard import (
    link_swizzle_standard_program,
)
from llm.frontend.wafer_frontend.passes.project_swizzle_ir2 import (
    project_swizzle_adapter,
)
from llm.frontend.wafer_frontend.passes.project_swizzle_plan import (
    project_swizzle_plan,
)
from llm.frontend.wafer_frontend.policies.swizzle.materialize import (
    force_swizzle_deployment,
    materialize_swizzle_decision,
)
from llm.frontend.wafer_frontend.policies.swizzle.materialize_ir1 import (
    materialize_swizzle_plan,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    CommandFragment,
    ManifestInputKind,
    RecordOpcode,
)
from llm.frontend.wafer_frontend.schema.common import DType, stable_artifact_id
from llm.frontend.wafer_frontend.schema.global_action import LogicalCoreRef
from llm.frontend.wafer_frontend.schema.ir2 import BufferOwnership
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass
from llm.frontend.wafer_frontend.schema.swizzle_ir2 import SwizzleIr2Projection
from llm.frontend.wafer_frontend.schema.swizzle_abi import (
    SwizzleBufferAddressBinding,
    SwizzleBufferSliceBinding,
    SwizzleCoreAddressABI,
    SwizzleRankCoreBinding,
    SwizzleValueAddressBinding,
)
from llm.frontend.wafer_frontend.schema.swizzle_lowering import (
    SWIZZLE_LINKED_MANIFEST_SCHEMA_VERSION,
    SwizzleFinalizerContract,
    SwizzleLinkedManifest,
    SwizzleLoweredProgram,
    SwizzleManifestInputKind,
    SwizzleOpcodeRecord,
)
from llm.frontend.wafer_frontend.schema.swizzle_operand_abi import (
    SwizzleOperandABI,
    build_swizzle_operand_abi,
)
from llm.frontend.wafer_frontend.schema.swizzle_standard import (
    SwizzleStandardLinkedProgram,
)
from llm.frontend.wafer_frontend.schema.swizzle_plan import SwizzleValueUse

from swizzle_cases import build_swizzle_integration_cases
from swizzle_scale_cases import build_swizzle_scale_case, build_swizzle_scale_points


def _products(case):
    adapter = force_swizzle_deployment(case.decision)
    plan = materialize_swizzle_plan(
        case.partitioned_graph,
        case.decision,
        case.partitioned_graph.profile,
        deployment_selection=adapter.deployment_selection,
    )
    projection = project_swizzle_adapter(adapter)
    lowered = lower_swizzle_projection(plan, projection)
    linked = link_swizzle_manifest(plan, projection, lowered)
    return plan, projection, lowered, linked


def _explicit_abi(case, plan, projection):
    operand_abi = build_swizzle_operand_abi(
        case.partitioned_graph,
        plan,
        projection,
    )
    views_by_value = {}
    for view in operand_abi.operands:
        key = (view.value_ref, view.slot)
        prior = views_by_value.setdefault(key, view)
        if (
            prior.shape,
            prior.layout,
            prior.dtype,
            prior.byte_offset,
            prior.byte_extent,
        ) != (
            view.shape,
            view.layout,
            view.dtype,
            view.byte_offset,
            view.byte_extent,
        ):
            raise SchemaError("one value-slot must have one exact typed view")
    rank_cores = []
    for dag in projection.rank_dags:
        die = next(item for item in case.partitioned_graph.fabric.dies if item.id == dag.die_id)
        core = die.cores[0]
        rank_cores.append(
            SwizzleRankCoreBinding(
                dag.rank,
                LogicalCoreRef(dag.die_id, core.local_core_id),
                core.runtime_core_id,
            )
        )
    buffers = []
    values = []
    for dag in projection.rank_dags:
        logical_core = rank_cores[dag.rank].logical_core
        core_spec = next(
            core
            for die in case.partitioned_graph.fabric.dies
            if die.id == dag.die_id
            for core in die.cores
            if core.local_core_id == logical_core.local_core_id
        )
        profile = next(
            item
            for item in case.partitioned_graph.fabric.sram_profiles
            if item.id == core_spec.sram_profile_ref
        )
        region = profile.regions[0]
        address = region.base_bytes
        for buffer in dag.buffers:
            ordered_refs = tuple(sorted(buffer.value_refs))
            span = buffer.size_bytes * buffer.slot_count * len(ordered_refs)
            # Caller-owned gap for LOCAL_REDUCE input0 immediately before the
            # loop accumulator at the first buffer slice.
            address += buffer.size_bytes + 64
            buffers.append(
                SwizzleBufferAddressBinding(
                    dag.rank,
                    buffer.buffer_ref,
                    logical_core,
                    region.id,
                    address,
                    span,
                    64,
                    tuple(
                        SwizzleBufferSliceBinding(
                            ref,
                            slot,
                            (value_index * buffer.slot_count + slot)
                            * buffer.size_bytes,
                            buffer.size_bytes,
                        )
                        for value_index, ref in enumerate(ordered_refs)
                        for slot in range(buffer.slot_count)
                    ),
                )
            )
            values.extend(
                SwizzleValueAddressBinding(
                    ref,
                    slot,
                    dag.rank,
                    logical_core,
                    region.id,
                    address
                    + (value_index * buffer.slot_count + slot)
                    * buffer.size_bytes,
                    buffer.size_bytes,
                    64,
                )
                for value_index, ref in enumerate(ordered_refs)
                for slot in range(buffer.slot_count)
            )
            address += (span + 63) // 64 * 64
        for value in dag.values:
            if value.buffer_ref is None:
                values.append(
                    SwizzleValueAddressBinding(
                        value.id,
                        0,
                        dag.rank,
                        logical_core,
                        region.id,
                        address,
                        views_by_value[(value.id, 0)].byte_extent,
                        64,
                    )
                )
                address += views_by_value[(value.id, 0)].byte_extent
                address = (address + 63) // 64 * 64
    value_index = {(item.value_ref, item.slot): item for item in values}
    projection_values = {
        value.id: value for dag in projection.rank_dags for value in dag.values
    }
    for dag in projection.rank_dags:
        for task in dag.tasks:
            if task.kind.value != "reduce":
                continue
            slot_by_buffer = {use.buffer_ref: use.slot for use in task.buffer_uses}
            keys = tuple(
                (
                    ref,
                    slot_by_buffer.get(projection_values[ref].buffer_ref, 0),
                )
                for ref in task.read_value_refs + task.write_value_refs
            )
            input0, accumulator, output = (value_index[key] for key in keys)
            value_index[keys[0]] = replace(
                input0,
                address=accumulator.address - accumulator.size_bytes,
                size_bytes=accumulator.size_bytes,
            )
            value_index[keys[2]] = replace(
                output,
                address=accumulator.address,
                size_bytes=accumulator.size_bytes,
            )
    values = list(value_index.values())
    return build_swizzle_core_address_abi(
        case.partitioned_graph,
        plan,
        projection,
        rank_cores=tuple(rank_cores),
        value_bindings=tuple(values),
        buffer_bindings=tuple(buffers),
    )


# Focused tests consume the same deterministic allocator used by production;
# the expanded implementation above remains useful only as legacy fixture
# documentation until its callers are removed.
def _explicit_abi(case, plan, projection):
    return allocate_swizzle_core_address_abi(
        case.partitioned_graph,
        plan,
        projection,
    )


class SwizzleW9LoweringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.products = tuple(
            (case, *_products(case))
            for case in build_swizzle_integration_cases()
        )

    def test_three_patterns_lower_and_link_deterministically(self) -> None:
        self.assertEqual(len(self.products), 3)
        for case, plan, projection, lowered, linked in self.products:
            with self.subTest(pattern=case.pattern.value):
                lowered.validate_against(plan, projection)
                linked.validate_against(plan, projection, lowered)
                self.assertEqual(lower_swizzle_projection(plan, projection), lowered)
                self.assertEqual(
                    link_swizzle_manifest(plan, projection, lowered),
                    linked,
                )
                opcodes = {
                    opcode
                    for stream in lowered.rank_streams
                    for record in stream.records
                    for opcode in record.opcodes
                }
                self.assertTrue(
                    {
                        RecordOpcode.MATMUL,
                        RecordOpcode.DTE_SEND,
                        RecordOpcode.DTE_RECV,
                        RecordOpcode.DTE_WAIT,
                    }.issubset(opcodes)
                )
                self.assertEqual(
                    {item.kind for item in linked.input_digests},
                    set(SwizzleManifestInputKind),
                )
                self.assertIs(
                    linked.finalizer_contract,
                    SwizzleFinalizerContract.REQUIRES_EXACT_CORE_ADDRESS_ABI_V1,
                )
                self.assertTrue(linked.timing_execution)
                self.assertFalse(linked.functional_execution)
        all_opcodes = {
            opcode
            for _case, _plan, _projection, lowered, _linked in self.products
            for stream in lowered.rank_streams
            for record in stream.records
            for opcode in record.opcodes
        }
        self.assertTrue(
            {
                RecordOpcode.LOCAL_REDUCE,
                RecordOpcode.DTE_ISSUE,
                RecordOpcode.EVENT_SET,
                RecordOpcode.EVENT_WAIT,
            }.issubset(all_opcodes)
        )

    def test_standard_fragment_manifest_and_closed_source_are_exact(self) -> None:
        expected = {
            "ag_gemm": (38, 10, 16, 21, 54, 9, 2),
            "gemm_rs": (44, 14, 12, 29, 68, 9, 2),
            "gemm_ar": (50, 14, 24, 29, 68, 9, 2),
        }
        sources = []
        for case, plan, projection, lowered, _linked in self.products:
            operand_abi = build_swizzle_operand_abi(
                case.partitioned_graph, plan, projection
            )
            core_abi = _explicit_abi(case, plan, projection)
            source = link_swizzle_standard_program(
                case.partitioned_graph,
                plan,
                projection,
                lowered,
                core_abi,
                operand_abi,
            )
            source.validate_against()
            self.assertEqual(
                (
                    sum(len(stream.records) for stream in source.fragment.core_streams),
                    len(source.fragment.buffer_abi),
                    len(source.manifest.runtime_symbol_definitions),
                    len(source.manifest.program_symbol_definitions),
                    len(source.manifest.address_operand_bindings),
                    len(source.manifest.input_digests),
                    len(source.manifest.envelope.terminal_cores),
                ),
                expected[case.pattern.value],
            )
            orders = {
                item.task_ref: item.core_order
                for item in source.core_abi.task_bindings
            }
            views_by_key = {}
            for view in source.operand_abi.operands:
                views_by_key.setdefault((view.value_ref, view.slot), []).append(view)
            abi_by_binding = {
                item.binding_id: item for item in source.fragment.buffer_abi
            }
            for key, views in views_by_key.items():
                binding_id = stable_artifact_id(
                    "swizzle_standard_buffer_binding",
                    {"core_abi": source.core_abi.id, "key": key},
                    schema_version="wafer_frontend.swizzle_standard_lowering/v1alpha1",
                )
                abi = abi_by_binding[binding_id]
                if abi.ownership is BufferOwnership.ALIASED:
                    continue
                first_use = min(
                    views,
                    key=lambda item: (orders[item.task_ref], item.ordinal),
                ).use
                self.assertIs(
                    abi.ownership,
                    BufferOwnership.BORROWED
                    if first_use is SwizzleValueUse.READ
                    else BufferOwnership.OWNED,
                )
            self.assertEqual(
                {item.kind for item in source.manifest.input_digests},
                {
                    ManifestInputKind.IR1,
                    ManifestInputKind.SWIZZLE_DECISION,
                    ManifestInputKind.SWIZZLE_CANDIDATE,
                    ManifestInputKind.SWIZZLE_FUSION_PLAN,
                    ManifestInputKind.SWIZZLE_PROJECTION,
                    ManifestInputKind.SWIZZLE_LOWERED_PROGRAM,
                    ManifestInputKind.SWIZZLE_CORE_ADDRESS_ABI,
                    ManifestInputKind.SWIZZLE_OPERAND_ABI,
                    ManifestInputKind.COMMAND_FRAGMENT,
                },
            )
            sources.append(source)

        manifest = sources[0].manifest
        with self.assertRaisesRegex(SchemaError, "nine typed input digests"):
            replace(
                manifest,
                input_digests=manifest.input_digests[:-1],
            ).validate()
        with self.assertRaisesRegex(SchemaError, "source provenance"):
            replace(manifest, source_projection_id="forged_projection").validate()
        old_digest = replace(
            manifest.input_digests[0],
            schema_version="wafer_frontend.command_fragment/v1alpha12",
        )
        with self.assertRaisesRegex(SchemaError, "schema versions are not exact"):
            replace(
                manifest,
                input_digests=(old_digest, *manifest.input_digests[1:]),
            ).validate()
        restabled = SwizzleStandardLinkedProgram.create(
            ir1=sources[0].ir1,
            plan=sources[0].plan,
            projection=sources[0].projection,
            lowered=sources[0].lowered,
            core_abi=sources[0].core_abi,
            operand_abi=sources[0].operand_abi,
            fragment=sources[0].fragment,
            manifest=sources[1].manifest,
        )
        with self.assertRaisesRegex(SchemaError, "exact producer result"):
            restabled.validate_against()

        root = next(
            item
            for item in sources[0].fragment.buffer_abi
            if item.ownership is not BufferOwnership.ALIASED
        )
        forged_ownership = replace(
            root,
            ownership=(
                BufferOwnership.OWNED
                if root.ownership is BufferOwnership.BORROWED
                else BufferOwnership.BORROWED
            ),
        )
        fragment_semantic = sources[0].fragment._semantic_key()
        fragment_semantic["buffer_abi"] = tuple(
            forged_ownership if item == root else item
            for item in sources[0].fragment.buffer_abi
        )
        forged_fragment = CommandFragment.create(
            producer_pass=sources[0].fragment.producer_pass,
            **fragment_semantic,
        )
        forged_source = SwizzleStandardLinkedProgram.create(
            ir1=sources[0].ir1,
            plan=sources[0].plan,
            projection=sources[0].projection,
            lowered=sources[0].lowered,
            core_abi=sources[0].core_abi,
            operand_abi=sources[0].operand_abi,
            fragment=forged_fragment,
            manifest=sources[0].manifest,
        )
        with self.assertRaisesRegex(SchemaError, "exact producer result"):
            forged_source.validate_against()

    def test_lowered_and_linked_carriers_round_trip_strictly(self) -> None:
        for case, _plan, _projection, lowered, linked in self.products:
            with self.subTest(pattern=case.pattern.value):
                self.assertEqual(
                    loads_dataclass(
                        SwizzleLoweredProgram,
                        canonical_json(lowered),
                        path="lowered",
                    ),
                    lowered,
                )
                self.assertEqual(
                    loads_dataclass(
                        SwizzleLinkedManifest,
                        canonical_json(linked),
                        path="linked",
                    ),
                    linked,
                )

    def test_explicit_core_address_abi_counts_and_tamper_gates(self) -> None:
        expected = {
            "ag_gemm": (12, 12, 2, 6, 4),
            "gemm_rs": (14, 16, 2, 8, 0),
            "gemm_ar": (20, 16, 2, 12, 4),
        }
        abis = []
        for case, plan, projection, _lowered, _linked in self.products:
            abi = _explicit_abi(case, plan, projection)
            abis.append(abi)
            self.assertEqual(
                (
                    len(abi.task_bindings),
                    len(abi.value_bindings),
                    len(abi.buffer_bindings),
                    len(abi.runtime_bindings),
                    len(abi.barrier_events),
                ),
                expected[case.pattern.value],
            )
            self.assertEqual(
                loads_dataclass(
                    SwizzleCoreAddressABI,
                    canonical_json(abi),
                    path="abi",
                ),
                abi,
            )

        case, plan, projection, _lowered, _linked = self.products[0]
        abi = abis[0]
        semantic = abi._semantic_key()
        semantic["task_bindings"] = abi.task_bindings[1:]
        missing_task = SwizzleCoreAddressABI.create(
            producer_pass=abi.producer_pass,
            **semantic,
        )
        with self.assertRaisesRegex(SchemaError, "cover every task"):
            missing_task.validate_against(case.partitioned_graph, plan, projection)

        wait = next(
            item
            for item in abi.runtime_bindings
            if any(
                task.id == item.task_ref and task.kind.value == "wait"
                for dag in projection.rank_dags
                for task in dag.tasks
            )
        )
        semantic = abi._semantic_key()
        semantic["runtime_bindings"] = tuple(
            replace(item, token_symbol_ref="forged_token") if item == wait else item
            for item in abi.runtime_bindings
        )
        forged_token = SwizzleCoreAddressABI.create(
            producer_pass=abi.producer_pass,
            **semantic,
        )
        with self.assertRaisesRegex(SchemaError, "share the depended RECV token"):
            forged_token.validate_against(case.partitioned_graph, plan, projection)

        send_task_refs = {
            task.id
            for dag in projection.rank_dags
            for task in dag.tasks
            if task.kind.value == "send"
        }
        send = next(
            item for item in abi.runtime_bindings if item.task_ref in send_task_refs
        )
        self.assertIsNone(send.token_symbol_ref)
        semantic = abi._semantic_key()
        semantic["runtime_bindings"] = tuple(
            replace(item, token_symbol_ref="forged_send_token")
            if item == send
            else item
            for item in abi.runtime_bindings
        )
        forged_send_token = SwizzleCoreAddressABI.create(
            producer_pass=abi.producer_pass,
            **semantic,
        )
        with self.assertRaisesRegex(SchemaError, "exact flow endpoints"):
            forged_send_token.validate_against(
                case.partitioned_graph,
                plan,
                projection,
            )

        buffer = abi.buffer_bindings[0]
        semantic = abi._semantic_key()
        semantic["buffer_bindings"] = (
            replace(buffer, span_bytes=buffer.span_bytes + 64),
            *abi.buffer_bindings[1:],
        )
        forged_lifecycle = SwizzleCoreAddressABI.create(
            producer_pass=abi.producer_pass,
            **semantic,
        )
        with self.assertRaisesRegex(SchemaError, "region/span/slices"):
            forged_lifecycle.validate_against(
                case.partitioned_graph,
                plan,
                projection,
            )

        last_rank_zero = next(
            item for item in reversed(abi.task_bindings) if item.rank == 0
        )
        split_core = replace(
            last_rank_zero,
            logical_core=LogicalCoreRef(last_rank_zero.logical_core.die_id, 1),
            runtime_core_id=1,
            core_order=0,
        )
        semantic = abi._semantic_key()
        semantic["task_bindings"] = tuple(
            split_core if item == last_rank_zero else item
            for item in abi.task_bindings
        )
        forged_rank_core = SwizzleCoreAddressABI.create(
            producer_pass=abi.producer_pass,
            **semantic,
        )
        with self.assertRaisesRegex(SchemaError, "exactly one logical/runtime core"):
            forged_rank_core.validate_against(
                case.partitioned_graph,
                plan,
                projection,
            )

        transport = next(
            item for item in abi.runtime_bindings if item.flow_ref is not None
        )
        semantic = abi._semantic_key()
        semantic["runtime_bindings"] = tuple(
            replace(item, fsm_symbol_ref="forged_fsm")
            if item == transport
            else item
            for item in abi.runtime_bindings
        )
        forged_flow_runtime = SwizzleCoreAddressABI.create(
            producer_pass=abi.producer_pass,
            **semantic,
        )
        with self.assertRaisesRegex(SchemaError, "exact flow endpoints"):
            forged_flow_runtime.validate_against(
                case.partitioned_graph,
                plan,
                projection,
            )

        reduction_value_refs = {
            ref
            for dag in projection.rank_dags
            for task in dag.tasks
            if task.kind.value == "reduce"
            for ref in task.read_value_refs + task.write_value_refs
        }
        unbuffered = next(
            item
            for item in abi.value_bindings
            if item.value_ref not in reduction_value_refs
            and item.value_ref
            not in {
                ref
                for buffer_binding in abi.buffer_bindings
                for ref in (slice_binding.value_ref for slice_binding in buffer_binding.slices)
            }
        )
        semantic = abi._semantic_key()
        semantic["value_bindings"] = tuple(
            replace(item, address=abi.buffer_bindings[0].base_address)
            if item == unbuffered
            else item
            for item in abi.value_bindings
        )
        forged_alias = SwizzleCoreAddressABI.create(
            producer_pass=abi.producer_pass,
            **semantic,
        )
        with self.assertRaisesRegex(SchemaError, "must not alias"):
            forged_alias.validate_against(
                case.partitioned_graph,
                plan,
                projection,
            )

        double_case = build_swizzle_integration_cases()[0]
        double_candidate = next(
            item
            for item in double_case.decision.ranked_candidates
            if item.unroll_degree == 2
        )
        double_adapter = force_swizzle_deployment(
            double_case.decision,
            candidate_ref=double_candidate.id,
        )
        double_plan = materialize_swizzle_plan(
            double_case.partitioned_graph,
            double_case.decision,
            double_case.partitioned_graph.profile,
            deployment_selection=double_adapter.deployment_selection,
        )
        double_projection = project_swizzle_adapter(double_adapter)
        double_abi = _explicit_abi(double_case, double_plan, double_projection)
        self.assertTrue(any(buffer.slot_count == 2 for dag in double_projection.rank_dags for buffer in dag.buffers))
        slot_one = next(item for item in double_abi.value_bindings if item.slot == 1)
        slot_zero = next(
            item
            for item in double_abi.value_bindings
            if item.value_ref == slot_one.value_ref and item.slot == 0
        )
        semantic = double_abi._semantic_key()
        semantic["value_bindings"] = tuple(
            replace(item, address=slot_zero.address)
            if item == slot_one
            else item
            for item in double_abi.value_bindings
        )
        forged_slot_alias = SwizzleCoreAddressABI.create(
            producer_pass=double_abi.producer_pass,
            **semantic,
        )
        with self.assertRaisesRegex(SchemaError, "slice disagrees"):
            forged_slot_alias.validate_against(
                double_case.partitioned_graph,
                double_plan,
                double_projection,
            )

    def test_s1_s2_ag_exact_reuse_lifetimes_and_terminal_roots(self) -> None:
        for point_index, expected_chunks, expected_roots in (
            (1, 8, 16),
            (2, 16, 20),
        ):
            case = build_swizzle_scale_case(
                build_swizzle_scale_points()[point_index]
            )
            decision = case.decisions[0]
            adapter = materialize_swizzle_decision(decision)
            plan = materialize_swizzle_plan(
                case.partitioned_graph,
                decision,
                case.partitioned_graph.profile,
                deployment_selection=adapter.deployment_selection,
            )
            projection = project_swizzle_plan(
                case.partitioned_graph, plan,
            ).projection
            abi = allocate_swizzle_core_address_abi(
                case.partitioned_graph, plan, projection,
            )
            abi.validate_against(case.partitioned_graph, plan, projection)
            self.assertEqual(
                abi,
                allocate_swizzle_core_address_abi(
                    case.partitioned_graph, plan, projection,
                ),
            )
            projected_values = {
                item.id: item
                for dag in projection.rank_dags for item in dag.values
            }
            self.assertEqual(
                len({
                    (item.rank, item.storage_ref)
                    for item in abi.value_bindings
                }),
                expected_roots,
            )
            boundary_groups = {}
            for binding in abi.value_bindings:
                value = projected_values[binding.value_ref]
                if (
                    not value.producer_task_refs
                    and value.symbolic_ref.startswith(
                        f"{value.origin_ref}::chunk"
                    )
                ):
                    boundary_groups.setdefault(
                        (binding.rank, binding.storage_ref), []
                    ).append(binding)
            self.assertEqual(len(boundary_groups), len(projection.rank_dags))
            self.assertEqual(
                {len(items) for items in boundary_groups.values()},
                {expected_chunks // len(projection.rank_dags)},
            )
            self.assertTrue(all(
                left.lifetime_end_exclusive <= right.lifetime_start
                or right.lifetime_end_exclusive <= left.lifetime_start
                for items in boundary_groups.values()
                for index, left in enumerate(items)
                for right in items[index + 1 :]
            ))
            terminal_groups = {}
            for ownership in projection.output_ownership:
                for binding in abi.value_bindings:
                    value = projected_values[binding.value_ref]
                    if binding.rank == ownership.rank and any(
                        value.symbolic_ref == boundary_ref
                        or value.symbolic_ref.startswith(f"{boundary_ref}::")
                        for boundary_ref in ownership.boundary_output_refs
                    ):
                        terminal_groups.setdefault(
                            (binding.rank, binding.storage_ref), []
                        ).append(binding)
            self.assertEqual(len(terminal_groups), len(projection.rank_dags))
            for (rank, _storage_ref), chunks in terminal_groups.items():
                ordered = sorted(chunks, key=lambda item: item.storage_offset_bytes)
                self.assertEqual(len(ordered), expected_chunks)
                self.assertEqual(
                    tuple(item.storage_offset_bytes for item in ordered),
                    tuple(
                        index * ordered[0].size_bytes
                        for index in range(expected_chunks)
                    ),
                )
                self.assertEqual(
                    sum(item.size_bytes for item in ordered),
                    max(
                        item.storage_offset_bytes + item.size_bytes
                        for item in ordered
                    ),
                )
                self.assertTrue(all(
                    item.lifetime_end_exclusive
                    == len(projection.rank_dags[rank].tasks)
                    for item in ordered
                ))

            reused_pairs = tuple(
                (left, right)
                for index, left in enumerate(abi.value_bindings)
                for right in abi.value_bindings[index + 1 :]
                if left.logical_core == right.logical_core
                and left.address == right.address
                and left.size_bytes == right.size_bytes
                and left.storage_ref == right.storage_ref
                and left.storage_offset_bytes == right.storage_offset_bytes
                and projected_values[left.value_ref].buffer_ref is not None
                and projected_values[right.value_ref].buffer_ref is not None
            )
            self.assertTrue(reused_pairs)
            self.assertTrue(all(
                left.lifetime_end_exclusive <= right.lifetime_start
                or right.lifetime_end_exclusive <= left.lifetime_start
                for left, right in reused_pairs
            ))

    def test_s1_ag_root_lifetime_overlap_and_view_tamper_fail_closed(self) -> None:
        case = build_swizzle_scale_case(build_swizzle_scale_points()[1])
        decision = case.decisions[0]
        adapter = materialize_swizzle_decision(decision)
        plan = materialize_swizzle_plan(
            case.partitioned_graph,
            decision,
            case.partitioned_graph.profile,
            deployment_selection=adapter.deployment_selection,
        )
        projection = project_swizzle_plan(
            case.partitioned_graph, plan,
        ).projection
        abi = allocate_swizzle_core_address_abi(
            case.partitioned_graph, plan, projection,
        )
        values = {
            item.id: item for dag in projection.rank_dags for item in dag.values
        }

        terminal = next(
            item
            for item in abi.value_bindings
            if any(
                values[item.value_ref].symbolic_ref == boundary_ref
                or values[item.value_ref].symbolic_ref.startswith(
                    f"{boundary_ref}::"
                )
                for boundary_ref in projection.output_ownership[item.rank].boundary_output_refs
            )
        )
        semantic = abi._semantic_key()
        semantic["value_bindings"] = tuple(
            replace(item, storage_ref="forged_terminal_root")
            if item == terminal else item
            for item in abi.value_bindings
        )
        forged_root = SwizzleCoreAddressABI.create(
            producer_pass=abi.producer_pass, **semantic,
        )
        with self.assertRaisesRegex(SchemaError, "exact deterministic"):
            forged_root.validate_against(
                case.partitioned_graph, plan, projection,
            )

        live = next(
            item
            for item in abi.value_bindings
            if values[item.value_ref].buffer_ref is not None
        )
        semantic = abi._semantic_key()
        semantic["value_bindings"] = tuple(
            replace(item, lifetime_end_exclusive=item.lifetime_end_exclusive + 1)
            if item == live else item
            for item in abi.value_bindings
        )
        forged_lifetime = SwizzleCoreAddressABI.create(
            producer_pass=abi.producer_pass, **semantic,
        )
        with self.assertRaisesRegex(SchemaError, "lifetime disagrees"):
            forged_lifetime.validate_against(
                case.partitioned_graph, plan, projection,
            )

        operand_abi = build_swizzle_operand_abi(
            case.partitioned_graph, plan, projection,
        )
        typed_keys = {
            (item.value_ref, item.slot) for item in operand_abi.operands
        }
        untyped = next(
            item
            for item in abi.value_bindings
            if values[item.value_ref].buffer_ref is not None
            and (item.value_ref, item.slot) not in typed_keys
        )
        typed = next(
            item
            for item in abi.value_bindings
            if item.storage_ref == untyped.storage_ref
            and (item.value_ref, item.slot) in typed_keys
        )
        buffer = next(
            item for item in abi.buffer_bindings
            if item.rank == untyped.rank
            and item.buffer_ref == values[untyped.value_ref].buffer_ref
        )
        forged_slices = tuple(
            replace(
                item,
                offset_bytes=typed.address - buffer.base_address,
            )
            if (item.value_ref, item.slot)
            == (untyped.value_ref, untyped.slot)
            else item
            for item in buffer.slices
        )
        forged_buffer = replace(
            buffer,
            span_bytes=max(
                item.offset_bytes + item.size_bytes for item in forged_slices
            ),
            slices=forged_slices,
        )
        semantic = abi._semantic_key()
        semantic["value_bindings"] = tuple(
            replace(
                item,
                address=typed.address,
                storage_offset_bytes=typed.storage_offset_bytes,
            )
            if item == untyped else item
            for item in abi.value_bindings
        )
        semantic["buffer_bindings"] = tuple(
            forged_buffer if item == buffer else item
            for item in abi.buffer_bindings
        )
        forged_view = SwizzleCoreAddressABI.create(
            producer_pass=abi.producer_pass, **semantic,
        )
        with self.assertRaisesRegex(SchemaError, "must not alias"):
            forged_view.validate_against(
                case.partitioned_graph, plan, projection,
            )

    def test_operand_abi_exact_counts_roundtrip_and_tamper_gates(self) -> None:
        expected = {
            "ag_gemm": (20, 4, 4, 0),
            "gemm_rs": (30, 4, 6, 2),
            "gemm_ar": (34, 4, 8, 2),
        }
        products = []
        for case, plan, projection, _lowered, _linked in self.products:
            core_abi = _explicit_abi(case, plan, projection)
            operand_abi = build_swizzle_operand_abi(
                case.partitioned_graph,
                plan,
                projection,
            )
            operand_abi.validate_against(
                case.partitioned_graph,
                plan,
                projection,
                core_abi,
            )
            self.assertEqual(
                (
                    len(operand_abi.operands),
                    len(operand_abi.matmul_contracts),
                    len(operand_abi.dte_contracts),
                    len(operand_abi.reduce_contracts),
                ),
                expected[case.pattern.value],
            )
            self.assertEqual(
                loads_dataclass(
                    SwizzleOperandABI,
                    canonical_json(operand_abi),
                    path="operand_abi",
                ),
                operand_abi,
            )
            products.append((case, plan, projection, core_abi, operand_abi))

        case, plan, projection, core_abi, operand_abi = products[1]
        typed = operand_abi.operands[0]
        semantic = operand_abi._semantic_key()
        semantic["operands"] = tuple(
            replace(
                item,
                dtype=DType.FP32,
                byte_extent=item.byte_extent * 2,
            )
            if item == typed
            else item
            for item in operand_abi.operands
        )
        wrong_dtype = SwizzleOperandABI.create(
            producer_pass=operand_abi.producer_pass,
            **semantic,
        )
        with self.assertRaisesRegex(SchemaError, "origins/chunk witnesses"):
            wrong_dtype.validate_against(
                case.partitioned_graph,
                plan,
                projection,
                core_abi,
            )

        reduction_refs = {
            view.value_ref
            for contract in operand_abi.reduce_contracts
            for view in operand_abi.operands
            if view.task_ref == contract.task_ref
        }
        short_view = next(
            item
            for item in operand_abi.operands
            if item.value_ref not in reduction_refs
        )
        short_binding = next(
            item
            for item in core_abi.value_bindings
            if (item.value_ref, item.slot)
            == (short_view.value_ref, short_view.slot)
        )
        semantic = core_abi._semantic_key()
        semantic["value_bindings"] = tuple(
            replace(item, size_bytes=short_view.byte_extent - 1)
            if item == short_binding
            else item
            for item in core_abi.value_bindings
        )
        short_span = SwizzleCoreAddressABI.create(
            producer_pass=core_abi.producer_pass,
            **semantic,
        )
        with self.assertRaisesRegex(SchemaError, "exceeds address span"):
            operand_abi.validate_against(
                case.partitioned_graph,
                plan,
                projection,
                short_span,
            )

        reduce_contract = operand_abi.reduce_contracts[0]
        views = tuple(
            item
            for item in operand_abi.operands
            if item.task_ref == reduce_contract.task_ref
        )
        bindings = tuple(
            next(
                binding
                for binding in core_abi.value_bindings
                if (binding.value_ref, binding.slot)
                == (view.value_ref, view.slot)
            )
            for view in views
        )
        first, accumulator, output = bindings
        for label, target, replacement, error in (
            (
                "gap",
                first,
                replace(first, address=first.address - 64),
                "ordered contiguous",
            ),
            (
                "overlap",
                first,
                replace(first, address=first.address + 64),
                "must not alias|ordered contiguous",
            ),
            (
                "reverse",
                first,
                replace(first, address=accumulator.address + accumulator.size_bytes),
                "must not alias|ordered contiguous",
            ),
            (
                "wrong_output_alias",
                output,
                replace(output, address=output.address + 64),
                "must not alias|output alias",
            ),
        ):
            with self.subTest(tamper=label):
                semantic = core_abi._semantic_key()
                semantic["value_bindings"] = tuple(
                    replacement if item == target else item
                    for item in core_abi.value_bindings
                )
                forged = SwizzleCoreAddressABI.create(
                    producer_pass=core_abi.producer_pass,
                    **semantic,
                )
                with self.assertRaisesRegex(SchemaError, error):
                    operand_abi.validate_against(
                        case.partitioned_graph,
                        plan,
                        projection,
                        forged,
                    )

    def test_provenance_opcode_digest_and_old_schema_tamper_fail_closed(self) -> None:
        _case, plan, projection, lowered, linked = self.products[0]
        projection_semantic = projection._semantic_key()
        projection_semantic["source_candidate_ref"] = "forged_candidate"
        forged_projection = SwizzleIr2Projection.create(**projection_semantic)
        with self.assertRaisesRegex(SchemaError, "provenance is not exact"):
            lower_swizzle_projection(plan, forged_projection)

        stream = lowered.rank_streams[0]
        source_record = stream.records[0]
        forged_record = SwizzleOpcodeRecord.create(
            task=source_record.task,
            opcodes=(RecordOpcode.DTE_SEND,),
        )
        lowered_semantic = lowered._semantic_key()
        lowered_semantic["rank_streams"] = (
            replace(
                stream,
                records=(forged_record, *stream.records[1:]),
            ),
            *lowered.rank_streams[1:],
        )
        forged_lowered = SwizzleLoweredProgram.create(
            producer_pass=lowered.producer_pass,
            **lowered_semantic,
        )
        with self.assertRaisesRegex(SchemaError, "opcode sequence drifts"):
            forged_lowered.validate_against(plan, projection)

        digest = linked.input_digests[0]
        linked_semantic = linked._semantic_key()
        linked_semantic["input_digests"] = (
            replace(digest, digest="0" * 64),
            *linked.input_digests[1:],
        )
        forged_linked = SwizzleLinkedManifest.create(
            producer_pass=linked.producer_pass,
            **linked_semantic,
        )
        with self.assertRaisesRegex(SchemaError, "must exactly preserve"):
            forged_linked.validate_against(plan, projection, lowered)

        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(
                linked,
                schema_version=SWIZZLE_LINKED_MANIFEST_SCHEMA_VERSION.replace(
                    "v1alpha1",
                    "v0",
                ),
            ).validate()


if __name__ == "__main__":
    unittest.main()
