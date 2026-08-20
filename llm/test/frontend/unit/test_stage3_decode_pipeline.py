from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import (
    build_stage3_dense_inference_oracle,
    build_global_bundle,
    link_bundle,
    logical_expand,
    lower_bundle,
    partition_bundle,
    place_bundle,
    plan_bundle,
    project_bundle,
    schedule_bundle,
)
from llm.frontend.wafer_frontend.passes.build_ir0 import build_ir0
from llm.frontend.wafer_frontend.passes.load_fabric import load_physical_fabric
from llm.frontend.wafer_frontend.passes.placement import (
    validate_placement_against,
)
from llm.frontend.wafer_frontend.schema.common import ProfileKey
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec
from llm.frontend.wafer_frontend.schema.ir0 import (
    AttentionMode,
    AttentionWorkload,
    OpKind,
    StateAccessMode,
)
from llm.frontend.wafer_frontend.schema.ir1 import SramAllocator
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind
from llm.frontend.wafer_frontend.schema.n4 import FusionPartitionContext
from llm.frontend.wafer_frontend.schema.n5 import ProjectToIR2Context
from llm.frontend.wafer_frontend.schema.ir2 import SemanticTaskKind
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.serde import from_data
from llm.frontend.wafer_frontend.schema.stage3_profile import (
    KvPageSpan,
    Stage3StaticProfile,
    StaticRequestShape,
)

from _fixtures import (
    naive_inter_die_planning_context,
    naive_intra_die_scheduling_context,
    valid_hbm_address_spaces,
    valid_spec,
)


_ROOT = Path(__file__).resolve().parents[4]
_HARDWARE = _ROOT / "llm/test/sram/hardware_numa.json"
_MAPPING = _ROOT / "llm/test/default/mapping.spec"
_CONTEXTS = (4, 8, 12, 16, 20, 24, 28, 32)


def _exact_profile() -> Stage3StaticProfile:
    requests: list[StaticRequestShape] = []
    page_start = 0
    for index, context in enumerate(_CONTEXTS):
        page_count = (context + 15) // 16
        requests.append(
            StaticRequestShape(
                request_ref=f"r{index}",
                prefill_tokens=0,
                decode_tokens=1,
                context_tokens=context,
                kv_span=KvPageSpan(page_start, page_count, 16),
            )
        )
        page_start += page_count
    request_tuple = tuple(requests)
    return Stage3StaticProfile.create(
        key=ProfileKey(
            prefill_tokens=0,
            decode_tokens=8,
            num_seqs=8,
            context_sum=144,
            context_max=32,
            kv_pages=12,
            expert_load=None,
        ),
        requests=request_tuple,
    )


def _spec(tp: int) -> ExperimentSpec:
    raw = valid_spec()
    raw["model"].update(  # type: ignore[index]
        V=32,
        H=16,
        I=32,
        NH=4,
        KVH=4,
        DH=4,
        rotary_dim=4,
        L=2,
        max_position_embeddings=128,
    )
    raw["parallel"]["instances"][0].update(  # type: ignore[index]
        role="decode",
        tp=tp,
        sp=tp > 1,
    )
    raw["workload"]["infer"]["profile"] = {  # type: ignore[index]
        "prefill_tokens": 0,
        "decode_tokens": 8,
        "num_seqs": 8,
        "context_sum": 144,
        "context_max": 32,
        "kv_pages": 12,
        "expert_load": None,
    }
    return from_data(ExperimentSpec, raw, path="spec")


class Stage3DecodePipelineTest(unittest.TestCase):
    def test_tp1_tp2_decode_ir0_ir1_state_and_oracle_are_exact(self) -> None:
        exact_profile = _exact_profile()
        for tp, placed_parameter_bytes in ((1, 12448), (2, 14656)):
            with self.subTest(tp=tp):
                spec = _spec(tp)
                template = build_ir0(spec, exact_profiles=(exact_profile,))
                expanded = logical_expand(template)
                graph = expanded.entries[0].graph
                oracle = build_stage3_dense_inference_oracle(
                    template, exact_profile, tp_degree=tp
                )
                attention = tuple(
                    node.workload
                    for node in graph.nodes
                    if node.kind is OpKind.ATTENTION
                )
                self.assertEqual(len(attention), 2)
                self.assertTrue(
                    all(type(work) is AttentionWorkload for work in attention)
                )
                self.assertTrue(
                    all(work.mode is AttentionMode.DECODE for work in attention)
                )
                self.assertEqual(
                    sum(work.query_key_pairs for work in attention),
                    oracle.logical_work.attention.query_key_pairs,
                )
                self.assertEqual(
                    sum(work.logical_kv_read_bytes for work in attention),
                    oracle.kv.logical_read_bytes,
                )
                self.assertEqual(
                    sum(work.logical_kv_write_bytes for work in attention),
                    oracle.kv.logical_write_bytes,
                )

                fabric = load_physical_fabric(_HARDWARE, _MAPPING)
                context = PlacementContext.create(
                    producer_pass="stage3_decode_placement",
                    fabric=fabric,
                    placement=spec.placement,
                    hbm_address_spaces=valid_hbm_address_spaces(fabric),
                )
                placed = place_bundle(expanded, context)
                validate_placement_against(placed, expanded, context)
                ir1 = placed.entries[0].graph
                manifest = ir1.persistent_state_manifest
                self.assertIsNotNone(manifest)
                assert manifest is not None
                parameters = tuple(
                    declaration
                    for declaration in manifest.declarations
                    if declaration.identity.kind is StateKind.PARAMETER
                )
                kv = tuple(
                    declaration
                    for declaration in manifest.declarations
                    if declaration.identity.kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
                )
                self.assertEqual(len(parameters), 15 * tp)
                self.assertEqual(len(kv), 32 * tp)
                self.assertEqual(
                    sum(item.tensor_bytes for item in parameters),
                    placed_parameter_bytes,
                )
                self.assertEqual(
                    sum(item.tensor_bytes for item in kv),
                    oracle.kv.logical_reserved_bytes,
                )
                kv_refs = {item.id for item in kv}
                self.assertEqual(
                    {
                        access.mode
                        for access in ir1.state_accesses
                        if access.state_ref in kv_refs
                    },
                    {StateAccessMode.READ_WRITE},
                )
                kv_accesses = tuple(
                    access
                    for access in ir1.state_accesses
                    if access.state_ref in kv_refs
                )
                self.assertEqual(
                    {access.write_offset for access in kv_accesses},
                    {(context - 1, 0, 0) for context in _CONTEXTS},
                )
                self.assertEqual(
                    {access.write_shape for access in kv_accesses},
                    {(1, 4 // tp, 4)},
                )
                self.assertEqual(
                    {access.read_offset for access in kv_accesses},
                    {(0, 0, 0)},
                )
                self.assertEqual(
                    {access.read_shape for access in kv_accesses},
                    {(context, 4 // tp, 4) for context in _CONTEXTS},
                )
                forged_accesses = list(ir1.state_accesses)
                target_index = next(
                    index
                    for index, access in enumerate(forged_accesses)
                    if access.state_ref in kv_refs
                )
                target_access = forged_accesses[target_index]
                access_fields = target_access._semantic_key()
                access_fields["write_offset"] = (137, 0, 0)
                forged_accesses[target_index] = type(target_access).create(
                    **access_fields,
                )
                forged_fields = ir1._semantic_key()
                forged_fields["state_accesses"] = tuple(forged_accesses)
                forged = type(ir1).create(
                    producer_pass=ir1.producer_pass,
                    **forged_fields,
                )
                with self.assertRaisesRegex(SchemaError, "contained"):
                    forged.validate()
                self.assertEqual(
                    sum(binding.size_bytes for binding in manifest.bindings),
                    placed_parameter_bytes + oracle.kv.logical_reserved_bytes,
                )

    def test_tp1_tp2_decode_projection_schedule_and_global_are_exact(self) -> None:
        expected = {
            1: (104, 104, 275, 422, 47),
            2: (280, 212, 742, 1092, 94),
        }
        for tp, (
            expected_actions,
            expected_fragments,
            expected_records,
            expected_closures,
            expected_loads,
        ) in expected.items():
            with self.subTest(tp=tp):
                spec = _spec(tp)
                template = build_ir0(
                    spec,
                    exact_profiles=(_exact_profile(),),
                )
                expanded = logical_expand(template)
                fabric = load_physical_fabric(_HARDWARE, _MAPPING)
                fabric = replace(
                    fabric,
                    sram_profiles=tuple(
                        replace(
                            profile,
                            capacity_bytes=64 * 1024,
                            regions=tuple(
                                replace(
                                    region,
                                    base_bytes=(
                                        0
                                        if index == 0
                                        else (48 + 4 * (index - 1)) * 1024
                                    ),
                                    size_bytes=(48 if index == 0 else 4) * 1024,
                                    allocator=SramAllocator.BLOCK,
                                )
                                for index, region in enumerate(profile.regions)
                            ),
                        )
                        for profile in fabric.sram_profiles
                    ),
                )
                placement_context = PlacementContext.create(
                    producer_pass="stage3_decode_projection",
                    fabric=fabric,
                    placement=spec.placement,
                    hbm_address_spaces=valid_hbm_address_spaces(fabric),
                )
                placed = place_bundle(expanded, placement_context)
                partitioned = partition_bundle(
                    placed,
                    FusionPartitionContext.create(
                        producer_pass="stage3_decode_projection"
                    ),
                )
                planned = plan_bundle(
                    partitioned,
                    naive_inter_die_planning_context(
                        "stage3_decode_projection"
                    ),
                )
                projected = project_bundle(
                    planned,
                    ProjectToIR2Context.create(
                        producer_pass="stage3_decode_projection",
                        state_transfers=(),
                    ),
                )
                scheduling_context = naive_intra_die_scheduling_context(
                    "stage3_decode_schedule"
                )
                scheduled = schedule_bundle(projected, scheduling_context)
                global_bundle = build_global_bundle(scheduled)
                global_bundle.validate_against(scheduled)
                actions = global_bundle.entries[0].global_dag.actions
                self.assertEqual(len(actions), expected_actions)
                self.assertEqual(
                    sum(action.task_kind.value == "dma_in" for action in actions),
                    47 * tp,
                )
                self.assertEqual(
                    sum(action.task_kind.value == "dma_out" for action in actions),
                    32 * tp,
                )
                placed_manifest = placed.entries[0].graph.persistent_state_manifest
                self.assertIsNotNone(placed_manifest)
                assert placed_manifest is not None
                declarations = {
                    item.id: item
                    for item in placed_manifest.declarations
                }
                kv_actions = tuple(
                    action
                    for action in actions
                    if action.dma is not None
                    and declarations[action.dma.state_ref].identity.kind
                    in (StateKind.KV_KEY, StateKind.KV_VALUE)
                )
                kv_loads = tuple(
                    action
                    for action in kv_actions
                    if action.task_kind is SemanticTaskKind.DMA_IN
                )
                kv_stores = tuple(
                    action
                    for action in kv_actions
                    if action.task_kind is SemanticTaskKind.DMA_OUT
                )
                self.assertEqual(sum(action.bytes for action in kv_loads), 18432)
                self.assertEqual(sum(action.bytes for action in kv_stores), 1024)
                self.assertEqual(
                    {action.bytes for action in kv_stores},
                    {32 // tp},
                )
                self.assertEqual(
                    {action.dma.state_offset_bytes for action in kv_stores},
                    {(context - 1) * 32 // tp for context in _CONTEXTS},
                )
                lowered = lower_bundle(global_bundle)
                linked = link_bundle(lowered)
                manifest = linked.entries[0].manifest
                leaves = tuple(
                    fragment.fragment
                    if hasattr(fragment, "fragment")
                    else fragment
                    for fragment in manifest.fragments
                )
                opcodes = Counter(
                    record.opcode.name
                    for fragment in leaves
                    for stream in fragment.core_streams
                    for record in stream.records
                )
                self.assertEqual(len(leaves), expected_fragments)
                self.assertEqual(sum(opcodes.values()), expected_records)
                self.assertEqual(
                    len(manifest.address_operand_bindings),
                    expected_closures,
                )
                self.assertEqual(opcodes["LSU_LOAD"], expected_loads)
                self.assertEqual(opcodes["LSU_STORE"], 32 * tp)
                self.assertEqual(opcodes["ATTENTION_EXACT"], 2 * tp)
                attention_literals = tuple(
                    {
                        operand.name: operand.literal_value
                        for operand in record.operands
                        if operand.literal_value is not None
                    }
                    for fragment in leaves
                    for stream in fragment.core_streams
                    for record in stream.records
                    if record.opcode.name == "ATTENTION_EXACT"
                )
                self.assertEqual(
                    {
                        (
                            item["mode"],
                            item["query_key_pairs"],
                            item["rank_kv_read_bytes"],
                            item["rank_kv_write_bytes"],
                        )
                        for item in attention_literals
                    },
                    {(2, 144, 9216 // tp, 512 // tp)},
                )


if __name__ == "__main__":
    unittest.main()
