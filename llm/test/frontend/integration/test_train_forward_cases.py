from __future__ import annotations

from collections import Counter
from dataclasses import fields
from pathlib import Path
import unittest

from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    CommandFragment,
    FragmentKind,
    RegionManifest,
)
from llm.frontend.wafer_frontend.schema.n6 import _leaf_fragment

from llm.test.frontend.integration.train_forward_cases import (
    TrainForwardCase,
    build_train_forward_case,
)


_ROOT = Path(__file__).resolve().parents[4]


class TrainForwardCasesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = build_train_forward_case()

    def test_self_contained_formal_chain_and_exact_goldens(self) -> None:
        case = self.case
        oracle = case.oracle
        self.assertEqual(
            (
                oracle.parameters.unique_tensor_count,
                oracle.parameters.unique_elements,
                oracle.parameters.unique_bytes,
                oracle.parameters.tp_placed_elements,
                oracle.parameters.tp_placed_bytes,
                oracle.parameters.dp_replicated_bytes,
            ),
            (15, 6224, 12448, 7328, 14656, 29312),
        )
        self.assertEqual(
            (
                oracle.gemm_flops_per_microbatch,
                oracle.attention_query_key_pairs_per_microbatch,
                oracle.attention_flops_per_microbatch,
                oracle.logical_forward_flops_per_microbatch,
                oracle.rank_forward_flops_per_microbatch,
                oracle.cluster_forward_flops_per_step,
            ),
            (90112, 72, 4608, 94720, 47360, 378880),
        )
        self.assertEqual(
            (
                oracle.collectives.node_count,
                oracle.collectives.logical_tensor_bytes_per_node,
                oracle.collectives.group_payload_bytes_per_microbatch,
                oracle.collectives.cluster_step_group_payload_bytes,
            ),
            (8, 256, 2048, 8192),
        )
        self.assertEqual(
            (
                oracle.ce.logical_rows,
                oracle.ce.rank_rows,
                oracle.ce.logical_label_bytes,
                oracle.ce.rank_label_bytes,
                oracle.ce.logical_loss_bytes,
                oracle.ce.rank_loss_bytes,
            ),
            (8, 4, 32, 16, 32, 16),
        )

        case.placed.validate()
        case.partitioned.validate_against(
            case.placed,
            case.partition_context,
        )
        case.planned.validate_against(
            case.partitioned,
            case.planning_context,
        )
        case.projected.validate_against(
            case.planned,
            case.projection_context,
        )
        case.scheduled.validate_against(
            case.projected,
            case.scheduling_context,
        )
        case.global_action.validate_against(case.scheduled)
        case.lowered.validate_against(case.global_action)
        case.linked.validate_against(case.lowered)

        self.assertEqual(
            tuple(
                tuple(
                    placement.die_id
                    for placement in replica.graph.groups[0].placements
                )
                for replica in case.placed.replicas
            ),
            ((0, 1), (2, 3)),
        )
        self.assertEqual(
            tuple(
                sum(
                    binding.size_bytes
                    for binding in replica.graph.persistent_state_manifest.bindings
                )
                for replica in case.placed.replicas
            ),
            (14656, 14656),
        )
        self.assertEqual(
            tuple(
                (len(replica.fusion_plans), len(replica.standalone_plans))
                for replica in case.planned.replicas
            ),
            ((4, 4), (4, 4)),
        )
        self.assertEqual(
            tuple(
                (
                    sum(len(dag.tasks) for dag in replica.projection.dags),
                    sum(
                        len(dag.state_staging_values)
                        for dag in replica.projection.dags
                    ),
                )
                for replica in case.projected.replicas
            ),
            ((154, 30), (154, 30)),
        )
        self.assertEqual(
            tuple(
                (
                    sum(
                        len(schedule.placements)
                        for schedule in replica.schedule_set.schedules
                    ),
                    sum(
                        len(schedule.buffer_bindings)
                        for schedule in replica.schedule_set.schedules
                    ),
                    sum(
                        len(schedule.task_state_uses)
                        for schedule in replica.schedule_set.schedules
                    ),
                )
                for replica in case.scheduled.replicas
            ),
            ((154, 134, 30), (154, 134, 30)),
        )
        self.assertEqual(
            tuple(len(replica.global_dag.actions) for replica in case.global_action.replicas),
            (154, 154),
        )

        expected_fragment_types = Counter({CommandFragment: 78, RegionManifest: 8})
        expected_fragment_kinds = Counter(
            {
                FragmentKind.COARSE: 44,
                FragmentKind.STATE_IO: 30,
                FragmentKind.ISA_REGION: 8,
                FragmentKind.STANDALONE_COLLECTIVE: 4,
            }
        )
        for replica in case.lowered.replicas:
            leaves = tuple(_leaf_fragment(item) for item in replica.fragments)
            self.assertEqual(len(replica.fragments), 86)
            self.assertEqual(Counter(type(item) for item in replica.fragments), expected_fragment_types)
            self.assertEqual(Counter(item.kind for item in leaves), expected_fragment_kinds)
            self.assertEqual(
                (
                    sum(
                        len(stream.records)
                        for leaf in leaves
                        for stream in leaf.core_streams
                    ),
                    sum(
                        len(stream.address_relocations)
                        for leaf in leaves
                        for stream in leaf.core_streams
                    ),
                    sum(
                        len(stream.runtime_relocations)
                        for leaf in leaves
                        for stream in leaf.core_streams
                    ),
                ),
                (498, 826, 144),
            )

        manifest = case.linked.manifest
        self.assertEqual(
            (
                len(manifest.fragments),
                len(manifest.fragment_interfaces),
                len(manifest.core_streams),
                sum(len(stream.records) for stream in manifest.core_streams),
                len(manifest.runtime_symbol_definitions),
                len(manifest.program_symbol_definitions),
                len(manifest.address_operand_bindings),
                len(manifest.state_operand_bindings),
                len(manifest.input_digests),
            ),
            (172, 172, 4, 996, 164, 613, 1592, 60, 213),
        )

        self.assertEqual(
            case.runtime_inputs.source_hardware_path,
            _ROOT / "notes/frontend/examples/hardware_2x2.json",
        )
        self.assertEqual(
            case.runtime_inputs.source_mapping_path,
            _ROOT / "llm/test/default/mapping.spec",
        )
        self.assertEqual(
            case.runtime_inputs.hardware_json,
            case.runtime_inputs.source_hardware_path.read_text(encoding="utf-8"),
        )
        self.assertEqual(
            case.runtime_inputs.mapping_text,
            case.runtime_inputs.source_mapping_path.read_text(encoding="utf-8"),
        )
        self.assertEqual(
            {item.name for item in fields(TrainForwardCase)}
            & {"profile", "manifest", "program_io"},
            set(),
        )

    def test_builder_is_deterministic(self) -> None:
        rebuilt = build_train_forward_case()
        self.assertEqual(rebuilt, self.case)


if __name__ == "__main__":
    unittest.main()
