from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import (
    SchemaError,
    UnsupportedFeatureError,
)
from llm.frontend.wafer_frontend.passes.stage4_logical_expand import (
    build_stage4_fused_ir0,
)
from llm.frontend.wafer_frontend.passes.stage4_pd import build_stage4_pd_plan
from llm.frontend.wafer_frontend.schema.ir0 import (
    EdgeKind,
    IR0,
    InstanceProfileBinding,
    NodeProfileBinding,
    StateAccessMode,
)
from llm.frontend.wafer_frontend.schema.ir1 import IR1
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass

from _fixtures import valid_ir1
from test_stage4_pd import _profile, _spec


def _case(tp: int = 1):
    spec = _spec(tp, tp, fused=True)
    plan = build_stage4_pd_plan(
        spec,
        prefill_profile=_profile(prefill=True),
        decode_profile=_profile(prefill=False),
    )
    return spec, plan


class Stage4FusedIr0Test(unittest.TestCase):
    def test_ir1_node_profile_carrier_is_strict_and_round_trips(self) -> None:
        base = valid_ir1()
        decode_profile = replace(
            base.profile,
            prefill_tokens=0,
            decode_tokens=1,
            context_sum=base.profile.context_sum + 1,
            context_max=base.profile.context_max + 1,
        )
        instance_ref = base.instances[0].id
        # The legacy fixture has no Stage4 bindings; construct the two exact
        # same-instance profiles without changing physical-node ABI.
        instance_profiles = tuple(
            sorted(
                (
                    InstanceProfileBinding(instance_ref, base.profile),
                    InstanceProfileBinding(instance_ref, decode_profile),
                ),
                key=lambda item: (item.instance_ref, item.profile.stable_id()),
            )
        )
        fields = base._semantic_key()
        fields.update(
            profile=instance_profiles[0].profile,
            instance_profiles=instance_profiles,
            node_profiles=tuple(
                NodeProfileBinding(node.id, base.profile) for node in base.nodes
            ),
            pd_plan_id="stage4_fused_plan_fixture",
        )
        graph = IR1.create(producer_pass=base.producer_pass, **fields)
        graph.validate("graph")
        self.assertEqual(
            loads_dataclass(IR1, canonical_json(graph), path="graph"), graph
        )

    def test_tp1_l2_exact_topology_and_shared_state(self) -> None:
        spec, plan = _case()
        graph = build_stage4_fused_ir0(spec, plan)

        self.assertEqual(
            (
                len(graph.instances),
                len(graph.instance_profiles),
                len(graph.node_profiles),
                len(graph.nodes),
                len(graph.values),
                len(graph.edges),
                len(graph.fusion_candidates),
                len(graph.persistent_states),
                len(graph.state_accesses),
            ),
            (1, 2, 50, 50, 67, 57, 0, 19, 38),
        )
        self.assertEqual(
            {binding.instance_ref for binding in graph.instance_profiles},
            {"F0"},
        )
        self.assertEqual(
            tuple(binding.node_ref for binding in graph.node_profiles),
            tuple(node.id for node in graph.nodes),
        )
        self.assertTrue(
            all(
                binding.profile == plan.prefill_profile.key
                for binding in graph.node_profiles[:25]
            )
        )
        self.assertTrue(
            all(
                binding.profile == plan.decode_profile.key
                for binding in graph.node_profiles[25:]
            )
        )

        declarations = {item.id: item for item in graph.persistent_states}
        parameters = tuple(
            item
            for item in graph.persistent_states
            if item.identity.kind is StateKind.PARAMETER
        )
        kv_states = tuple(
            item
            for item in graph.persistent_states
            if item.identity.kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
        )
        self.assertEqual((len(parameters), len(kv_states)), (15, 4))
        values = {value.id: value for value in graph.values}
        self.assertTrue(
            all(
                len(values[item.identity.tensor_ref].consumers) == 2
                for item in parameters
            )
        )

        parameter_accesses = tuple(
            access
            for access in graph.state_accesses
            if declarations[access.state_ref].identity.kind is StateKind.PARAMETER
        )
        prefill_kv = tuple(
            access
            for access in graph.state_accesses
            if access.node_ref.startswith("prefill.")
            and declarations[access.state_ref].identity.kind
            in (StateKind.KV_KEY, StateKind.KV_VALUE)
        )
        decode_kv = tuple(
            access
            for access in graph.state_accesses
            if access.node_ref.startswith("decode.")
            and declarations[access.state_ref].identity.kind
            in (StateKind.KV_KEY, StateKind.KV_VALUE)
        )
        self.assertEqual((len(parameter_accesses), len(prefill_kv), len(decode_kv)), (30, 4, 4))
        self.assertTrue(
            all(access.mode is StateAccessMode.READ for access in parameter_accesses)
        )
        self.assertEqual(
            {
                (access.mode, access.read_offset, access.write_offset, access.write_shape)
                for access in prefill_kv
            },
            {(StateAccessMode.WRITE, None, (0, 0, 0), (8, 4, 4))},
        )
        self.assertEqual(
            {
                (
                    access.mode,
                    access.read_offset,
                    access.read_shape,
                    access.write_offset,
                    access.write_shape,
                )
                for access in decode_kv
            },
            {
                (
                    StateAccessMode.READ_WRITE,
                    (0, 0, 0),
                    (9, 4, 4),
                    (8, 0, 0),
                    (1, 4, 4),
                )
            },
        )
        cross_phase = tuple(
            edge
            for edge in graph.edges
            if edge.source_node.startswith("prefill.")
            and edge.destination_node.startswith("decode.")
        )
        self.assertEqual(len(cross_phase), 1)
        self.assertIs(cross_phase[0].kind, EdgeKind.CONTROL)
        self.assertIsNone(cross_phase[0].value_id)
        graph.validate("graph")
        self.assertEqual(build_stage4_fused_ir0(spec, plan), graph)
        self.assertEqual(
            loads_dataclass(IR0, canonical_json(graph), path="graph"), graph
        )

    def test_node_profile_and_preview_boundaries_fail_closed(self) -> None:
        spec, plan = _case()
        graph = build_stage4_fused_ir0(spec, plan)

        missing_fields = graph._semantic_key()
        missing_fields["node_profiles"] = graph.node_profiles[:-1]
        missing = IR0.create(producer_pass=graph.producer_pass, **missing_fields)
        with self.assertRaisesRegex(SchemaError, "one binding per node"):
            missing.validate("graph")

        wrong_fields = graph._semantic_key()
        wrong_fields["node_profiles"] = (
            replace(
                graph.node_profiles[0],
                profile=plan.decode_profile.key,
            ),
            *graph.node_profiles[1:],
        )
        wrong = IR0.create(producer_pass=graph.producer_pass, **wrong_fields)
        with self.assertRaisesRegex(SchemaError, "workload profile"):
            wrong.validate("graph")

        tp2_spec, tp2_plan = _case(2)
        with self.assertRaisesRegex(
            UnsupportedFeatureError, "supports TP1 only"
        ):
            build_stage4_fused_ir0(tp2_spec, tp2_plan)

        with self.assertRaisesRegex(SchemaError, "requires a fused PD plan"):
            separated_spec = _spec(1, 1)
            separated_plan = build_stage4_pd_plan(
                separated_spec,
                prefill_profile=_profile(prefill=True),
                decode_profile=_profile(prefill=False),
            )
            build_stage4_fused_ir0(separated_spec, separated_plan)


if __name__ == "__main__":
    unittest.main()
