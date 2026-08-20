from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import (
    SchemaError,
    UnsupportedFeatureError,
)
from llm.frontend.wafer_frontend.passes.stage4_pd import (
    build_stage4_pd_oracle,
    build_stage4_pd_plan,
)
from llm.frontend.wafer_frontend.passes.build_ir0 import (
    build_ir0_template_for_profile,
)
from llm.frontend.wafer_frontend.passes.stage4_logical_expand import (
    build_stage4_separated_ir0,
)
from llm.frontend.wafer_frontend.passes.placement import place_ir0
from llm.frontend.wafer_frontend.passes.validate_ir0 import DenseIR0Validator
from llm.frontend.wafer_frontend.schema.ir0 import (
    EdgeKind,
    OpKind,
    StateAccessMode,
)
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind
from llm.frontend.wafer_frontend.schema.common import ProfileKey
from llm.frontend.wafer_frontend.schema.experiment import (
    ExperimentSpec,
)
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    from_data,
    loads_dataclass,
)
from llm.frontend.wafer_frontend.schema.stage3_profile import (
    KvPageSpan,
    Stage3StaticProfile,
    StaticRequestShape,
)
from llm.frontend.wafer_frontend.schema.stage4_pd import (
    Stage4KvEndpointPairMetrics,
    Stage4KvReshardKind,
    Stage4PdMode,
    Stage4PdOracle,
    Stage4PdPlan,
)

from _fixtures import valid_hbm_address_spaces, valid_ir1, valid_spec


def _profile(*, prefill: bool) -> Stage3StaticProfile:
    request = StaticRequestShape(
        request_ref="request_0",
        prefill_tokens=8 if prefill else 0,
        decode_tokens=0 if prefill else 1,
        context_tokens=8 if prefill else 9,
        kv_span=KvPageSpan(
            page_start=0,
            page_count=1,
            page_size_tokens=16,
        ),
    )
    return Stage3StaticProfile.create(
        key=ProfileKey(
            prefill_tokens=request.prefill_tokens,
            decode_tokens=request.decode_tokens,
            num_seqs=1,
            context_sum=request.context_tokens,
            context_max=request.context_tokens,
            kv_pages=1,
            expert_load=None,
        ),
        requests=(request,),
    )


def _profile_data(profile: Stage3StaticProfile) -> dict[str, object]:
    key = profile.key
    return {
        "prefill_tokens": key.prefill_tokens,
        "decode_tokens": key.decode_tokens,
        "num_seqs": key.num_seqs,
        "context_sum": key.context_sum,
        "context_max": key.context_max,
        "kv_pages": key.kv_pages,
        "expert_load": None,
    }


def _spec(
    prefill_tp: int,
    decode_tp: int,
    *,
    fused: bool = False,
    decode_replicas: int = 1,
    selected_decode: int = 0,
) -> ExperimentSpec:
    prefill = _profile(prefill=True)
    decode = _profile(prefill=False)
    raw = valid_spec()
    raw["model"].update(  # type: ignore[index,union-attr]
        {
            "V": 32,
            "H": 16,
            "I": 32,
            "NH": 4,
            "KVH": 4,
            "DH": 4,
            "rotary_dim": 4,
            "L": 2,
            "max_position_embeddings": 128,
        }
    )
    if fused:
        instances = [
            {
                "id": "F0",
                "role": "both",
                "tp": prefill_tp,
                "sp": False,
            }
        ]
        prefill_ref = decode_ref = "F0"
        raw["placement"] = {"strategy": "compact", "groups": []}
    else:
        instances = [
            {
                "id": "P0",
                "role": "prefill",
                "tp": prefill_tp,
                "sp": False,
            },
            *[
                {
                    "id": f"D{index}",
                    "role": "decode",
                    "tp": decode_tp,
                    "sp": False,
                }
                for index in range(decode_replicas)
            ],
        ]
        prefill_ref = "P0"
        decode_ref = f"D{selected_decode}"
        groups = []
        next_die = 0
        for instance in instances:
            tp = int(instance["tp"])
            groups.append(
                {
                    "instance_id": instance["id"],
                    "mesh_ref": f"{instance['id']}.mesh.tp",
                    "die_ids": list(range(next_die, next_die + tp)),
                }
            )
            next_die += tp
        raw["placement"] = {
            "strategy": "explicit",
            "groups": sorted(
                groups,
                key=lambda item: (item["instance_id"], item["mesh_ref"]),
            ),
        }
    raw["parallel"] = {"instances": instances}
    raw["workload"]["infer"] = {  # type: ignore[index]
        "source": "pd_static",
        "output": "logits",
        "pd_static": {
            "prefill_profile": _profile_data(prefill),
            "decode_profile": _profile_data(decode),
            "prefill_instance_ref": prefill_ref,
            "decode_instance_ref": decode_ref,
        },
    }
    return from_data(ExperimentSpec, raw, path="spec")


def _build(
    prefill_tp: int,
    decode_tp: int,
    **kwargs: object,
) -> tuple[Stage4PdPlan, Stage4PdOracle]:
    plan = build_stage4_pd_plan(
        _spec(prefill_tp, decode_tp, **kwargs),
        prefill_profile=_profile(prefill=True),
        decode_profile=_profile(prefill=False),
    )
    return plan, build_stage4_pd_oracle(plan)


class Stage4PdTest(unittest.TestCase):
    def test_dense_validator_uses_exact_instance_profile_bindings(self) -> None:
        spec = _spec(1, 1)
        plan = build_stage4_pd_plan(
            spec,
            prefill_profile=_profile(prefill=True),
            decode_profile=_profile(prefill=False),
        )
        graph = build_stage4_separated_ir0(spec, plan)
        DenseIR0Validator.validate(graph)

        profiles = {
            binding.instance_ref: binding.profile
            for binding in graph.instance_profiles
        }
        swapped_bindings = tuple(
            replace(
                binding,
                profile=profiles[
                    "D0" if binding.instance_ref == "P0" else "P0"
                ],
            )
            for binding in graph.instance_profiles
        )
        misbound_fields = graph._semantic_key()
        misbound_fields["instance_profiles"] = swapped_bindings
        misbound_fields["profile"] = swapped_bindings[0].profile
        misbound = type(graph).create(
            producer_pass=graph.producer_pass,
            **misbound_fields,
        )
        with self.assertRaisesRegex(
            SchemaError,
            "attention workload profile must belong to its instance",
        ):
            DenseIR0Validator.validate(misbound)

        prefill_embedding = next(
            node
            for node in graph.nodes
            if node.instance_id == "P0" and node.kind is OpKind.EMBEDDING
        )
        decode_embedding = next(
            node
            for node in graph.nodes
            if node.instance_id == "D0" and node.kind is OpKind.EMBEDDING
        )
        tampered_nodes = tuple(
            replace(node, workload=decode_embedding.workload)
            if node.id == prefill_embedding.id
            else node
            for node in graph.nodes
        )
        tampered_fields = graph._semantic_key()
        tampered_fields["nodes"] = tampered_nodes
        cross_instance_workload = type(graph).create(
            producer_pass=graph.producer_pass,
            **tampered_fields,
        )
        with self.assertRaisesRegex(
            SchemaError,
            "embedding workload must exactly match profile",
        ):
            DenseIR0Validator.validate(cross_instance_workload)

    def test_separated_ir0_is_one_namespaced_plan_bound_graph(self) -> None:
        spec = _spec(1, 1)
        plan = build_stage4_pd_plan(
            spec,
            prefill_profile=_profile(prefill=True),
            decode_profile=_profile(prefill=False),
        )
        graph = build_stage4_separated_ir0(spec, plan)
        self.assertEqual(graph.producer_pass, "stage4_logical_expand")
        self.assertEqual(graph.pd_plan_id, plan.id)
        self.assertEqual(
            tuple(instance.id for instance in graph.instances),
            ("P0", "D0"),
        )
        self.assertEqual(len(graph.instance_profiles), 2)
        self.assertEqual(
            (
                len(graph.nodes),
                len(graph.values),
                len(graph.edges),
                len(graph.fusion_candidates),
                len(graph.persistent_states),
                len(graph.state_accesses),
            ),
            (50, 82, 57, 0, 38, 38),
        )
        self.assertTrue(
            all(node.id.startswith("prefill.") for node in graph.nodes[:25])
        )
        self.assertTrue(
            all(node.id.startswith("decode.") for node in graph.nodes[25:])
        )
        controls = tuple(
            edge for edge in graph.edges if edge.kind is EdgeKind.CONTROL
        )
        self.assertEqual(len(controls), 1)
        self.assertTrue(controls[0].source_node.startswith("prefill."))
        self.assertTrue(controls[0].destination_node.startswith("decode."))
        self.assertEqual(
            len({state.identity.id for state in graph.persistent_states}),
            len(graph.persistent_states),
        )
        states = {state.id: state for state in graph.persistent_states}
        kv_accesses = tuple(
            access
            for access in graph.state_accesses
            if states[access.state_ref].identity.kind
            in (StateKind.KV_KEY, StateKind.KV_VALUE)
        )
        self.assertEqual(len(kv_accesses), 8)
        prefill_kv_accesses = tuple(
            access
            for access in kv_accesses
            if access.node_ref.startswith("prefill.")
        )
        decode_kv_accesses = tuple(
            access
            for access in kv_accesses
            if access.node_ref.startswith("decode.")
        )
        self.assertEqual(len(prefill_kv_accesses), 4)
        self.assertEqual(len(decode_kv_accesses), 4)
        self.assertTrue(
            all(
                access.mode is StateAccessMode.WRITE
                for access in prefill_kv_accesses + decode_kv_accesses
            )
        )
        self.assertFalse(
            any(
                access.mode
                in (StateAccessMode.READ, StateAccessMode.READ_WRITE)
                for access in decode_kv_accesses
            )
        )
        self.assertTrue(
            all(
                access.read_offset is None and access.read_shape is None
                for access in decode_kv_accesses
            )
        )
        self.assertEqual(
            {
                (access.write_offset[0], access.write_shape[0])
                for access in decode_kv_accesses
            },
            {(8, 1)},
        )
        graph.validate("test.stage4_separated_ir0")
        self.assertEqual(build_stage4_separated_ir0(spec, plan), graph)
        self.assertEqual(
            loads_dataclass(type(graph), canonical_json(graph), path="graph"),
            graph,
        )

        wrong_source = Stage4PdPlan.create(
            **(plan._semantic_key() | {"source_spec_digest": "0" * 64})
        )
        with self.assertRaisesRegex(SchemaError, "does not match"):
            build_stage4_separated_ir0(spec, wrong_source)
        fused_spec = _spec(1, 1, fused=True)
        fused_plan = build_stage4_pd_plan(
            fused_spec,
            prefill_profile=_profile(prefill=True),
            decode_profile=_profile(prefill=False),
        )
        with self.assertRaisesRegex(
            UnsupportedFeatureError, "state unification"
        ):
            build_stage4_separated_ir0(fused_spec, fused_plan)

    def test_per_instance_templates_bind_exact_phase_profiles(self) -> None:
        spec = _spec(1, 1)
        prefill = build_ir0_template_for_profile(
            spec,
            instance_ref="P0",
            exact_profile=_profile(prefill=True),
        )
        decode = build_ir0_template_for_profile(
            spec,
            instance_ref="D0",
            exact_profile=_profile(prefill=False),
        )
        self.assertEqual(prefill.instance.id, "P0")
        self.assertEqual(decode.instance.id, "D0")
        self.assertEqual(prefill.instance.parallel.tp, 1)
        self.assertEqual(decode.instance.parallel.tp, 1)
        self.assertEqual(prefill.profiles[0].key.prefill_tokens, 8)
        self.assertEqual(decode.profiles[0].key.decode_tokens, 1)

        with self.assertRaisesRegex(SchemaError, "unknown instance"):
            build_ir0_template_for_profile(
                spec,
                instance_ref="missing",
                exact_profile=_profile(prefill=True),
            )
        foreign_request = replace(
            _profile(prefill=True).requests[0],
            prefill_tokens=4,
            context_tokens=4,
        )
        foreign = Stage3StaticProfile.create(
            key=ProfileKey(
                prefill_tokens=4,
                decode_tokens=0,
                num_seqs=1,
                context_sum=4,
                context_max=4,
                kv_pages=1,
                expert_load=None,
            ),
            requests=(foreign_request,),
        )
        with self.assertRaisesRegex(SchemaError, "absent from the ExperimentSpec"):
            build_ir0_template_for_profile(
                spec,
                instance_ref="P0",
                exact_profile=foreign,
            )

    def test_separated_ir0_places_each_instance_on_its_explicit_die(self) -> None:
        spec = _spec(1, 1)
        plan = build_stage4_pd_plan(
            spec,
            prefill_profile=_profile(prefill=True),
            decode_profile=_profile(prefill=False),
        )
        graph = build_stage4_separated_ir0(spec, plan)
        fabric = valid_ir1().fabric
        context = PlacementContext.create(
            producer_pass="test_stage4_pd",
            fabric=fabric,
            placement=spec.placement,
            hbm_address_spaces=valid_hbm_address_spaces(fabric),
        )
        placed = place_ir0(graph, context)
        self.assertEqual(
            tuple(
                (
                    group.instance_id,
                    tuple(
                        (placement.rank, placement.die_id)
                        for placement in group.placements
                    ),
                )
                for group in placed.groups
            ),
            (("P0", ((0, 0),)), ("D0", ((0, 1),))),
        )
        self.assertEqual(
            tuple(
                (instance.id, instance.die_region)
                for instance in placed.instances
            ),
            (("P0", (0,)), ("D0", (1,))),
        )
        self.assertEqual(len(placed.state_accesses), 38)
        self.assertIsNotNone(placed.persistent_state_manifest)
        assert placed.persistent_state_manifest is not None
        self.assertEqual(len(placed.persistent_state_manifest.bindings), 38)

    def test_fused_handoff_is_not_applicable_and_zero(self) -> None:
        plan, oracle = _build(2, 2, fused=True)
        self.assertIs(plan.mode, Stage4PdMode.FUSED)
        self.assertIs(plan.reshard, Stage4KvReshardKind.NONE)
        self.assertEqual(plan.handoffs, ())
        self.assertFalse(oracle.handoff_applicable)
        self.assertEqual(oracle.endpoint_pair_metrics, ())
        self.assertEqual(oracle.unique_endpoint_route_count, 0)
        self.assertEqual(
            (
                oracle.logical_unique_bytes,
                oracle.delivered_bytes,
                oracle.rank_flow_count,
                oracle.state_transfer_count,
                oracle.decode_wait_dependency_count,
            ),
            (0, 0, 0, 0, 0),
        )

    def test_endpoint_pair_aggregates_are_exact_for_pds_and_reshard(self) -> None:
        for prefill_tp, decode_tp, expected_pairs in (
            (1, 1, ((0, 0, 1024, 1024, 4),)),
            (
                2,
                1,
                (
                    (0, 0, 512, 512, 4),
                    (1, 0, 512, 512, 4),
                ),
            ),
            (
                1,
                2,
                (
                    (0, 0, 512, 512, 4),
                    (0, 1, 512, 512, 4),
                ),
            ),
        ):
            with self.subTest(prefill_tp=prefill_tp, decode_tp=decode_tp):
                plan, oracle = _build(prefill_tp, decode_tp)
                self.assertEqual(
                    tuple(
                        (
                            metric.source_rank,
                            metric.destination_rank,
                            metric.logical_unique_bytes,
                            metric.delivered_bytes,
                            metric.state_transfer_count,
                        )
                        for metric in oracle.endpoint_pair_metrics
                    ),
                    expected_pairs,
                )
                self.assertEqual(
                    oracle.unique_endpoint_route_count, len(expected_pairs)
                )
                self.assertEqual(
                    sum(
                        metric.delivered_bytes
                        for metric in oracle.endpoint_pair_metrics
                    ),
                    1024,
                )
                self.assertEqual(
                    sum(
                        metric.state_transfer_count
                        for metric in oracle.endpoint_pair_metrics
                    ),
                    4 if (prefill_tp, decode_tp) == (1, 1) else 8,
                )
                oracle.validate_against(plan)

    def test_equal_tp_and_gather_have_exact_head_intersections(self) -> None:
        equal_plan, equal = _build(2, 2)
        gather_plan, gather = _build(2, 1)
        for plan, oracle, kind, destination_ranks in (
            (
                equal_plan,
                equal,
                Stage4KvReshardKind.ONE_TO_ONE,
                (0, 1),
            ),
            (
                gather_plan,
                gather,
                Stage4KvReshardKind.GATHER,
                (0, 0),
            ),
        ):
            with self.subTest(kind=kind.value):
                self.assertIs(plan.reshard, kind)
                self.assertEqual(len(plan.handoffs), 2)
                self.assertEqual(
                    tuple(flow.head_slice.start for flow in plan.handoffs[0].flows),
                    (0, 2),
                )
                self.assertEqual(
                    tuple(flow.head_slice.count for flow in plan.handoffs[0].flows),
                    (2, 2),
                )
                self.assertEqual(
                    tuple(flow.destination_rank for flow in plan.handoffs[0].flows),
                    destination_ranks,
                )
                self.assertEqual(
                    tuple(flow.bytes for flow in plan.handoffs[0].flows),
                    (256, 256),
                )
                self.assertEqual(
                    (
                        oracle.logical_unique_bytes,
                        oracle.delivered_bytes,
                        oracle.rank_flow_count,
                        oracle.state_transfer_count,
                        oracle.decode_wait_dependency_count,
                    ),
                    (1024, 1024, 4, 8, 8),
                )

    def test_scatter_and_decode_replica_selection_are_explicit(self) -> None:
        scatter, oracle = _build(
            1,
            2,
            decode_replicas=2,
            selected_decode=1,
        )
        self.assertIs(scatter.reshard, Stage4KvReshardKind.SCATTER)
        self.assertEqual(scatter.decode_instance_ref, "D1")
        self.assertEqual(
            tuple(
                (flow.source_rank, flow.destination_rank)
                for flow in scatter.handoffs[0].flows
            ),
            ((0, 0), (0, 1)),
        )
        oracle.validate_against(scatter)

    def test_tp4_equal_gather_and_scatter_skip_empty_head_intersections(
        self,
    ) -> None:
        for prefill_tp, decode_tp, expected_kind in (
            (4, 4, Stage4KvReshardKind.ONE_TO_ONE),
            (4, 2, Stage4KvReshardKind.GATHER),
            (2, 4, Stage4KvReshardKind.SCATTER),
        ):
            with self.subTest(prefill_tp=prefill_tp, decode_tp=decode_tp):
                plan, oracle = _build(prefill_tp, decode_tp)
                self.assertIs(plan.reshard, expected_kind)
                self.assertEqual(len(plan.handoffs), 2)
                self.assertEqual(len(plan.handoffs[0].flows), 4)
                self.assertTrue(
                    all(
                        flow.head_slice.count == 1
                        for flow in plan.handoffs[0].flows
                    )
                )
                self.assertTrue(
                    all(flow.bytes == 128 for flow in plan.handoffs[0].flows)
                )
                self.assertEqual(
                    (
                        oracle.logical_unique_bytes,
                        oracle.delivered_bytes,
                        oracle.rank_flow_count,
                        oracle.state_transfer_count,
                    ),
                    (1024, 1024, 8, 16),
                )

    def test_strict_serde_stable_id_and_independent_tamper_gates(self) -> None:
        plan, oracle = _build(2, 1)
        self.assertEqual(
            loads_dataclass(Stage4PdPlan, canonical_json(plan), path="plan"),
            plan,
        )
        self.assertEqual(
            loads_dataclass(
                Stage4PdOracle, canonical_json(oracle), path="oracle"
            ),
            oracle,
        )
        with self.assertRaisesRegex(SchemaError, "intersection matrix"):
            Stage4PdPlan.create(
                **(
                    plan._semantic_key()
                    | {"handoffs": plan.handoffs[:-1]}
                )
            )
        with self.assertRaisesRegex(SchemaError, "aggregate metrics"):
            Stage4PdOracle.create(
                **(
                    oracle._semantic_key()
                    | {
                        "logical_unique_bytes":
                            oracle.logical_unique_bytes + 1
                    }
                )
            )
        restable = replace(
            oracle,
            decode_wait_dependency_count=0,
        )
        with self.assertRaisesRegex(SchemaError, "wait for every"):
            restable.validate_against(plan)

        endpoint_pairs = oracle.endpoint_pair_metrics
        self.assertEqual(len(endpoint_pairs), 2)
        merged_pair = Stage4KvEndpointPairMetrics(
            source_rank=endpoint_pairs[0].source_rank,
            destination_rank=endpoint_pairs[0].destination_rank,
            logical_unique_bytes=oracle.logical_unique_bytes,
            delivered_bytes=oracle.delivered_bytes,
            state_transfer_count=oracle.state_transfer_count,
        )
        missing_endpoint = Stage4PdOracle.create(
            **(
                oracle._semantic_key()
                | {
                    "endpoint_pair_metrics": (merged_pair,),
                    "unique_endpoint_route_count": 1,
                }
            )
        )
        with self.assertRaisesRegex(SchemaError, "source plan"):
            missing_endpoint.validate_against(plan)

        with self.assertRaisesRegex(SchemaError, "canonical"):
            Stage4PdOracle.create(
                **(
                    oracle._semantic_key()
                    | {
                        "endpoint_pair_metrics": tuple(
                            reversed(endpoint_pairs)
                        )
                    }
                )
            )

        byte_shift = (
            replace(
                endpoint_pairs[0],
                logical_unique_bytes=endpoint_pairs[0].logical_unique_bytes + 1,
                delivered_bytes=endpoint_pairs[0].delivered_bytes + 1,
            ),
            replace(
                endpoint_pairs[1],
                logical_unique_bytes=endpoint_pairs[1].logical_unique_bytes - 1,
                delivered_bytes=endpoint_pairs[1].delivered_bytes - 1,
            ),
        )
        byte_tamper = Stage4PdOracle.create(
            **(
                oracle._semantic_key()
                | {"endpoint_pair_metrics": byte_shift}
            )
        )
        with self.assertRaisesRegex(SchemaError, "source plan"):
            byte_tamper.validate_against(plan)

        count_shift = (
            replace(
                endpoint_pairs[0],
                state_transfer_count=endpoint_pairs[0].state_transfer_count + 2,
            ),
            replace(
                endpoint_pairs[1],
                state_transfer_count=endpoint_pairs[1].state_transfer_count - 2,
            ),
        )
        count_tamper = Stage4PdOracle.create(
            **(
                oracle._semantic_key()
                | {"endpoint_pair_metrics": count_shift}
            )
        )
        with self.assertRaisesRegex(SchemaError, "source plan"):
            count_tamper.validate_against(plan)

        with self.assertRaisesRegex(SchemaError, "unique endpoint"):
            Stage4PdOracle.create(
                **(
                    oracle._semantic_key()
                    | {
                        "unique_endpoint_route_count":
                            oracle.unique_endpoint_route_count + 1
                    }
                )
            )

        old_version = replace(
            oracle,
            schema_version="wafer_frontend.stage4_pd_oracle/v1alpha1",
        )
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            old_version.validate_against(plan)

    def test_experiment_pd_topology_and_placement_fail_closed(self) -> None:
        valid = _spec(2, 1)
        self.assertEqual(len(valid.parallel.instances), 2)

        raw = valid_spec()
        raw["schema_version"] = "wafer_frontend.experiment/v1alpha3"
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            from_data(ExperimentSpec, raw, path="spec")

        encoded = canonical_json(valid)
        restored = loads_dataclass(ExperimentSpec, encoded, path="spec")
        self.assertEqual(restored, valid)

        for mutation, message in (
            (
                lambda data: data["parallel"]["instances"].reverse(),
                "canonical role/id order",
            ),
            (
                lambda data: data["placement"]["groups"][0].update(
                    {"die_ids": [0]}
                ),
                "disjoint",
            ),
            (
                lambda data: data["workload"]["infer"]["pd_static"].update(
                    {"decode_instance_ref": "missing"}
                ),
                "unknown instance",
            ),
        ):
            with self.subTest(message=message):
                data = __import__("json").loads(encoded)
                mutation(data)
                with self.assertRaisesRegex(SchemaError, message):
                    from_data(ExperimentSpec, data, path="spec")


if __name__ == "__main__":
    unittest.main()
