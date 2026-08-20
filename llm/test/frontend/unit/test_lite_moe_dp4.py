from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

from llm.frontend.wafer_frontend import (
    lowering as public_lowering,
    passes as public_passes,
    schema as public_schema,
)
from llm.frontend.wafer_frontend.lowering import lite_moe_dp4 as direct_dp4_lowering
from llm.frontend.wafer_frontend.lowering import lite_moe_dp4_backward_linker
from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.lite_moe_dp4 import (
    build_lite_moe_dp4_ir0_adapter,
    build_lite_moe_dp4_oracle,
    build_lite_moe_dp4_spec,
    build_lite_moe_dp4_topology,
)
from llm.frontend.wafer_frontend.passes.lite_moe_dp4_backward import (
    build_lite_moe_dp4_backward,
    validate_lite_moe_dp4_backward,
)
from llm.frontend.wafer_frontend.passes.lite_moe_dp4_execution import (
    build_lite_moe_dp4_execution_case,
    validate_lite_moe_dp4_execution_case,
)
from llm.frontend.wafer_frontend.passes.lite_moe_dp4_train_forward import (
    build_lite_moe_dp4_train_forward,
    validate_lite_moe_dp4_train_forward,
)
from llm.frontend.wafer_frontend.passes.load_fabric import (
    hbm_address_spaces_from_data,
    physical_fabric_from_data,
)
from llm.frontend.wafer_frontend.policies.registry import production_registry
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir2 import BufferOwnership
from llm.frontend.wafer_frontend.schema.lite_moe_dp4_backward import (
    LiteMoeDp4Backward,
)
from llm.frontend.wafer_frontend.schema.lite_moe_dp4_train_forward import (
    LiteMoeDp4TrainForward,
)
from llm.frontend.wafer_frontend.schema.n4 import (
    FusionPartitionContext,
    InterDiePlanningContext,
)
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.policy import RegistryKind
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass
from llm.test.frontend.integration.lite_moe_cases import (
    build_lite_moe_execution_case,
    build_lite_moe_source_case,
)


_HARDWARE = (
    Path(__file__).resolve().parents[4]
    / "notes"
    / "frontend"
    / "examples"
    / "hardware_2x2.json"
)


class LiteMoeDp4Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = build_lite_moe_source_case()
        cls.spec = build_lite_moe_dp4_spec(source.moe_spec.trace)
        cls.topology = build_lite_moe_dp4_topology(cls.spec)
        cls.oracle = build_lite_moe_dp4_oracle(cls.spec, cls.topology)
        cls.adapter = build_lite_moe_dp4_ir0_adapter(
            source.spec, cls.spec, cls.topology, cls.oracle
        )
        hardware = json.loads(_HARDWARE.read_text(encoding="utf-8"))
        placement = PlacementContext.create(
            producer_pass="test_lite_moe_dp4",
            fabric=physical_fabric_from_data(hardware),
            placement=source.spec.placement,
            hbm_address_spaces=hbm_address_spaces_from_data(hardware),
        )
        registry = production_registry()
        partition = FusionPartitionContext.create(
            producer_pass="test_lite_moe_dp4"
        )
        planning = InterDiePlanningContext.create(
            producer_pass="test_lite_moe_dp4",
            fused_policy=registry.instantiate(
                RegistryKind.INTER_DIE, "naive"
            ).selection,
            standalone_policy=registry.instantiate(
                RegistryKind.STANDALONE_COLLECTIVE, "direct_all_gather"
            ).selection,
        )
        cls.forward = build_lite_moe_dp4_execution_case(
            source.spec, cls.adapter, placement, partition, planning
        )
        cls.train_forward = build_lite_moe_dp4_train_forward(cls.forward)
        cls.backward = build_lite_moe_dp4_backward(cls.train_forward)

    def test_public_exports_are_canonical_identities(self) -> None:
        self.assertIs(public_schema.LiteMoeDp4ExecutionCase, type(self.forward))
        self.assertIs(public_schema.LiteMoeDp4TrainForward, type(self.train_forward))
        self.assertIs(public_schema.LiteMoeDp4Backward, type(self.backward))
        self.assertIs(
            public_passes.build_lite_moe_dp4_execution_case,
            build_lite_moe_dp4_execution_case,
        )
        self.assertIs(
            public_passes.build_lite_moe_dp4_backward,
            build_lite_moe_dp4_backward,
        )
        self.assertIs(
            public_lowering.lower_lite_moe_dp4_infer,
            direct_dp4_lowering.lower_lite_moe_dp4_infer,
        )
        self.assertIs(
            public_lowering.link_lite_moe_dp4_backward_manifest,
            lite_moe_dp4_backward_linker.link_lite_moe_dp4_backward_manifest,
        )

    def test_infer_production_chain_exact(self) -> None:
        result = self.forward
        validate_lite_moe_dp4_execution_case(result)
        self.assertEqual(self.topology.die_grid, (2, 2))
        self.assertEqual(self.topology.remote_token_indices, (1, 2, 3, 4, 5, 6))
        self.assertEqual(
            (
                len(self.adapter.graph.nodes),
                len(self.adapter.graph.values),
                len(self.adapter.graph.edges),
                len(self.adapter.p2p_bindings),
            ),
            (44, 64, 42, 12),
        )
        self.assertEqual(
            tuple(len(item.tasks) for item in result.projection.dies),
            (20, 26, 26, 20),
        )
        self.assertEqual(
            (len(result.projection.flows), len(result.schedule.placements),
             len(result.schedule.buffers), len(result.global_dag.actions)),
            (12, 92, 68, 92),
        )
        self.assertEqual(sum(item.bytes for item in result.projection.flows), 384)
        routes = {
            route.id: route
            for route in result.n4.graph.groups[0].embedding.routes
        }
        self.assertTrue(
            any(len(routes[item.pair_route_ref].die_path) == 3
                for item in result.projection.flows)
        )
        self.assertEqual(len(result.global_dag.combined_output_refs), 8)

    def test_training_forward_tape_is_real_owned_terminal_fork(self) -> None:
        result = self.train_forward
        validate_lite_moe_dp4_train_forward(result, self.forward)
        self.assertEqual((len(result.tape_buffers), result.total_tape_bytes), (8, 512))
        self.assertTrue(
            all(
                item.size_bytes == 64
                and item.ownership is BufferOwnership.OWNED
                and item.terminal
                and item.alias_of is None
                for item in result.tape_buffers
            )
        )
        self.assertEqual(
            tuple(item.die_id for item in result.tape_buffers),
            (0, 0, 1, 1, 2, 2, 3, 3),
        )
        self.assertEqual(
            {item.destination_buffer_ref for item in result.tape_copies},
            {item.id for item in result.tape_buffers},
        )
        self.assertTrue(
            all(item.deps == (item.source_action_ref,) for item in result.tape_copies)
        )

    def test_backward_exact_remote_wgrad_reduce_sgd_state(self) -> None:
        result = self.backward
        validate_lite_moe_dp4_backward(result, self.train_forward)
        self.assertEqual(
            tuple(
                (item.token_index, item.source_die_id, item.destination_die_id)
                for item in result.remote_gradients
            ),
            ((1, 1, 0), (2, 2, 1), (3, 3, 1),
             (4, 0, 2), (5, 1, 2), (6, 2, 3)),
        )
        self.assertTrue(
            all(item.size_bytes == 2048 and item.dtype is DType.FP32
                for item in result.token_wgrads)
        )
        self.assertTrue(
            all(
                (item.input_count, item.element_count, item.input_stride_bytes,
                 item.source_span_bytes, item.destination_span_bytes)
                == (2, 512, 2048, 4096, 2048)
                and item.output_ownership is BufferOwnership.ALIASED
                for item in result.expert_reduces
            )
        )
        self.assertEqual(
            tuple(
                (item.home_die_id, item.declaration.identity.kind.value,
                 item.declaration.lifetime.value, item.declaration.access.value)
                for item in result.trainable_down_states
            ),
            tuple((index, "trainable_parameter", "persistent", "read_write")
                  for index in range(4)),
        )
        self.assertTrue(
            all(item.deps == (item.reduce_ref,) and item.state_store_bytes == 1024
                for item in result.sgd_stores)
        )

    def test_strict_serde_and_determinism(self) -> None:
        rebuilt_tf = build_lite_moe_dp4_train_forward(self.forward)
        rebuilt_tb = build_lite_moe_dp4_backward(rebuilt_tf)
        self.assertEqual(rebuilt_tf, self.train_forward)
        self.assertEqual(rebuilt_tb, self.backward)
        self.assertEqual(
            loads_dataclass(
                LiteMoeDp4TrainForward,
                canonical_json(self.train_forward),
                path="train_forward",
            ),
            self.train_forward,
        )
        self.assertEqual(
            loads_dataclass(
                LiteMoeDp4Backward,
                canonical_json(self.backward),
                path="backward",
            ),
            self.backward,
        )
        raw = json.loads(canonical_json(self.backward))
        raw["unexpected"] = True
        with self.assertRaises(SchemaError):
            loads_dataclass(LiteMoeDp4Backward, json.dumps(raw), path="backward")

    def test_tamper_fails_closed(self) -> None:
        with self.assertRaises(SchemaError):
            replace(self.topology, token_source_die_ids=(0,) * 8).validate()
        with self.assertRaises(SchemaError):
            replace(
                self.forward.projection,
                flows=self.forward.projection.flows[:-1],
            ).validate()
        with self.assertRaises(SchemaError):
            replace(
                self.train_forward.tape_buffers[0],
                ownership=BufferOwnership.ALIASED,
            ).validate("tape")
        with self.assertRaises(SchemaError):
            replace(
                self.backward.remote_gradients[0],
                source_die_id=self.backward.remote_gradients[0].destination_die_id,
            ).validate("remote")
        with self.assertRaises(SchemaError):
            replace(self.backward.token_wgrads[0], dtype=DType.FP16).validate("wgrad")
        with self.assertRaises(SchemaError):
            replace(self.backward.expert_reduces[0], input_count=1).validate("reduce")
        with self.assertRaises(SchemaError):
            replace(
                self.backward.sgd_stores[0],
                deps=(self.backward.token_wgrads[0].id,),
            ).validate("sgd")

    def test_old_two_die_case_has_no_drift(self) -> None:
        old = build_lite_moe_execution_case()
        old.validate()
        self.assertEqual(
            (
                len(old.adapter.graph.nodes),
                len(old.global_dag.actions),
                len(old.n6_intent.dte_units),
            ),
            (40, 80, 8),
        )


if __name__ == "__main__":
    unittest.main()
