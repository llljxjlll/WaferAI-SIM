from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.global_action import (
    ActionBufferUse,
    ActionStateUse,
    GLOBAL_ACTION_DAG_SCHEMA_VERSION,
    GLOBAL_ACTION_SCHEMA_VERSION,
    GlobalAction,
    GlobalActionDAG,
    LogicalCoreRef,
    ScheduledDagRef,
    ScheduledSourceRef,
)
from llm.frontend.wafer_frontend.schema.ir2 import (
    BufferAccess,
    BufferUseRole,
    DmaContract,
    OriginKind,
    RegionLowering,
    SemanticTaskKind,
    StateIoOrigin,
    StateUseAccess,
    TensorSlice,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    loads_dataclass,
    to_primitive,
)


def _state_action(kind: SemanticTaskKind) -> GlobalAction:
    if kind not in (SemanticTaskKind.DMA_IN, SemanticTaskKind.DMA_OUT):
        raise AssertionError("test fixture requires a state DMA kind")
    staging_id = "state-staging"
    tensor_slice = TensorSlice(staging_id, (0, 0), (4, 4))
    is_in = kind is SemanticTaskKind.DMA_IN
    return GlobalAction.create(
        source=ScheduledSourceRef("dag", "schedule", f"task-{kind.value}"),
        task_kind=kind,
        origin_ref=StateIoOrigin(
            OriginKind.STATE_IO,
            "state-access",
            "node",
            0,
        ),
        lowering=RegionLowering.STRICT_STATE_IO,
        region_id=f"region-{kind.value}",
        op_kind=None,
        member_id=None,
        flow_id=None,
        chunk_id=None,
        collective_step=None,
        source_rank=None,
        destination_rank=None,
        tensor_slice=tensor_slice,
        bytes=32,
        dtype=DType.FP16,
        shape=(4, 4),
        read_values=() if is_in else (staging_id,),
        write_values=(staging_id,) if is_in else (),
        compute=None,
        reduction=None,
        sync=None,
        dma=DmaContract(
            "state",
            staging_id,
            0,
            ("access-task",),
        ),
        logical_core=LogicalCoreRef(0, 0),
        core_order_index=0,
        flow=None,
        flow_route=None,
        runtime_binding=None,
        buffer_uses=(
            ActionBufferUse(
                "sram-binding",
                BufferAccess.WRITE if is_in else BufferAccess.READ,
                (
                    BufferUseRole.DMA_DESTINATION
                    if is_in
                    else BufferUseRole.DMA_SOURCE
                ),
                0,
                None,
                tensor_slice,
            ),
        ),
        state_uses=(
            ActionStateUse(
                "hbm-binding",
                StateUseAccess.READ if is_in else StateUseAccess.WRITE,
            ),
        ),
        deps=(),
    )


def _state_dag(action: GlobalAction) -> GlobalActionDAG:
    return GlobalActionDAG.create(
        producer_pass="global_action_state_fixture",
        source_ir1_id="ir1",
        source_state_manifest_id="manifest",
        source_projection_id="projection",
        source_schedule_set_id="schedule-set",
        scheduled_dags=(ScheduledDagRef("dag", "schedule", 0),),
        actions=(action,),
    )


class GlobalActionStateSchemaTest(unittest.TestCase):
    def test_versions_and_directional_state_dma_round_trip(self) -> None:
        self.assertEqual(
            GLOBAL_ACTION_SCHEMA_VERSION,
            "wafer_frontend.global_action/v1alpha8",
        )
        self.assertEqual(
            GLOBAL_ACTION_DAG_SCHEMA_VERSION,
            "wafer_frontend.global_action_dag/v1alpha11",
        )
        for kind in (SemanticTaskKind.DMA_IN, SemanticTaskKind.DMA_OUT):
            with self.subTest(kind=kind):
                action = _state_action(kind)
                action.validate("action")
                dag = _state_dag(action)
                dag.validate()
                decoded = loads_dataclass(
                    GlobalActionDAG,
                    canonical_json(dag),
                )
                self.assertEqual(decoded, dag)
                self.assertEqual(
                    canonical_digest(decoded),
                    canonical_digest(dag),
                )

    def test_dma_contract_and_hbm_state_use_are_required(self) -> None:
        action = _state_action(SemanticTaskKind.DMA_IN)
        with self.assertRaisesRegex(SchemaError, "DmaContract"):
            replace(action, dma=None).validate("action")
        with self.assertRaisesRegex(SchemaError, "direction-matching HBM"):
            replace(action, state_uses=()).validate("action")
        with self.assertRaisesRegex(SchemaError, "direction-matching HBM"):
            replace(
                action,
                state_uses=(
                    ActionStateUse(
                        "hbm-binding",
                        StateUseAccess.WRITE,
                    ),
                ),
            ).validate("action")

    def test_hbm_use_is_not_a_second_sram_buffer_use(self) -> None:
        action = _state_action(SemanticTaskKind.DMA_OUT)
        with self.assertRaisesRegex(SchemaError, "local SRAM use"):
            replace(
                action,
                buffer_uses=action.buffer_uses
                + (
                    replace(
                        action.buffer_uses[0],
                        binding_id="hbm-binding",
                    ),
                ),
            ).validate("action")

    def test_manifest_provenance_is_required_for_state_actions(self) -> None:
        dag = _state_dag(_state_action(SemanticTaskKind.DMA_IN))
        fields = dag._semantic_key()
        fields["source_state_manifest_id"] = None
        with self.assertRaisesRegex(SchemaError, "manifest provenance"):
            GlobalActionDAG.create(
                producer_pass=dag.producer_pass,
                **fields,
            ).validate()

    def test_strict_serde_rejects_missing_and_unknown_state_fields(self) -> None:
        dag = _state_dag(_state_action(SemanticTaskKind.DMA_IN))
        missing = to_primitive(dag)
        assert isinstance(missing, dict)
        missing.pop("source_state_manifest_id")
        with self.assertRaises(SchemaError):
            loads_dataclass(
                GlobalActionDAG,
                canonical_json(missing),
            )
        unknown = to_primitive(dag.actions[0])
        assert isinstance(unknown, dict)
        state_uses = unknown["state_uses"]
        assert isinstance(state_uses, list)
        assert isinstance(state_uses[0], dict)
        state_uses[0]["unexpected"] = True
        with self.assertRaises(SchemaError):
            loads_dataclass(
                GlobalAction,
                canonical_json(unknown),
            )


if __name__ == "__main__":
    unittest.main()
