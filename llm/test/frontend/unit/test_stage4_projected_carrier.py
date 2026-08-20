from __future__ import annotations

from dataclasses import replace
import inspect
import json
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import (
    project_stage4 as public_project_stage4,
)
from llm.frontend.wafer_frontend.passes.project_to_ir2 import project_stage4
from llm.frontend.wafer_frontend.schema import (
    Stage4ProjectedIR2 as PublicStage4ProjectedIR2,
    Stage4ProjectToIR2Context as PublicStage4ProjectToIR2Context,
)
from llm.frontend.wafer_frontend.schema.n5 import (
    STAGE4_PROJECTED_IR2_SCHEMA_VERSION,
    STAGE4_PROJECT_TO_IR2_CONTEXT_SCHEMA_VERSION,
    ProjectToIR2Contract,
    Stage4ProjectedIR2,
    Stage4ProjectToIR2Context,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass

from llm.frontend.wafer_frontend.schema.state_transfer import (
    SegmentedKvStateTransferContract,
)
from test_stage4_carriers import _chain, _fused_chain


def _context() -> Stage4ProjectToIR2Context:
    return Stage4ProjectToIR2Context.create(
        producer_pass="stage4_projected_carrier_test"
    )


class Stage4ProjectedCarrierTest(unittest.TestCase):
    def test_public_api_and_context_have_no_caller_transfer_argument(self) -> None:
        self.assertIs(public_project_stage4, project_stage4)
        self.assertIs(PublicStage4ProjectedIR2, Stage4ProjectedIR2)
        self.assertIs(
            PublicStage4ProjectToIR2Context,
            Stage4ProjectToIR2Context,
        )
        self.assertNotIn(
            "state_transfers", inspect.signature(project_stage4).parameters
        )
        self.assertNotIn(
            "state_transfers",
            inspect.signature(Stage4ProjectToIR2Context.create).parameters,
        )
        with self.assertRaises(TypeError):
            Stage4ProjectToIR2Context.create(
                producer_pass="unit",
                state_transfers=(),  # type: ignore[call-arg]
            )

    def test_fused_tp1_projects_zero_transfers_with_exact_provenance(self) -> None:
        source = _fused_chain()[-1]
        context = _context()
        result = project_stage4(source, context)
        result.validate_against(source, context)
        self.assertEqual(
            STAGE4_PROJECT_TO_IR2_CONTEXT_SCHEMA_VERSION,
            "wafer_frontend.stage4_project_to_ir2_context/v1alpha1",
        )
        self.assertEqual(
            STAGE4_PROJECTED_IR2_SCHEMA_VERSION,
            "wafer_frontend.stage4_projected_ir2/v1alpha2",
        )
        self.assertEqual(result.projection.state_transfers, ())
        self.assertEqual(result.graph.cross_routes, ())
        self.assertEqual(result.pd_plan.handoffs, ())
        self.assertEqual(result.source_planned_carrier_id, source.id)
        self.assertEqual(
            result.source_partitioned_carrier_id,
            source.source_partitioned_carrier_id,
        )
        self.assertEqual(result.graph, source.graph)
        self.assertEqual(result.fusion_plans, source.fusion_plans)
        self.assertEqual(result.standalone_plans, source.standalone_plans)
        self.assertEqual(
            loads_dataclass(
                Stage4ProjectToIR2Context,
                canonical_json(context),
                path="context",
            ),
            context,
        )
        self.assertEqual(
            loads_dataclass(
                Stage4ProjectedIR2,
                canonical_json(result),
                path="result",
            ),
            result,
        )
        self.assertEqual(project_stage4(source, context), result)
        for artifact_type, artifact, field_name in (
            (Stage4ProjectToIR2Context, context, "contract"),
            (Stage4ProjectedIR2, result, "projection"),
        ):
            with self.subTest(artifact_type=artifact_type.__name__):
                raw = json.loads(canonical_json(artifact))
                raw["unexpected"] = True
                with self.assertRaises(SchemaError):
                    loads_dataclass(
                        artifact_type,
                        canonical_json(raw),
                        path="artifact",
                    )
                del raw["unexpected"]
                del raw[field_name]
                with self.assertRaises(SchemaError):
                    loads_dataclass(
                        artifact_type,
                        canonical_json(raw),
                        path="artifact",
                    )

    def test_pds_tp1_derives_four_transfers_without_caller_payload(self) -> None:
        source = _chain(1, 1)[-1]
        context = _context()
        result = project_stage4(source, context)
        result.validate_against(source, context)
        self.assertEqual(len(result.projection.state_transfers), 4)
        self.assertEqual(
            [contract.bytes for contract in result.projection.state_transfers],
            [256, 256, 256, 256],
        )
        self.assertEqual(
            sum(
                contract.bytes
                for contract in result.projection.state_transfers
            ),
            1024,
        )
        self.assertEqual(len(result.graph.cross_routes), 1)
        self.assertEqual(result.projection_context_id, context.id)

    def test_pdr_tp2_to_tp1_derives_segmented_transfers_without_caller_payload(self) -> None:
        source = _chain(2, 1)[-1]
        context = _context()
        result = project_stage4(source, context)
        result.validate_against(source, context)
        self.assertEqual(
            loads_dataclass(
                Stage4ProjectedIR2, canonical_json(result)
            ),
            result,
        )
        self.assertEqual(len(result.projection.state_transfers), 8)
        self.assertTrue(
            all(
                type(contract) is SegmentedKvStateTransferContract
                for contract in result.projection.state_transfers
            )
        )
        self.assertEqual(
            {
                len(contract.segments)
                for contract in result.projection.state_transfers
            },
            {8},
        )
        self.assertEqual(
            sum(
                contract.bytes
                for contract in result.projection.state_transfers
            ),
            1024,
        )
        self.assertEqual(
            [
                (dag.die_id, len(dag.tasks), len(dag.flows))
                for dag in result.projection.dags
            ],
            [(0, 112, 48), (1, 144, 80), (2, 172, 64)],
        )

    def test_version_source_context_projection_and_plan_tamper_fail_closed(self) -> None:
        source = _chain(1, 1)[-1]
        context = _context()
        result = project_stage4(source, context)
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(
                context,
                schema_version=(
                    "wafer_frontend.stage4_project_to_ir2_context/v0"
                ),
            ).validate()
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(
                result,
                schema_version="wafer_frontend.stage4_projected_ir2/v1alpha1",
            ).validate()
        fused_source = _fused_chain()[-1]
        with self.assertRaisesRegex(SchemaError, "source planned carrier"):
            result.validate_against(fused_source, context)
        with self.assertRaisesRegex(
            SchemaError, "unsupported Stage 4 projection contract"
        ):
            replace(
                context,
                contract=(
                    ProjectToIR2Contract.EXACT_NAIVE_PROJECTION_DENSE_FORWARD_V5
                ),
            ).validate()
        with self.assertRaises(SchemaError):
            replace(
                result,
                projection=replace(
                    result.projection,
                    state_transfers=(
                        result.projection.state_transfers[:-1]
                    ),
                ),
            ).validate_against(source, context)
        with self.assertRaisesRegex(SchemaError, "produced by project_to_ir2"):
            replace(
                result,
                projection=replace(
                    result.projection,
                    producer_pass="wrong_projector",
                ),
            ).validate()
        tp2_plan = _chain(2, 1)[-1].fusion_plans[0]
        with self.assertRaisesRegex(SchemaError, "fusion plan ids"):
            replace(result, fusion_plans=(tp2_plan,)).validate()


if __name__ == "__main__":
    unittest.main()
