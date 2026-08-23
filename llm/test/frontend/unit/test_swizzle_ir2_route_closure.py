from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.project_swizzle_ir2 import (
    project_swizzle_adapter,
    validate_swizzle_projection_against_adapter,
)
from llm.frontend.wafer_frontend.policies.swizzle.materialize import (
    materialize_swizzle_decision,
)
from llm.frontend.wafer_frontend.schema.swizzle_ir2 import (
    SwizzleIr2Flow,
    SwizzleIr2Projection,
)

import test_swizzle_materialize as materialize_fixture


class SwizzleIr2RouteClosureTest(unittest.TestCase):
    def test_restable_flow_cannot_drift_from_frozen_adapter_route(self) -> None:
        adapter = materialize_swizzle_decision(
            materialize_fixture._decision(fused=True)
        )
        projection = project_swizzle_adapter(adapter)
        flow = projection.flows[0]
        forged_flow = SwizzleIr2Flow.create(
            route_ref=flow.route_ref,
            source_rank=flow.source_rank,
            destination_rank=flow.destination_rank,
            source_die=flow.source_die,
            destination_die=flow.destination_die,
            die_path=(flow.source_die, 999, flow.destination_die),
            send_task_ref=flow.send_task_ref,
            recv_task_ref=flow.recv_task_ref,
            chunk_index=flow.chunk_index,
            logical_bytes=flow.logical_bytes,
        )
        semantic = projection._semantic_key()
        semantic["flows"] = (forged_flow,)
        with self.assertRaisesRegex(SchemaError, "route|endpoint tasks"):
            forged = SwizzleIr2Projection.create(**semantic)
            validate_swizzle_projection_against_adapter(forged, adapter)


if __name__ == "__main__":
    unittest.main()
