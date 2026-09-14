from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import (
    build_ir0,
    logical_expand,
    place_ir0,
)
from llm.frontend.wafer_frontend.passes.memory_plan import (
    plan_hierarchical_memory,
    plan_persistent_hbm,
)
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec
from llm.frontend.wafer_frontend.schema.memory_plan import (
    MemoryAllocationRequest,
    MemoryObjectKind,
    MemoryPeak,
    MemoryPlan,
    MemoryPlanExecution,
    MemoryResidency,
    MemoryStateVersion,
    MemoryTier,
    MemoryTierCapacity,
    ResidencyStatus,
)
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    from_data,
    to_primitive,
)

from _fixtures import valid_hbm_address_spaces, valid_ir1, valid_spec


def capacity(
    tier: MemoryTier = MemoryTier.SRAM,
    *,
    location: str = "die:0/core:0",
    size: int = 256,
) -> MemoryTierCapacity:
    return MemoryTierCapacity.create(
        tier=tier,
        location_ref=location,
        base_address=0,
        capacity_bytes=size,
        alignment_bytes=16,
    )


def version(name: str, *, writable: bool = True) -> MemoryStateVersion:
    return MemoryStateVersion.create(
        state_ref=name,
        generation=0,
        predecessor_ref=None,
        writable=writable,
    )


def request(
    state: MemoryStateVersion,
    *,
    start: int,
    end: int,
    size: int = 64,
    tier: MemoryTier = MemoryTier.SRAM,
    location: str = "die:0/core:0",
    pinned_address: int | None = None,
) -> MemoryAllocationRequest:
    return MemoryAllocationRequest.create(
        state_version_ref=state.id,
        object_kind=MemoryObjectKind.ACTIVATION,
        tier=tier,
        location_ref=location,
        size_bytes=size,
        alignment_bytes=16,
        lifetime_start=start,
        lifetime_end_exclusive=end,
        pinned_address=pinned_address,
    )


class MemoryPlanTest(unittest.TestCase):
    def test_lifetime_reuse_peak_and_stable_round_trip(self) -> None:
        first, second, overlap = version("first"), version("second"), version("overlap")
        requests = (
            request(first, start=0, end=2, size=48),
            request(second, start=2, end=4, size=48),
            request(overlap, start=1, end=3, size=32),
        )
        plan = plan_hierarchical_memory(
            capacities=(capacity(),),
            state_versions=(first, second, overlap),
            requests=requests,
            dirty_state_version_refs=(overlap.id,),
        )
        allocation_by_request = {item.request_ref: item for item in plan.allocations}
        self.assertEqual(
            allocation_by_request[requests[0].id].address,
            allocation_by_request[requests[1].id].address,
        )
        self.assertNotEqual(
            allocation_by_request[requests[0].id].address,
            allocation_by_request[requests[2].id].address,
        )
        self.assertEqual(plan.peaks[0].peak_bytes, 80)
        self.assertEqual(plan.peaks[0].at_tick, 1)
        self.assertEqual(
            plan.execution,
            MemoryPlanExecution.NO_EXTERNAL_TRANSPORT_REQUIRED,
        )

        repeated = plan_hierarchical_memory(
            capacities=(capacity(),),
            state_versions=(overlap, second, first),
            requests=tuple(reversed(requests)),
            dirty_state_version_refs=(overlap.id,),
        )
        self.assertEqual(plan, repeated)
        decoded = from_data(type(plan), to_primitive(plan), path="plan")
        decoded.validate()
        self.assertEqual(canonical_digest(decoded), canonical_digest(plan))

    def test_overlapping_lifetimes_require_real_capacity(self) -> None:
        first, second = version("first"), version("second")
        with self.assertRaisesRegex(SchemaError, "capacity exceeded") as caught:
            plan_hierarchical_memory(
                capacities=(capacity(size=64),),
                state_versions=(first, second),
                requests=(
                    request(first, start=0, end=3),
                    request(second, start=1, end=2),
                ),
            )
        self.assertEqual(caught.exception.code, "memory_capacity_exceeded")

    def test_exact_boundary_alignment_and_pinned_overlap(self) -> None:
        first, second = version("first"), version("second")
        exact = plan_hierarchical_memory(
            capacities=(capacity(size=96),),
            state_versions=(first, second),
            requests=(
                request(first, start=0, end=2, size=17),
                request(second, start=0, end=2, size=64),
            ),
        )
        self.assertEqual(exact.peaks[0].peak_bytes, 96)
        with self.assertRaisesRegex(SchemaError, "capacity exceeded"):
            plan_hierarchical_memory(
                capacities=(capacity(size=96),),
                state_versions=(first, second),
                requests=(
                    request(first, start=0, end=2, size=64, pinned_address=0),
                    request(second, start=0, end=2, size=64, pinned_address=32),
                ),
            )

    def test_state_generation_lineage_and_dirty_read_only_fail_closed(self) -> None:
        base = version("parameter")
        next_version = MemoryStateVersion.create(
            state_ref="parameter",
            generation=1,
            predecessor_ref=base.id,
            writable=True,
        )
        plan_hierarchical_memory(
            capacities=(capacity(),),
            state_versions=(base, next_version),
            requests=(request(next_version, start=0, end=1),),
        )

        unrelated = version("other")
        bad_lineage = MemoryStateVersion.create(
            state_ref="parameter",
            generation=1,
            predecessor_ref=unrelated.id,
            writable=True,
        )
        with self.assertRaisesRegex(SchemaError, "prior generation"):
            plan_hierarchical_memory(
                capacities=(capacity(),),
                state_versions=(base, unrelated, bad_lineage),
                requests=(request(bad_lineage, start=0, end=1),),
            )

        read_only = version("weight", writable=False)
        with self.assertRaisesRegex(SchemaError, "read-only state cannot be dirty"):
            plan_hierarchical_memory(
                capacities=(capacity(),),
                state_versions=(read_only,),
                requests=(request(read_only, start=0, end=1),),
                dirty_state_version_refs=(read_only.id,),
            )

    def test_external_tier_is_explicitly_schema_only(self) -> None:
        state = version("offloaded")
        plan = plan_hierarchical_memory(
            capacities=(
                capacity(MemoryTier.EXTERNAL, location="host:0", size=1024),
            ),
            state_versions=(state,),
            requests=(
                request(
                    state,
                    start=0,
                    end=2,
                    tier=MemoryTier.EXTERNAL,
                    location="host:0",
                ),
            ),
        )
        self.assertEqual(plan.execution, MemoryPlanExecution.SCHEMA_ONLY_EXTERNAL)

    def test_existing_persistent_manifest_lifts_without_address_change(self) -> None:
        spec = from_data(ExperimentSpec, valid_spec(), path="spec")
        graph = logical_expand(build_ir0(spec)).entries[0].graph
        fabric = valid_ir1().fabric
        context = PlacementContext.create(
            producer_pass="test_memory_plan",
            fabric=fabric,
            placement=spec.placement,
            hbm_address_spaces=valid_hbm_address_spaces(fabric),
        )
        placed = place_ir0(graph, context)
        manifest = placed.persistent_state_manifest
        assert manifest is not None
        plan = plan_persistent_hbm(manifest, lifetime_end_exclusive=10)
        request_by_version = {item.state_version_ref: item for item in plan.requests}
        version_by_state = {item.state_ref: item for item in plan.state_versions}
        for binding in manifest.bindings:
            request_item = request_by_version[version_by_state[binding.state_ref].id]
            self.assertEqual(request_item.pinned_address, binding.address)
            allocation = next(
                item for item in plan.allocations if item.request_ref == request_item.id
            )
            self.assertEqual(allocation.address, binding.address)
        self.assertEqual(
            plan.execution,
            MemoryPlanExecution.NO_EXTERNAL_TRANSPORT_REQUIRED,
        )

    def test_plan_rejects_forged_peak_and_out_of_lifetime_residency(self) -> None:
        state = version("activation")
        plan = plan_hierarchical_memory(
            capacities=(capacity(),),
            state_versions=(state,),
            requests=(request(state, start=2, end=4),),
        )
        forged_peak = MemoryPeak.create(
            capacity_ref=plan.peaks[0].capacity_ref,
            at_tick=0,
            peak_bytes=0,
            active_allocation_refs=(),
        )
        with self.assertRaisesRegex(SchemaError, "recomputed lifetime peak"):
            MemoryPlan.create(
                capacities=plan.capacities,
                state_versions=plan.state_versions,
                requests=plan.requests,
                allocations=plan.allocations,
                residencies=plan.residencies,
                peaks=(forged_peak,),
            )
        forged_residency = MemoryResidency.create(
            state_version_ref=plan.residencies[0].state_version_ref,
            allocation_ref=plan.residencies[0].allocation_ref,
            status=ResidencyStatus.CLEAN,
            valid_from=1,
            valid_until_exclusive=4,
        )
        with self.assertRaisesRegex(SchemaError, "residency exceeds"):
            MemoryPlan.create(
                capacities=plan.capacities,
                state_versions=plan.state_versions,
                requests=plan.requests,
                allocations=plan.allocations,
                residencies=(forged_residency,),
                peaks=plan.peaks,
            )


if __name__ == "__main__":
    unittest.main()
