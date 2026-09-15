"""Require offload source capacity to account for every physical training StateABI.

The generic materializer declares rank-local parameter allocations, while the
legacy training lowerer may emit a different set of physical parameter states.
An offload planner cannot use the former as the latter's external authority.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..schema.artifact_manifest import LinkedProgramManifest, RegionManifest
from ..schema.memory_plan import MemoryObjectKind, MemoryTier
from ..schema.serde import canonical_digest
from ..schema.workload_materialization import WorkloadMaterializationManifest
from ..schema.workload_run import WorkloadFamily, WorkloadOptimizerKind


@dataclass(frozen=True, slots=True)
class PhysicalTrainingDieRequirement:
    die_id: int
    source_parameter_bytes: int
    linked_state_bytes: int
    linked_span_bytes: int
    linked_padding_bytes: int
    state_abi_count: int
    state_abi_digest: str


def inspect_physical_training_state_abis(
    source: WorkloadMaterializationManifest,
    linked: LinkedProgramManifest,
) -> tuple[PhysicalTrainingDieRequirement, ...]:
    """Inspect exact per-Die physical states without claiming P3 coverage.

    This diagnostic is useful even when the source cannot safely be lowered.
    The strict ``require_offload_state_abi_source`` gate must be used before
    constructing any signed external DMA program from that P3 memory plan.
    """

    source.validate("source")
    linked.validate("linked")
    if (source.request.family is not WorkloadFamily.DENSE_TRAINING or
            source.request.optimizer is None or
            source.request.optimizer.kind is not WorkloadOptimizerKind.SGD):
        raise SchemaError("this source reconciliation requires Dense SGD",
                          path="source.request", code="external_state_abi_source_mismatch")
    active = source.request.parallel.active_die_ids
    if not active:
        active = tuple(range(source.request.mesh.rows * source.request.mesh.columns))
    declarations: dict[int, int] = {die: 0 for die in active}
    requests = {item.id: item for item in source.memory_plan.requests}
    for allocation in source.memory_plan.allocations:
        request = requests[allocation.request_ref]
        if request.object_kind is not MemoryObjectKind.PARAMETER:
            continue
        if request.tier is not MemoryTier.HBM or not request.location_ref.startswith("die:"):
            raise SchemaError("physical offload source needs per-Die resident parameter declarations",
                              path="source.memory_plan.requests",
                              code="external_state_abi_source_mismatch")
        die = int(request.location_ref.removeprefix("die:"))
        if die not in declarations:
            raise SchemaError("declared parameter owner is not an active Die",
                              path="source.memory_plan.requests",
                              code="external_state_abi_source_mismatch")
        declarations[die] += request.size_bytes

    capacity_home = {
        int(item.location_ref.removeprefix("die:")): item
        for item in source.memory_plan.capacities
        if item.tier is MemoryTier.HBM and item.location_ref.startswith("die:")
    }
    physical: dict[int, list[object]] = {die: [] for die in active}
    for fragment in linked.fragments:
        for abi in (
            fragment.fragment.state_abi if isinstance(fragment, RegionManifest)
            else fragment.state_abi
        ):
            if abi.die_id not in physical:
                raise SchemaError("linked StateABI is outside the active Die placement",
                                  path="linked.fragments.state_abi",
                                  code="external_state_abi_source_mismatch")
            physical[abi.die_id].append(abi)

    if not declarations or any(not physical[die] or die not in capacity_home
                               for die in declarations):
        raise SchemaError("physical StateABI or HBM home is missing for an active Die",
                          path="linked.fragments.state_abi",
                          code="external_state_abi_source_mismatch")
    output = []
    for die in sorted(declarations):
        ordered = tuple(sorted(physical[die], key=lambda item: (item.address, item.id)))
        home = capacity_home[die]
        base = home.base_address
        cursor = base
        for abi in ordered:
            if abi.address < cursor:
                raise SchemaError("physical training StateABIs overlap or precede their HBM home",
                                  path=f"linked.fragments.state_abi[die:{die}]",
                                  code="external_state_abi_source_mismatch")
            cursor = abi.address + abi.size_bytes
        if cursor > base + home.capacity_bytes:
            raise SchemaError("physical StateABI exceeds declared Die HBM home",
                              path=f"source.memory_plan.capacities[die:{die}]",
                              code="external_state_abi_hbm_capacity_exceeded")
        logical = sum(item.size_bytes for item in ordered)
        span = cursor - base
        output.append(PhysicalTrainingDieRequirement(
            die_id=die,
            source_parameter_bytes=declarations[die],
            linked_state_bytes=logical,
            linked_span_bytes=span,
            linked_padding_bytes=span - logical,
            state_abi_count=len(ordered),
            state_abi_digest=canonical_digest(tuple(
                (item.id, item.state_ref, item.address, item.size_bytes)
                for item in ordered
            )),
        ))
    return tuple(output)


def require_offload_state_abi_source(
    source: WorkloadMaterializationManifest,
    linked: LinkedProgramManifest,
) -> tuple[PhysicalTrainingDieRequirement, ...]:
    """Reject offload if the source plan omits physical StateABI bytes.

    The source also needs per-state allocations rather than an aggregate with
    a different state version before it can represent individual DMA chunks;
    this gate is only the initial capacity check.
    """

    witness = inspect_physical_training_state_abis(source, linked)
    mismatched = [item for item in witness
                  if item.source_parameter_bytes != item.linked_state_bytes]
    if mismatched:
        detail = ", ".join(
            f"die:{item.die_id} declared={item.source_parameter_bytes} "
            f"linked={item.linked_state_bytes} span={item.linked_span_bytes} "
            f"padding={item.linked_padding_bytes} abis={item.state_abi_count}"
            for item in mismatched
        )
        raise SchemaError(
            "P3 source parameter bytes differ from linked physical StateABI "
            f"and cannot authorize offload: {detail}",
            path="source.memory_plan.allocations",
            code="external_state_abi_source_mismatch",
        )
    return witness


__all__ = [
    "PhysicalTrainingDieRequirement",
    "inspect_physical_training_state_abis",
    "require_offload_state_abi_source",
]
