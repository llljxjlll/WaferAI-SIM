from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.external_dma_action_graph import (
    build_external_dma_action_graph,
    validate_external_dma_action_graph,
)
from llm.frontend.wafer_frontend.passes.external_dma_program import (
    finalize_external_dma_program,
)
from llm.frontend.wafer_frontend.passes.offload import (
    plan_offload_blocking,
    plan_resident_only_memory,
)
from llm.frontend.wafer_frontend.passes.workload_materialization import (
    materialize_workload_preflight,
)
from llm.frontend.wafer_frontend.schema.external_dma_program import (
    ExternalDmaBackendBinding,
    ExternalDmaProbe,
    ExternalDmaSeed,
)
from llm.frontend.wafer_frontend.schema.external_dma_action_graph import (
    ExternalDmaActionGraph,
    ExternalDmaBoundAction,
)
from llm.frontend.wafer_frontend.schema.external_memory import (
    ExternalMemoryConnection,
    ExternalMemoryFabric,
    ExternalMemoryLink,
)
from llm.frontend.wafer_frontend.schema.memory_plan import MemoryTier, MemoryTierCapacity
from llm.frontend.wafer_frontend.schema.offload import (
    OffloadChunk,
    OffloadEventKind,
    OffloadStateMapping,
    OffloadTraceEvent,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest
from llm.frontend.wafer_frontend.schema.workload_run import (
    WorkloadCapabilityLevel,
    WorkloadFamily,
    WorkloadMemoryMode,
    WorkloadMemoryPolicy,
    WorkloadMeshSpec,
    WorkloadParallelSpec,
    WorkloadRunCapability,
    WorkloadRunRequest,
)
from llm.frontend.wafer_frontend.workload_external_dma import (
    create_external_dma_workload_adapter,
)
from llm.frontend.wafer_frontend.workload_capability_evidence import (
    WorkloadCapabilityArtifact,
    WorkloadCapabilityArtifactKind,
    WorkloadCapabilityEvidence,
    WorkloadEvidenceScope,
    build_workload_capability_from_evidence,
)
from llm.frontend.wafer_frontend.workload_runner import run_workload
from llm.test.frontend.unit.test_workload_materialization import (
    _capability,
    _request,
)


def _fixture(runtime_binary: Path | None = None):
    base = _request(WorkloadFamily.DENSE_INFERENCE)
    request = WorkloadRunRequest.create(
        family=base.family,
        model=base.model,
        steps=base.steps,
        mesh=WorkloadMeshSpec(1, 1),
        parallel=WorkloadParallelSpec(),
        memory=WorkloadMemoryPolicy(
            mode=WorkloadMemoryMode.EXTERNAL_OFFLOAD,
            external_tier_ref="host:0",
        ),
        optimizer=None,
        execution=base.execution,
    )
    yes = WorkloadCapabilityLevel.SUPPORTED
    base_capability = _capability(supported=True)
    families = tuple(
        replace(item, external_offload=yes)
        for item in base_capability.families
    )
    capability = WorkloadRunCapability.create(
        max_mesh_rows=10,
        max_mesh_columns=10,
        max_mesh_ranks=100,
        families=families,
    )
    derivation = None
    if runtime_binary is not None:
        derivation = _runtime_derivation(request, runtime_binary)
        capability = derivation.capability
    external = MemoryTierCapacity.create(
        tier=MemoryTier.EXTERNAL,
        location_ref="host:0",
        base_address=0,
        capacity_bytes=1 << 20,
        alignment_bytes=16,
    )
    manifest_hbm = MemoryTierCapacity.create(
        tier=MemoryTier.HBM,
        location_ref="die:0",
        base_address=0,
        capacity_bytes=1 << 20,
        alignment_bytes=16,
    )
    capacities = (external, manifest_hbm)
    manifest = materialize_workload_preflight(
        request, capability, capacities=capacities
    )
    requests = {item.id: item for item in manifest.memory_plan.requests}
    versions = {item.id: item for item in manifest.memory_plan.state_versions}
    parameter_allocations = tuple(
        item
        for item in manifest.memory_plan.allocations
        if requests[item.request_ref].tier is MemoryTier.EXTERNAL
    )
    assert len(parameter_allocations) == 1
    source_allocation = parameter_allocations[0]
    source_request = requests[source_allocation.request_ref]
    version = versions[source_request.state_version_ref]
    window = MemoryTierCapacity.create(
        tier=MemoryTier.HBM,
        location_ref="die:0",
        base_address=0,
        capacity_bytes=source_request.size_bytes,
        alignment_bytes=16,
    )
    link = ExternalMemoryLink.create(
        external_capacity_ref=external.id,
        ingress_die_id=0,
        bytes_per_cycle=256,
        latency_cycles=2,
        queue_depth=2,
        max_outstanding=2,
    )
    connection = ExternalMemoryConnection.create(
        link_ref=link.id,
        hbm_capacity_ref=window.id,
        target_die_id=0,
        route_die_ids=(0,),
        route_latency_cycles=0,
        route_bytes_per_cycle=None,
    )
    fabric = ExternalMemoryFabric.create(
        external_capacities=(external,),
        hbm_capacities=(window,),
        links=(link,),
        connections=(connection,),
    )
    chunk = OffloadChunk.create(
        initial_version=version,
        object_kind=source_request.object_kind,
        size_bytes=source_request.size_bytes,
        alignment_bytes=source_request.alignment_bytes,
        external_capacity_ref=external.id,
        external_address=source_allocation.address,
        connection_ref=connection.id,
    )
    mapping = OffloadStateMapping.create(
        chunk_ref=chunk.id,
        source_state_version_ref=version.id,
        source_allocation_ref=source_allocation.id,
    )
    plan = plan_offload_blocking(
        request_digest=request.digest,
        logical_graph_digest=manifest.logical_graph_digest,
        source_memory_plan_digest=canonical_digest(manifest.memory_plan),
        source_memory_plan=manifest.memory_plan,
        state_mappings=(mapping,),
        fabric=fabric,
        chunks=(chunk,),
        events=(OffloadTraceEvent.create(
            ordinal=0,
            kind=OffloadEventKind.READ,
            chunk_ref=chunk.id,
        ),),
    )
    payload = bytes((index % 251) + 1 for index in range(chunk.size_bytes))
    program = finalize_external_dma_program(
        plan=plan,
        case_digest=canonical_digest(request.case_id),
        backend_bindings=(ExternalDmaBackendBinding.create(
            hbm_capacity_ref=window.id,
            owner_die_id=0,
            stack_id=0,
            channel_id=0,
        ),),
        external_seeds=(ExternalDmaSeed.create(
            external_capacity_ref=external.id,
            address=source_allocation.address,
            payload=payload,
        ),),
        external_probes=(ExternalDmaProbe.create(
            external_capacity_ref=external.id,
            address=source_allocation.address,
            expected_payload=payload,
        ),),
    )
    return (
        request,
        capability,
        capacities,
        manifest,
        plan,
        program,
        chunk,
        derivation,
    )


def _runtime_derivation(request, runtime_binary: Path):
    binary_digest = hashlib.sha256(runtime_binary.read_bytes()).hexdigest()
    evidence = WorkloadCapabilityEvidence.create(
        request=request,
        scope=WorkloadEvidenceScope.FULL_MODEL,
        source_digest="1" * 64,
        binary_digest=binary_digest,
        toolchain_digest="2" * 64,
        artifacts=tuple(
            WorkloadCapabilityArtifact.create(
                kind=kind,
                artifact_digest=f"{index + 3:x}" * 64,
            )
            for index, kind in enumerate(WorkloadCapabilityArtifactKind)
        ),
        independent_execution_digests=("9" * 64, "9" * 64),
    )
    return build_workload_capability_from_evidence(
        (evidence,),
        max_mesh_rows=10,
        max_mesh_columns=10,
        max_mesh_ranks=100,
    )


class ExternalDmaActionGraphTest(unittest.TestCase):
    def test_action_graph_covers_logical_dma_and_residency_sources(self) -> None:
        _, _, _, manifest, plan, program, _, _ = _fixture()
        graph = build_external_dma_action_graph(
            manifest=manifest, plan=plan, program=program
        )
        self.assertEqual(graph.logical_graph_digest, manifest.logical_graph_digest)
        self.assertTrue(any(item.residency_refs for item in graph.actions))
        self.assertEqual(
            len(graph.actions),
            len(program.descriptors) + len(manifest.logical_graph.operations),
        )
        validate_external_dma_action_graph(
            graph, manifest=manifest, plan=plan, program=program
        )
        forged = ExternalDmaActionGraph.create(
            manifest_digest=graph.manifest_digest,
            case_digest=graph.case_digest,
            request_digest="a" * 64,
            logical_graph_digest=graph.logical_graph_digest,
            source_memory_plan_digest=graph.source_memory_plan_digest,
            blocking_offload_plan_digest=graph.blocking_offload_plan_digest,
            external_dma_program_digest=graph.external_dma_program_digest,
            actions=graph.actions,
        )
        with self.assertRaisesRegex(SchemaError, "source binding mismatch"):
            validate_external_dma_action_graph(
                forged,
                manifest=manifest,
                plan=plan,
                program=program,
            )
        last = graph.actions[-1]
        forged_last = ExternalDmaBoundAction.create(
            sequence=last.sequence,
            kind=last.kind,
            source_ref=last.source_ref,
            depends_on=last.depends_on,
            state_version_refs=(),
            residency_refs=last.residency_refs,
        )
        rebound = ExternalDmaActionGraph.create(
            manifest_digest=graph.manifest_digest,
            case_digest=graph.case_digest,
            request_digest=graph.request_digest,
            logical_graph_digest=graph.logical_graph_digest,
            source_memory_plan_digest=graph.source_memory_plan_digest,
            blocking_offload_plan_digest=graph.blocking_offload_plan_digest,
            external_dma_program_digest=graph.external_dma_program_digest,
            actions=graph.actions[:-1] + (forged_last,),
        )
        with self.assertRaisesRegex(SchemaError, "binding mismatch"):
            validate_external_dma_action_graph(
                rebound,
                manifest=manifest,
                plan=plan,
                program=program,
            )

    def test_same_workload_resident_oom_and_offload_component_succeeds(self) -> None:
        runtime = os.environ.get("NPUSIM_EXTERNAL_DMA_RUNNER")
        runtime_path = Path(runtime) if runtime is not None else None
        (
            request,
            capability,
            capacities,
            manifest,
            plan,
            program,
            chunk,
            derivation,
        ) = _fixture(runtime_path)
        tiny_hbm = MemoryTierCapacity.create(
            tier=MemoryTier.HBM,
            location_ref="die:0",
            base_address=0,
            capacity_bytes=chunk.size_bytes - 16,
            alignment_bytes=16,
        )
        tiny_link = ExternalMemoryLink.create(
            external_capacity_ref=program.fabric.external_capacities[0].id,
            ingress_die_id=0,
            bytes_per_cycle=256,
            latency_cycles=2,
            queue_depth=2,
            max_outstanding=2,
        )
        tiny_connection = ExternalMemoryConnection.create(
            link_ref=tiny_link.id,
            hbm_capacity_ref=tiny_hbm.id,
            target_die_id=0,
            route_die_ids=(0,),
            route_latency_cycles=0,
            route_bytes_per_cycle=None,
        )
        tiny_fabric = ExternalMemoryFabric.create(
            external_capacities=program.fabric.external_capacities,
            hbm_capacities=(tiny_hbm,),
            links=(tiny_link,),
            connections=(tiny_connection,),
        )
        tiny_chunk = OffloadChunk.create(
            initial_version=chunk.initial_version,
            object_kind=chunk.object_kind,
            size_bytes=chunk.size_bytes,
            alignment_bytes=chunk.alignment_bytes,
            external_capacity_ref=chunk.external_capacity_ref,
            external_address=chunk.external_address,
            connection_ref=tiny_connection.id,
        )
        with self.assertRaisesRegex(SchemaError, "capacity"):
            plan_resident_only_memory(fabric=tiny_fabric, chunks=(tiny_chunk,))

        if runtime is None:
            self.skipTest("C++ external DMA runner not supplied")
        adapter = create_external_dma_workload_adapter(
            manifest=manifest,
            plan=plan,
            program=program,
            runtime_binary=Path(runtime),
            source_digest="1" * 64,
            toolchain_digest="2" * 64,
        )
        assert derivation is not None
        with tempfile.TemporaryDirectory() as root:
            result = run_workload(
                request,
                derivation.capability,
                capacities=capacities,
                output_dir=Path(root) / "dense_1x1",
                adapter=adapter,
                capability_derivation=derivation,
            )
            report = json.loads((
                result.output_dir
                / "execution_0/artifacts/external_dma_runtime_report.json"
            ).read_text(encoding="utf-8"))
        self.assertEqual(result.manifest.logical_graph_digest, manifest.logical_graph_digest)
        self.assertGreater(report["external_read_bytes"], 0)
        self.assertGreater(report["hbm_write_bytes"], 0)
        self.assertEqual(report["pending_requests"], 0)


if __name__ == "__main__":
    unittest.main()
