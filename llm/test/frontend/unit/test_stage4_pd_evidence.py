from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import json
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.link_program import link_stage4
from llm.frontend.wafer_frontend.passes.lower_program import lower_stage4
from llm.frontend.wafer_frontend.policies.registry import production_registry
from llm.frontend.wafer_frontend.schema import (
    Stage4PdRuntimeReport as PublicStage4PdRuntimeReport,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    LinkedProgramManifest,
    RegionManifest,
)
from llm.frontend.wafer_frontend.schema.capability import CapabilityStatus
from llm.frontend.wafer_frontend.schema.n4 import InterDiePlanningContext
from llm.frontend.wafer_frontend.schema.n5 import IntraDieSchedulingContext
from llm.frontend.wafer_frontend.schema.policy import RegistryKind
from llm.frontend.wafer_frontend.schema.program_io import ProgramIoMode
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    loads_dataclass,
)
from llm.frontend.wafer_frontend.schema.stage4_pd_evidence import (
    STAGE4_PD_BASELINE_EPOCH,
    STAGE4_PD_RUNTIME_REPORT_SCHEMA_VERSION,
    Stage4PdArtifactEvidence,
    Stage4PdControlEvidence,
    Stage4PdCoreCount,
    Stage4PdD2DEvidence,
    Stage4PdD2DLinkEvidence,
    Stage4PdEndpointRouteEvidence,
    Stage4PdMemoryEvidence,
    Stage4PdNamedDigest,
    Stage4PdNamedCount,
    Stage4PdProgramIoEvidence,
    Stage4PdRepeatEvidence,
    Stage4PdRuntimeReport,
)
from llm.frontend.wafer_frontend.schema.s1_naive_evidence import (
    S1_N_BASELINE_EPOCH,
    S1NaivePolicyEvidence,
)

from test_stage4_n6_carriers import _global
from test_stage4_pd import _build


_DIGEST = "1" * 64


@lru_cache(maxsize=1)
def _policy_contexts() -> tuple[
    S1NaivePolicyEvidence,
    InterDiePlanningContext,
    IntraDieSchedulingContext,
]:
    registry = production_registry()
    inter = registry.instantiate(RegistryKind.INTER_DIE, "naive").selection
    standalone = registry.instantiate(
        RegistryKind.STANDALONE_COLLECTIVE, "direct_all_gather"
    ).selection
    intra = registry.instantiate(RegistryKind.INTRA_DIE, "naive").selection
    planning = InterDiePlanningContext.create(
        producer_pass="test_stage4_pd_evidence",
        fused_policy=inter,
        standalone_policy=standalone,
    )
    scheduling = IntraDieSchedulingContext.create(
        producer_pass="test_stage4_pd_evidence",
        policy=intra,
    )
    policy = S1NaivePolicyEvidence(
        selections=(inter, standalone, intra),
        planning_context_id=planning.id,
        scheduling_context_id=scheduling.id,
    )
    policy.validate_against(planning, scheduling)
    return policy, planning, scheduling


@lru_cache(maxsize=2)
def _manifest(fused: bool) -> LinkedProgramManifest:
    return link_stage4(lower_stage4(_global(fused=fused))).manifest


def _artifact(manifest: LinkedProgramManifest) -> Stage4PdArtifactEvidence:
    leaf_fragments = tuple(
        item.fragment if isinstance(item, RegionManifest) else item
        for item in manifest.fragments
    )
    runtime_relocations = sum(
        len(stream.runtime_relocations)
        for fragment in leaf_fragments
        for stream in fragment.core_streams
    )
    address_relocations = sum(
        len(stream.address_relocations)
        for fragment in leaf_fragments
        for stream in fragment.core_streams
    )
    return Stage4PdArtifactEvidence(
        linked_manifest_id=manifest.id,
        linked_manifest_digest=canonical_digest(manifest),
        program_artifact_sha256=_DIGEST,
        artifact_size_bytes=1,
        action_count=len(
            {
                record.source_global_action_id
                for stream in manifest.core_streams
                for record in stream.records
            }
        ),
        fragment_count=len(manifest.fragments),
        record_count=sum(len(stream.records) for stream in manifest.core_streams),
        runtime_relocation_count=runtime_relocations,
        address_relocation_count=address_relocations,
        relocation_count=runtime_relocations + address_relocations,
        address_operand_binding_count=len(manifest.address_operand_bindings),
        state_operand_binding_count=len(manifest.state_operand_bindings),
    )


def _report(
    prefill_tp: int,
    decode_tp: int,
    *,
    fused: bool = False,
) -> tuple[Stage4PdRuntimeReport, object, object, LinkedProgramManifest]:
    plan, oracle = _build(prefill_tp, decode_tp, fused=fused)
    manifest = _manifest(fused)
    memory = tuple(
        Stage4PdMemoryEvidence(
            runtime_core_id=binding.runtime_core_id,
            lsu_issued=1,
            lsu_completed=1,
            lsu_hbm_read_bytes=1,
            lsu_hbm_write_bytes=1,
            lsu_sram_read_bytes=1,
            lsu_sram_write_bytes=1,
            lsu_residual=0,
            dte_residual=0,
        )
        for binding in sorted(
            manifest.core_bindings, key=lambda item: item.runtime_core_id
        )
    )
    runtime_by_logical = {
        binding.logical_core: binding.runtime_core_id
        for binding in manifest.core_bindings
    }
    control = Stage4PdControlEvidence(
        ack_counts=tuple(
            sorted(
                (
                    Stage4PdCoreCount(runtime_by_logical[core], 2)
                    for core in manifest.envelope.expected_ack_cores
                ),
                key=lambda item: item.runtime_core_id,
            )
        ),
        done_counts=tuple(
            sorted(
                (
                    Stage4PdCoreCount(runtime_by_logical[core], 1)
                    for core in manifest.envelope.expected_done_cores
                ),
                key=lambda item: item.runtime_core_id,
            )
        ),
        drain_residuals=tuple(
            Stage4PdNamedCount(name, 0)
            for name in ("collective", "global", "p2p", "timing")
        ),
        all_done_boundary_reached=True,
    )
    endpoint_routes = tuple(
        Stage4PdEndpointRouteEvidence(
            source_rank=item.source_rank,
            destination_rank=item.destination_rank,
            logical_unique_bytes=item.logical_unique_bytes,
            delivered_bytes=item.delivered_bytes,
            state_transfer_count=item.state_transfer_count,
        )
        for item in oracle.endpoint_pair_metrics
    )
    logical_packets = oracle.delivered_bytes // 16
    links = ()
    if logical_packets:
        links = (
            Stage4PdD2DLinkEvidence(
                source_die_id=0,
                destination_die_id=1,
                request_in_packets=oracle.state_transfer_count,
                request_out_packets=oracle.state_transfer_count,
                ack_in_packets=0,
                ack_out_packets=0,
                data_in_packets=logical_packets,
                data_out_packets=logical_packets,
            ),
            Stage4PdD2DLinkEvidence(
                source_die_id=1,
                destination_die_id=0,
                request_in_packets=0,
                request_out_packets=0,
                ack_in_packets=2 * oracle.state_transfer_count,
                ack_out_packets=2 * oracle.state_transfer_count,
                data_in_packets=0,
                data_out_packets=0,
            ),
        )
    d2d = Stage4PdD2DEvidence(
        flow_count=oracle.state_transfer_count,
        logical_packet_count=logical_packets,
        physical_packet_count=logical_packets,
        request_packet_count=oracle.state_transfer_count,
        ack_packet_count=2 * oracle.state_transfer_count,
        logical_bytes=oracle.delivered_bytes,
        byte_hop_bytes=oracle.delivered_bytes,
        state_transport_logical_bytes=oracle.delivered_bytes,
        state_transfer_count=oracle.state_transfer_count,
        unique_endpoint_route_count=oracle.unique_endpoint_route_count,
        endpoint_routes=endpoint_routes,
        links=links,
    )
    program_io = Stage4PdProgramIoEvidence(
        contract_id="program_io",
        contract_digest=_DIGEST,
        mode=ProgramIoMode.TIMING,
        hbm_initialization_count=1,
        sram_initialization_count=1,
        hbm_probe_count=0,
        sram_probe_count=1,
        all_probes_passed=True,
    )
    repeat_fields = {
        "makespan_cycles": 10,
        "marker_digest": _DIGEST,
        "memory_digest": canonical_digest(memory),
        "program_io_digest": canonical_digest(program_io),
        "control_digest": canonical_digest(control),
        "d2d_digest": canonical_digest(d2d),
    }
    policy, _planning, _scheduling = _policy_contexts()
    report = Stage4PdRuntimeReport.create(
        baseline_epoch=STAGE4_PD_BASELINE_EPOCH,
        case_id=(
            "case.stage4.pd_f.tp1"
            if fused
            else f"case.stage4.pd.{prefill_tp}_to_{decode_tp}"
        ),
        mode=oracle.mode,
        reshard=oracle.reshard,
        capability_status=CapabilityStatus.E2E_TIMING,
        plan_id=plan.id,
        plan_digest=canonical_digest(plan),
        oracle_id=oracle.id,
        oracle_digest=canonical_digest(oracle),
        policy=policy,
        tool_digests=tuple(
            Stage4PdNamedDigest(name, _DIGEST)
            for name in ("finalizer", "npusim", "resolver", "runner")
        ),
        input_digests=(
            Stage4PdNamedDigest("hardware", _DIGEST),
            Stage4PdNamedDigest("manifest", canonical_digest(manifest)),
            Stage4PdNamedDigest("mapping", _DIGEST),
            Stage4PdNamedDigest("oracle", canonical_digest(oracle)),
            Stage4PdNamedDigest("plan", canonical_digest(plan)),
            Stage4PdNamedDigest("policy", canonical_digest(policy)),
            Stage4PdNamedDigest("program_io", _DIGEST),
            Stage4PdNamedDigest("simulation", _DIGEST),
            Stage4PdNamedDigest("spec", plan.source_spec_digest),
        ),
        hardware_digest=_DIGEST,
        simulation_digest=_DIGEST,
        mapping_digest=_DIGEST,
        artifact=_artifact(manifest),
        program_io=program_io,
        memory=memory,
        control=control,
        d2d=d2d,
        repeat_count=2,
        makespan_cycles=10,
        repeats=tuple(
            Stage4PdRepeatEvidence(run_index=index, **repeat_fields)
            for index in range(2)
        ),
        timing_execution=True,
        state_transport_exact=True,
        program_io_boundary_exact=True,
        model_functional=False,
    )
    return report, plan, oracle, manifest


def _recreate(
    report: Stage4PdRuntimeReport,
    **changes: object,
) -> Stage4PdRuntimeReport:
    semantic = report._semantic_key() | changes
    memory = semantic["memory"]
    program_io = semantic["program_io"]
    control = semantic["control"]
    d2d = semantic["d2d"]
    semantic["repeats"] = tuple(
        Stage4PdRepeatEvidence(
            run_index=index,
            makespan_cycles=semantic["makespan_cycles"],
            marker_digest=_DIGEST,
            memory_digest=canonical_digest(memory),
            program_io_digest=canonical_digest(program_io),
            control_digest=canonical_digest(control),
            d2d_digest=canonical_digest(d2d),
        )
        for index in range(semantic["repeat_count"])
    )
    return Stage4PdRuntimeReport.create(**semantic)


class Stage4PdEvidenceTest(unittest.TestCase):
    def test_fused_pds_and_pdr_round_trip_and_exact_closure(self) -> None:
        self.assertIs(PublicStage4PdRuntimeReport, Stage4PdRuntimeReport)
        self.assertEqual(
            STAGE4_PD_RUNTIME_REPORT_SCHEMA_VERSION,
            "wafer_frontend.stage4_pd_runtime_report/v1alpha2",
        )
        for prefill_tp, decode_tp, fused, expected in (
            (1, 1, True, (0, 0, 0)),
            (1, 1, False, (1, 4, 1024)),
            (2, 1, False, (2, 8, 1024)),
            (1, 2, False, (2, 8, 1024)),
        ):
            with self.subTest(
                prefill_tp=prefill_tp,
                decode_tp=decode_tp,
                fused=fused,
            ):
                report, plan, oracle, manifest = _report(
                    prefill_tp, decode_tp, fused=fused
                )
                _policy, planning, scheduling = _policy_contexts()
                report.validate_against(
                    plan, oracle, manifest, planning, scheduling
                )
                self.assertEqual(
                    (
                        report.d2d.unique_endpoint_route_count,
                        report.d2d.state_transfer_count,
                        report.d2d.state_transport_logical_bytes,
                    ),
                    expected,
                )
                self.assertEqual(
                    loads_dataclass(
                        Stage4PdRuntimeReport,
                        canonical_json(report),
                        path="report",
                    ),
                    report,
                )

    def test_strict_serde_version_and_stable_id_fail_closed(self) -> None:
        report, plan, oracle, manifest = _report(1, 1, fused=True)
        self.assertEqual(report.baseline_epoch, S1_N_BASELINE_EPOCH)
        raw = json.loads(canonical_json(report))
        raw["unexpected"] = True
        with self.assertRaisesRegex(SchemaError, "unknown field"):
            loads_dataclass(
                Stage4PdRuntimeReport,
                json.dumps(raw),
                path="report",
            )
        del raw["unexpected"]
        del raw["d2d"]
        with self.assertRaisesRegex(SchemaError, "missing required field"):
            loads_dataclass(
                Stage4PdRuntimeReport,
                json.dumps(raw),
                path="report",
            )
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(
                report,
                schema_version="wafer_frontend.stage4_pd_runtime_report/v0",
            ).validate()
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            replace(report, id="forged").validate()
        with self.assertRaisesRegex(SchemaError, "s1-n-v1"):
            _recreate(report, baseline_epoch="moving").validate()
        with self.assertRaisesRegex(SchemaError, "canonical order"):
            _recreate(
                report, tool_digests=tuple(reversed(report.tool_digests))
            ).validate()
        with self.assertRaisesRegex(SchemaError, "exactly cover"):
            _recreate(report, tool_digests=report.tool_digests[:-1]).validate()
        with self.assertRaisesRegex(SchemaError, "exactly cover"):
            _recreate(
                report,
                tool_digests=(
                    *report.tool_digests[:-1],
                    Stage4PdNamedDigest("unknown", _DIGEST),
                ),
            ).validate()
        bad_inputs = tuple(
            replace(item, digest="2" * 64)
            if item.name == "manifest"
            else item
            for item in report.input_digests
        )
        with self.assertRaisesRegex(SchemaError, "input digests"):
            _recreate(report, input_digests=bad_inputs).validate_against(
                plan, oracle, manifest, *_policy_contexts()[1:]
            )
        with self.assertRaisesRegex(SchemaError, "supplied plan"):
            _recreate(report, plan_digest="2" * 64).validate_against(
                plan, oracle, manifest, *_policy_contexts()[1:]
            )

    def test_manifest_memory_control_and_proof_tamper_fail_closed(self) -> None:
        report, plan, oracle, manifest = _report(1, 1)
        artifact_cases = (
            replace(report.artifact, linked_manifest_digest="2" * 64),
            replace(report.artifact, action_count=report.artifact.action_count + 1),
            replace(report.artifact, record_count=report.artifact.record_count + 1),
            replace(
                report.artifact,
                runtime_relocation_count=(
                    report.artifact.runtime_relocation_count + 1
                ),
                relocation_count=report.artifact.relocation_count + 1,
            ),
        )
        for artifact in artifact_cases:
            with self.subTest(artifact=artifact), self.assertRaisesRegex(
                SchemaError, "linked manifest"
            ):
                _recreate(report, artifact=artifact).validate_against(
                    plan, oracle, manifest, *_policy_contexts()[1:]
                )
        foreign_core = replace(
            report.memory[0], runtime_core_id=1
        )
        with self.assertRaisesRegex(SchemaError, "every manifest runtime core"):
            _recreate(
                report,
                memory=(foreign_core, *report.memory[1:]),
            ).validate_against(
                plan, oracle, manifest, *_policy_contexts()[1:]
            )
        bad_ack = (
            replace(report.control.ack_counts[0], count=3),
            *report.control.ack_counts[1:],
        )
        with self.assertRaisesRegex(SchemaError, "ACK/DONE"):
            _recreate(
                report,
                control=replace(report.control, ack_counts=bad_ack),
            ).validate_against(
                plan, oracle, manifest, *_policy_contexts()[1:]
            )
        for field_name, value in (
            ("timing_execution", False),
            ("state_transport_exact", False),
            ("program_io_boundary_exact", False),
            ("model_functional", True),
        ):
            with self.subTest(field=field_name), self.assertRaisesRegex(
                SchemaError, "timing/state-transport/ProgramIo"
            ):
                Stage4PdRuntimeReport.create(
                    **(report._semantic_key() | {field_name: value})
                )

    def test_endpoint_physical_control_repeat_and_source_tamper(self) -> None:
        report, plan, oracle, manifest = _report(2, 1)
        first, second = report.d2d.endpoint_routes
        route_cases = (
            (
                Stage4PdEndpointRouteEvidence(
                    source_rank=first.source_rank,
                    destination_rank=first.destination_rank,
                    logical_unique_bytes=1024,
                    delivered_bytes=1024,
                    state_transfer_count=8,
                ),
            ),
            (
                replace(
                    first,
                    logical_unique_bytes=first.logical_unique_bytes + 16,
                    delivered_bytes=first.delivered_bytes + 16,
                ),
                replace(
                    second,
                    logical_unique_bytes=second.logical_unique_bytes - 16,
                    delivered_bytes=second.delivered_bytes - 16,
                ),
            ),
            (
                replace(
                    first,
                    state_transfer_count=first.state_transfer_count + 2,
                ),
                replace(
                    second,
                    state_transfer_count=second.state_transfer_count - 2,
                ),
            ),
        )
        for routes in route_cases:
            with self.subTest(routes=routes), self.assertRaisesRegex(
                SchemaError, "PD oracle"
            ):
                _recreate(
                    report,
                    d2d=replace(
                        report.d2d,
                        unique_endpoint_route_count=len(routes),
                        endpoint_routes=routes,
                    ),
                ).validate_against(
                    plan, oracle, manifest, *_policy_contexts()[1:]
                )
        with self.assertRaisesRegex(SchemaError, "per-link"):
            _recreate(
                report,
                d2d=replace(
                    report.d2d,
                    links=(
                        replace(
                            report.d2d.links[0],
                            request_in_packets=(
                                report.d2d.links[0].request_in_packets + 1
                            ),
                            request_out_packets=(
                                report.d2d.links[0].request_out_packets + 1
                            ),
                        ),
                        *report.d2d.links[1:],
                    ),
                ),
            )
        with self.assertRaisesRegex(SchemaError, "residuals"):
            _recreate(
                report,
                control=replace(
                    report.control,
                    drain_residuals=(
                        replace(report.control.drain_residuals[0], count=1),
                        *report.control.drain_residuals[1:],
                    ),
                ),
            )
        with self.assertRaisesRegex(SchemaError, "repeat"):
            replace(
                report,
                repeats=(
                    report.repeats[0],
                    replace(report.repeats[1], makespan_cycles=11),
                ),
            ).validate()
        fused_report, fused_plan, fused_oracle, fused_manifest = _report(
            1, 1, fused=True
        )
        with self.assertRaisesRegex(SchemaError, "supplied plan"):
            report.validate_against(
                fused_plan, oracle, manifest, *_policy_contexts()[1:]
            )
        with self.assertRaisesRegex(SchemaError, "supplied oracle"):
            report.validate_against(
                plan, fused_oracle, manifest, *_policy_contexts()[1:]
            )
        with self.assertRaisesRegex(SchemaError, "input digests"):
            report.validate_against(
                plan, oracle, fused_manifest, *_policy_contexts()[1:]
            )
        fused_report.validate_against(
            fused_plan,
            fused_oracle,
            fused_manifest,
            *_policy_contexts()[1:],
        )


if __name__ == "__main__":
    unittest.main()
