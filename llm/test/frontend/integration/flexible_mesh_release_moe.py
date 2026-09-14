"""Flexible-MoE adapter for the common strict flexible-Mesh release runner."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.flexible_moe_production import (
    build_flexible_moe_production_program_io,
    lower_link_flexible_moe_production,
)
from llm.frontend.wafer_frontend.passes.flexible_moe import (
    build_round_robin_flexible_moe_spec,
    compile_flexible_moe_baseline,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    RecordOpcode,
    RuntimeSymbolKind,
)
from llm.frontend.wafer_frontend.schema.flexible_mesh_release import (
    FlexibleMeshReleaseCase,
    FlexibleMeshReleaseFamily,
    FlexibleMeshReleaseResidual,
    expected_completion_markers,
)
from llm.frontend.wafer_frontend.schema.flexible_moe import (
    FlexibleMoeSpec,
    FlexibleMoeMode,
    MoeRectActionKind,
    MoeRectFlowStage,
)
from llm.frontend.wafer_frontend.schema.program_io import ProgramIoContract
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.frontend.wafer_frontend.schema.serde import canonical_digest

from run_flexible_mesh_release import (
    FlexibleMeshReleaseMaterialized,
    FlexibleMeshReleaseObservation,
)
from flexible_mesh_runtime_markers import _validate_program_io, validate_credit_balance
from flexible_mesh_release_hardware import specialize_release_hardware
from flexible_mesh_release_profiles import release_trace_model_digest


_MODE = {
    FlexibleMeshReleaseFamily.MOE_INFERENCE: FlexibleMoeMode.INFERENCE,
    FlexibleMeshReleaseFamily.MOE_TRAIN: FlexibleMoeMode.TRAIN,
}
_CORES_PER_DIE = 16


class FlexibleMoeReleaseAdapter:
    """Produce and observe real timing-only Direct-XY MoE programs."""

    def __init__(
        self,
        family: FlexibleMeshReleaseFamily,
        *,
        hardware_template_json: str,
        mapping_text: str,
        trusted_spec_builders: tuple[
            tuple[str, Callable[[RectMeshSpec, FlexibleMoeMode], FlexibleMoeSpec]], ...
        ] = (),
    ) -> None:
        if family not in _MODE:
            raise SchemaError("requires one MoE release family", path="family")
        if not mapping_text:
            raise SchemaError("mapping text is empty", path="mapping_text")
        # Validate the template at the adapter boundary, before any case runs.
        specialize_release_hardware(hardware_template_json, 1, 1)
        self.family = family
        self._hardware_template_json = hardware_template_json
        self._mapping_text = mapping_text
        self._trusted_spec_builders = dict(trusted_spec_builders)
        if (
            len(self._trusted_spec_builders) != len(trusted_spec_builders)
            or any(
                type(digest) is not str
                or len(digest) != 64
                or digest != digest.lower()
                or any(character not in "0123456789abcdef" for character in digest)
                or digest == release_trace_model_digest(family)
                for digest in self._trusted_spec_builders
            )
        ):
            raise SchemaError("trusted supplemental profiles are invalid", path="trusted_spec_builders")
        self._sources: dict[str, tuple[object, object, object]] = {}

    def _spec(self, case: FlexibleMeshReleaseCase) -> FlexibleMoeSpec:
        if case.trace_model_digest == release_trace_model_digest(case.family):
            spec = build_round_robin_flexible_moe_spec(
                case.mesh,
                _MODE[self.family],
                routing_shift=0 if case.mesh.rank_count == 1 else 1,
            )
        else:
            builder = self._trusted_spec_builders.get(case.trace_model_digest)
            if builder is None:
                raise SchemaError(
                    "adapter trace/model profile drifted",
                    path="case.trace_model_digest",
                )
            spec = builder(case.mesh, _MODE[self.family])
        spec.validate("supplemental_spec")
        if spec.mesh != case.mesh or spec.mode is not _MODE[self.family]:
            raise SchemaError("trusted spec builder drifted", path="supplemental_spec")
        return spec

    def _hardware_json(self, case: FlexibleMeshReleaseCase) -> str:
        return specialize_release_hardware(
            self._hardware_template_json,
            case.mesh.rows,
            case.mesh.columns,
        )

    @property
    def mapping_sha256(self) -> str:
        return hashlib.sha256(self._mapping_text.encode("utf-8")).hexdigest()

    def expected_hardware_sha256(self, case: FlexibleMeshReleaseCase) -> str:
        if case.family is not self.family:
            raise SchemaError("adapter family drifted", path="case.family")
        if (
            case.trace_model_digest != release_trace_model_digest(case.family)
            and case.trace_model_digest not in self._trusted_spec_builders
        ):
            raise SchemaError("adapter trace/model profile drifted", path="case.trace_model_digest")
        return hashlib.sha256(self._hardware_json(case).encode("utf-8")).hexdigest()

    def materialize(
        self,
        case: FlexibleMeshReleaseCase,
    ) -> FlexibleMeshReleaseMaterialized:
        if case.family is not self.family:
            raise SchemaError("adapter family drifted", path="case.family")
        spec = self._spec(case)
        plan = compile_flexible_moe_baseline(spec)
        artifacts = lower_link_flexible_moe_production(
            plan, spec, physical_region_name="dense_release",
        )
        self._sources[case.id] = (spec, plan, artifacts)
        tags = {
            symbol.id
            for fragment in artifacts.fragments
            for symbol in fragment.runtime_symbols
            if symbol.kind is RuntimeSymbolKind.DTE_TOKEN
        }
        result = FlexibleMeshReleaseMaterialized(
            case_id=case.id,
            spec_digest=spec.digest,
            plan_digest=canonical_digest(plan),
            manifest=artifacts.manifest,
            hardware_json=self._hardware_json(case),
            mapping_text=self._mapping_text,
            peak_sessions_per_core_per_wave=plan.max_sessions_per_rank_wave,
            transport_tag_count=len(tags),
        )
        result.validate(case)
        return result

    def build_program_io(
        self,
        materialized: FlexibleMeshReleaseMaterialized,
        artifact_sha256: str,
    ) -> ProgramIoContract:
        source = self._sources.get(materialized.case_id)
        if source is None:
            raise SchemaError("case was not materialized", path="materialized.case_id")
        spec, plan, artifacts = source
        effective_sha = (
            "f" * 64 if artifact_sha256 == "0" * 64 else artifact_sha256
        )
        return build_flexible_moe_production_program_io(
            artifacts, plan, spec, effective_sha,
        )

    def observe(
        self,
        case: FlexibleMeshReleaseCase,
        materialized: FlexibleMeshReleaseMaterialized,
        contract: ProgramIoContract,
        artifact_sha256: str,
        npusim_output: str,
    ) -> FlexibleMeshReleaseObservation:
        source = self._sources.get(case.id)
        if source is None:
            raise SchemaError("case was not materialized", path="case.id")
        spec, plan, artifacts = source
        if contract.program_artifact_sha256 != artifact_sha256:
            raise SchemaError("ProgramIO artifact SHA drifted", path="artifact_sha256")
        makespans = re.findall(r"\[SIM_RESULT\] makespan_cycles=(\d+)", npusim_output)
        if len(makespans) != 1:
            raise SchemaError("requires one exact makespan marker", path="npusim_output")
        _validate_program_io(npusim_output, artifact_sha256, contract)
        validate_credit_balance(npusim_output)
        expected_cores = tuple(rank * _CORES_PER_DIE for rank in range(case.mesh.rank_count))
        memory_cores = tuple(sorted(int(value) for value in re.findall(
            r"\[PROGRAM_MEMORY\] core=(\d+)", npusim_output,
        )))
        if memory_cores != expected_cores:
            raise SchemaError("runtime core coverage is not exact", path="npusim_output")
        memory = tuple(tuple(map(int, item)) for item in re.findall(
            r"\[PROGRAM_MEMORY\] core=(\d+) lsu_issued=(\d+) lsu_completed=(\d+)",
            npusim_output,
        ))
        if (
            len(memory) != case.mesh.rank_count
            or any(issued <= 0 or issued != completed for _core, issued, completed in memory)
        ):
            raise SchemaError("LSU execution closure is not exact", path="npusim_output")
        residual_values = tuple(int(value) for value in re.findall(
            r"(?:lsu_residual|dte_residual|router_residual|d2d_link_residual)=(\d+)",
            npusim_output,
        ))
        if not residual_values or any(residual_values):
            raise SchemaError("runtime residual is nonzero or absent", path="npusim_output")
        p5_drains = tuple(tuple(map(int, item)) for item in re.findall(
            r"\[P5 P2P DRAIN\] core=(\d+) residual=(\d+)", npusim_output,
        ))
        expected_p5_cores = expected_cores if plan.flows else ()
        if (
            tuple(sorted(core for core, _residual in p5_drains)) != expected_p5_cores
            or any(residual for _core, residual in p5_drains)
        ):
            raise SchemaError("P5 drain coverage is not exact", path="npusim_output")
        timing_drains = re.findall(
            r"\[P5 P2P TIMING DRAIN\] residual=(\d+)", npusim_output,
        )
        if timing_drains != ["0"]:
            raise SchemaError("P5 timing drain is not exact", path="npusim_output")
        collective = re.findall(
            r"\[COLL_DRAIN\] tree_entries=(\d+) reduce_nodes=(\d+) "
            r"barriers=(\d+) gather=(\d+) reduce_rx=(\d+) endpoints=(\d+) "
            r"dte_tokens=(\d+) event=(\d+)",
            npusim_output,
        )
        if len(collective) != 1 or any(map(int, collective[0])):
            raise SchemaError("collective drain is not exact", path="npusim_output")
        hostlane = re.findall(r"\[HOSTLANE\][^\n]*mismatch=(\d+)", npusim_output)
        if hostlane != ["0"]:
            raise SchemaError("HOSTLANE completion mismatch", path="npusim_output")
        if plan.flows:
            d2d = re.findall(r"\[D2D_TYPE\][^\n]*data_in=(\d+) data_out=(\d+)", npusim_output)
            if len(d2d) != 1 or min(map(int, d2d[0])) <= 0:
                raise SchemaError("remote MoE requires positive D2D data", path="npusim_output")
            p2p = tuple(tuple(map(int, item)) for item in re.findall(
                r"\[P5 P2P STATS\][^\n]*tx_local_completions=(\d+) "
                r"rx_local_completions=(\d+)",
                npusim_output,
            ))
            if (
                len(p2p) != case.mesh.rank_count
                or sum(item[0] for item in p2p) != len(plan.flows)
                or sum(item[1] for item in p2p) != len(plan.flows)
            ):
                raise SchemaError("P2P completion count does not cover every typed flow", path="npusim_output")
        opcodes = tuple(
            record.opcode
            for fragment in materialized.manifest.fragments
            for stream in fragment.core_streams
            for record in stream.records
        )
        actions = tuple(plan.actions)
        if self.family is FlexibleMeshReleaseFamily.MOE_INFERENCE:
            if set(item.stage for item in plan.flows) - {
                MoeRectFlowStage.DISPATCH, MoeRectFlowStage.COMBINE,
            }:
                raise SchemaError("inference transport stage closure drifted", path="plan.flows")
        else:
            ranks = case.mesh.rank_count
            if (
                opcodes.count(RecordOpcode.SGD_UPDATE) != 2 * ranks
                or opcodes.count(RecordOpcode.LSU_STORE) != 2 * ranks
                or sum(item.kind in (
                    MoeRectActionKind.EXPERT_WGRAD, MoeRectActionKind.GATE_WGRAD,
                ) for item in actions) != 2 * ranks
                or sum(
                    item.kind is MoeRectActionKind.GATE_GRADIENT_LOCAL_REDUCE
                    for item in actions
                ) != ranks
                or opcodes.count(RecordOpcode.LOCAL_REDUCE) != 5 * ranks
                or sum(item.stage is MoeRectFlowStage.GATE_ALL_REDUCE for item in plan.flows)
                != 2 * max(0, ranks - 1)
                or len(contract.output_probes) != 2 * ranks
            ):
                raise SchemaError("train gradient/optimizer/state closure is incomplete", path="materialized")
        host_done = re.findall(r"\[HOSTSIG\] done=([^,\n]*(?:,[^,\n]*)*)?, ack=", npusim_output)
        if len(host_done) != 1 or any(
            f"{runtime_core}:1" not in host_done[0] for runtime_core in expected_cores
        ):
            raise SchemaError("HOSTSIG completion coverage is not exact", path="npusim_output")
        marker_lines = tuple(
            line for line in npusim_output.splitlines()
            if any(key in line for key in (
                "[SIM_RESULT]", "[PROGRAM_MEMORY]", "phase=verify",
                "[D2D_TYPE]", "TIMING DRAIN]", "[DRAIN]", "[CREDIT]",
            ))
        )
        result = FlexibleMeshReleaseObservation(
            makespan_cycles=int(makespans[0]),
            marker_digest=hashlib.sha256("\n".join(marker_lines).encode()).hexdigest(),
            residual=FlexibleMeshReleaseResidual(0, 0, 0, 0, 0, 0),
            rank_coverage=tuple(range(case.mesh.rank_count)),
            core_coverage=tuple(range(case.mesh.rank_count)),
            completion_markers=expected_completion_markers(case.family),
        )
        result.validate(case)
        return result


__all__ = ["FlexibleMoeReleaseAdapter"]
