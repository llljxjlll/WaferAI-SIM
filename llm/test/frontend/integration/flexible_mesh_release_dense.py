"""Stateful Dense-train adapter for the strict flexible-Mesh release runner."""

from __future__ import annotations

import hashlib
import json
import re

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.flexible_dense_backward import (
    build_flexible_dense_backward_program_io,
    materialize_flexible_dense_backward,
)
from llm.frontend.wafer_frontend.passes.flexible_dense_train import (
    materialize_flexible_dense_train_forward,
)
from llm.frontend.wafer_frontend.passes.load_fabric import (
    hbm_address_spaces_from_data,
    physical_fabric_from_data,
)
from llm.frontend.wafer_frontend.schema._validation_session import (
    builder_validation_session,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    RecordOpcode,
    RuntimeSymbolKind,
)
from llm.frontend.wafer_frontend.schema.experiment import (
    EXPERIMENT_SCHEMA_VERSION,
    ExperimentSpec,
)
from llm.frontend.wafer_frontend.schema.flexible_mesh_release import (
    FlexibleMeshReleaseCase,
    FlexibleMeshReleaseFamily,
    FlexibleMeshReleaseResidual,
    expected_completion_markers,
)
from llm.frontend.wafer_frontend.schema.flexible_dense_train import (
    FlexibleDenseTrainActionKind,
    FlexibleDenseTrainGradientSyncRole,
)
from llm.frontend.wafer_frontend.schema.program_io import (
    ProgramIoContract,
    ProgramIoTargetKind,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, from_data

from flexible_mesh_runtime_markers import _validate_program_io, validate_credit_balance
from flexible_mesh_release_hardware import (
    p5_large_hardware_template_json,
    specialize_release_hardware,
)
from flexible_mesh_release_profiles import release_trace_model_digest
from run_flexible_mesh_release import (
    FlexibleMeshReleaseMaterialized,
    FlexibleMeshReleaseObservation,
)


def _dense_release_spec(rows: int, columns: int) -> ExperimentSpec:
    raw = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "model": {
            "source": "analytic",
            "arch": "llama",
            "V": 8 * columns,
            "H": 4 * columns,
            "I": 8 * columns,
            "NH": columns,
            "KVH": columns,
            "DH": 4,
            "rotary_dim": 4,
            "L": 1,
            "dtype": "fp16",
            "tie_word_embeddings": False,
            "rms_norm_epsilon": 1.0e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 64,
            "moe": None,
        },
        "hardware": {"ref": "llm/test/program/p5_large_hardware.json"},
        "workload": {
            "mode": "train",
            "infer": None,
            "train": {
                "global_batch": rows,
                "micro_batch": 1,
                "seq_len": columns,
                "backward": False,
                "optimizer": "none",
                "structure": {
                    "micro_batch_count": 1,
                    "pp_schedule": "gpipe",
                    "interleave_chunks": 1,
                    "recompute": "none",
                },
            },
        },
        "parallel": {
            "instances": [{
                "id": "T0",
                "role": "train",
                "tp": columns,
                "sp": columns > 1,
                "replicas": 1,
                "dp": rows,
                "pp": 1,
                "ep": 1,
            }],
        },
        "placement": {"strategy": "compact", "groups": []},
        "policy": {
            "partition": "gemm_coll",
            "inter_die": "naive",
            "intra_die": "naive",
        },
        "backend": {
            "execution": "unified_stream",
            "ordinary_lowering": "json_coarse",
            "fused_lowering": "isa_region",
            "standalone_collective_lowering": "strict_actions",
            "reduction_contract": {
                "accumulate": "fp32",
                "rounding": "rne",
                "validation": "timing",
            },
            "transport": "strict",
            "static_link": True,
            "dynamic_region_dispatch": False,
        },
    }
    return from_data(ExperimentSpec, raw, path="dense_release.spec")


def _validate_dense_credit_closure(npusim_output: str) -> None:
    credits = re.findall(
        r"\[CREDIT\] data_balanced=(\d+) ctrl_balanced=(\d+)", npusim_output,
    )
    if credits != [("1", "1")]:
        raise SchemaError("credit closure is not exact", path="npusim_output")


def _validate_dense_p2p_closure(
    npusim_output: str,
    expected_cores: tuple[int, ...],
    expected_tx_by_core: dict[int, int],
    expected_rx_by_core: dict[int, int],
) -> None:
    p5_stats = tuple(tuple(map(int, row)) for row in re.findall(
        r"\[P5 P2P STATS\] core=(\d+)[^\n]*"
        r"tx_local_completions=(\d+) rx_local_completions=(\d+)",
        npusim_output,
    ))
    has_transport = any(expected_tx_by_core.values()) or any(
        expected_rx_by_core.values()
    )
    if has_transport and (
        tuple(sorted(core for core, _, _ in p5_stats)) != expected_cores
        or any(
            tx != expected_tx_by_core[core]
            or rx != expected_rx_by_core[core]
            for core, tx, rx in p5_stats
        )
    ):
        raise SchemaError(
            "P2P completions do not match typed DP flows", path="npusim_output",
        )
    if not has_transport and p5_stats:
        raise SchemaError(
            "no-transport DP case emitted P2P completions", path="npusim_output",
        )
    p5 = tuple(tuple(map(int, row)) for row in re.findall(
        r"\[P5 P2P DRAIN\] core=(\d+) residual=(\d+)", npusim_output,
    ))
    if has_transport and (
        tuple(sorted(core for core, _ in p5)) != expected_cores
        or any(residual for _, residual in p5)
    ):
        raise SchemaError("P2P drain closure is not exact", path="npusim_output")
    if not has_transport and p5:
        raise SchemaError(
            "no-transport DP case emitted P2P drain state", path="npusim_output",
        )
    if re.findall(
        r"\[P5 P2P TIMING DRAIN\] residual=(\d+)", npusim_output,
    ) != ["0"]:
        raise SchemaError("P2P timing drain closure is not exact", path="npusim_output")



class FlexibleDenseReleaseAdapter:
    """Materialize and observe one complete synthetic Dense train step."""

    family = FlexibleMeshReleaseFamily.DENSE_TRAIN

    def __init__(
        self,
        *,
        mapping_text: str,
        hardware_template_json: str | None = None,
    ) -> None:
        if type(mapping_text) is not str or not mapping_text:
            raise SchemaError("mapping text is empty", path="mapping_text")
        if hardware_template_json is None:
            hardware_template_json = p5_large_hardware_template_json()
        # Validate the exact bound template before accepting any case.
        specialize_release_hardware(hardware_template_json, 1, 1)
        self._mapping_text = mapping_text
        self._hardware_template_json = hardware_template_json
        self._sources: dict[str, object] = {}

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
        if case.trace_model_digest != release_trace_model_digest(case.family):
            raise SchemaError(
                "adapter trace/model profile drifted",
                path="case.trace_model_digest",
            )
        return hashlib.sha256(self._hardware_json(case).encode("utf-8")).hexdigest()

    def materialize(
        self, case: FlexibleMeshReleaseCase,
    ) -> FlexibleMeshReleaseMaterialized:
        with builder_validation_session():
            return self._materialize(case)

    def _materialize(
        self, case: FlexibleMeshReleaseCase,
    ) -> FlexibleMeshReleaseMaterialized:
        if case.family is not self.family:
            raise SchemaError("adapter family drifted", path="case.family")
        if case.trace_model_digest != release_trace_model_digest(case.family):
            raise SchemaError("adapter trace/model profile drifted", path="case.trace_model_digest")
        hardware_json = self._hardware_json(case)
        raw = json.loads(hardware_json)
        fabric = physical_fabric_from_data(raw)
        spaces = hbm_address_spaces_from_data(raw)
        spec = _dense_release_spec(case.mesh.rows, case.mesh.columns)
        forward = materialize_flexible_dense_train_forward(
            spec, case.mesh, fabric, spaces,
        )
        linked = materialize_flexible_dense_backward(forward, fabric, spaces)
        self._sources[case.id] = linked
        tags = {
            symbol.id
            for fragment in linked.manifest.fragments
            for symbol in fragment.runtime_symbols
            if symbol.kind is RuntimeSymbolKind.DTE_TOKEN
        }
        result = FlexibleMeshReleaseMaterialized(
            case_id=case.id,
            spec_digest=canonical_digest(spec),
            plan_digest=canonical_digest(linked.plan),
            manifest=linked.manifest,
            hardware_json=hardware_json,
            mapping_text=self._mapping_text,
            peak_sessions_per_core_per_wave=(0 if case.mesh.rows == 1 else 2),
            transport_tag_count=len(tags),
        )
        result.validate(case)
        return result

    def build_program_io(
        self,
        materialized: FlexibleMeshReleaseMaterialized,
        artifact_sha256: str,
    ) -> ProgramIoContract:
        linked = self._sources.get(materialized.case_id)
        if linked is None:
            raise SchemaError("case was not materialized", path="materialized.case_id")
        return build_flexible_dense_backward_program_io(
            linked, artifact_sha256,
        )

    def observe(
        self,
        case: FlexibleMeshReleaseCase,
        materialized: FlexibleMeshReleaseMaterialized,
        contract: ProgramIoContract,
        artifact_sha256: str,
        npusim_output: str,
    ) -> FlexibleMeshReleaseObservation:
        linked = self._sources.get(case.id)
        if linked is None:
            raise SchemaError("case was not materialized", path="case.id")
        if contract.program_artifact_sha256 != artifact_sha256:
            raise SchemaError("ProgramIO artifact SHA drifted", path="artifact_sha256")
        _validate_program_io(npusim_output, artifact_sha256, contract)
        validate_credit_balance(npusim_output)
        makespans = re.findall(
            r"\[SIM_RESULT\] makespan_cycles=(\d+)", npusim_output,
        )
        if len(makespans) != 1 or int(makespans[0]) <= 0:
            raise SchemaError("requires one positive makespan", path="npusim_output")
        expected_cores = tuple(sorted(
            binding.runtime_core_id
            for binding in materialized.manifest.core_bindings
        ))
        memory = tuple(tuple(map(int, row)) for row in re.findall(
            r"\[PROGRAM_MEMORY\] core=(\d+) lsu_issued=(\d+) "
            r"lsu_completed=(\d+)[^\n]*lsu_residual=(\d+) "
            r"dte_residual=(\d+)",
            npusim_output,
        ))
        if (
            tuple(sorted(row[0] for row in memory)) != expected_cores
            or any(
                row[1] <= 0 or row[1] - row[2] != row[3]
                or row[3] not in (0, 1) or row[4] != 0 for row in memory
            )
        ):
            raise SchemaError("LSU/DTE snapshot accounting is not exact", path="npusim_output")
        residuals = tuple(int(value) for value in re.findall(
            r"(?:router_residual|d2d_link_residual)=(\d+)", npusim_output,
        ))
        if len(residuals) != 2 or any(residuals):
            raise SchemaError("router/D2D residual closure is absent", path="npusim_output")
        _validate_dense_credit_closure(npusim_output)
        expected_tx_by_core = {
            binding.runtime_core_id: sum(
                action.rank == binding.logical_core.die_id
                and action.send_peer_rank is not None
                for action in linked.plan.rank_actions
            )
            for binding in materialized.manifest.core_bindings
        }
        expected_rx_by_core = {
            binding.runtime_core_id: sum(
                action.rank == binding.logical_core.die_id
                and action.receive_peer_rank is not None
                for action in linked.plan.rank_actions
            )
            for binding in materialized.manifest.core_bindings
        }
        _validate_dense_p2p_closure(
            npusim_output,
            expected_cores,
            expected_tx_by_core,
            expected_rx_by_core,
        )
        collective = re.findall(
            r"\[COLL_DRAIN\] tree_entries=(\d+) reduce_nodes=(\d+) "
            r"barriers=(\d+) gather=(\d+) reduce_rx=(\d+) endpoints=(\d+) "
            r"dte_tokens=(\d+) event=(\d+)", npusim_output,
        )
        if len(collective) != 1 or any(map(int, collective[0])):
            raise SchemaError("collective drain is not exact", path="npusim_output")
        hostlane = re.findall(r"\[HOSTLANE\][^\n]*mismatch=(\d+)", npusim_output)
        host_done = re.findall(r"\[HOSTSIG\] done=([^\n]*), ack=", npusim_output)
        if (
            hostlane != ["0"] or len(host_done) != 1
            or any(f"{core}:1" not in host_done[0] for core in expected_cores)
        ):
            raise SchemaError("host completion coverage is not exact", path="npusim_output")

        fragment = materialized.manifest.fragments[0]
        opcodes = tuple(
            record.opcode for stream in fragment.core_streams
            for record in stream.records
        )
        send_count = sum(
            action.send_peer_rank is not None
            for action in linked.plan.rank_actions
        )
        receive_count = sum(
            action.receive_peer_rank is not None
            for action in linked.plan.rank_actions
        )
        reduce_count = sum(
            action.gradient_sync_role
            is FlexibleDenseTrainGradientSyncRole.REDUCE_RECEIVE
            for action in linked.plan.rank_actions
        )
        state_count = len(fragment.state_abi)
        hbm_inits = sum(
            item.target.kind is ProgramIoTargetKind.HBM
            for item in contract.initializations
        )
        if (
            opcodes.count(RecordOpcode.DTE_SEND) != send_count
            or opcodes.count(RecordOpcode.DTE_RECV) != receive_count
            or opcodes.count(RecordOpcode.DTE_WAIT) != receive_count
            or opcodes.count(RecordOpcode.LOCAL_REDUCE) != reduce_count
            or opcodes.count(RecordOpcode.SGD_UPDATE) != state_count
            or opcodes.count(RecordOpcode.LSU_STORE) != state_count
            or hbm_inits != state_count
            or len(contract.output_probes) != state_count
        ):
            raise SchemaError(
                "gradient/optimizer/state closure is incomplete",
                path="materialized",
            )
        if case.mesh.rows > 1:
            d2d = re.findall(
                r"\[D2D_TYPE\][^\n]*data_in=(\d+) data_out=(\d+)",
                npusim_output,
            )
            if len(d2d) != 1 or min(map(int, d2d[0])) <= 0:
                raise SchemaError("DP transport lacks D2D data", path="npusim_output")
        marker_lines = tuple(
            line for line in npusim_output.splitlines()
            if any(key in line for key in (
                "[SIM_RESULT]", "[PROGRAM_MEMORY]", "phase=verify",
                "[D2D_TYPE]", "TIMING DRAIN]", "[COLL_DRAIN]",
                "[DRAIN]", "[HOSTSIG]", "[CREDIT]", "[P5 P2P STATS]",
            ))
        )
        result = FlexibleMeshReleaseObservation(
            makespan_cycles=int(makespans[0]),
            marker_digest=hashlib.sha256(
                "\n".join(marker_lines).encode("utf-8")
            ).hexdigest(),
            residual=FlexibleMeshReleaseResidual(0, 0, 0, 0, 0, 0),
            rank_coverage=tuple(range(case.mesh.rank_count)),
            core_coverage=tuple(range(case.mesh.rank_count)),
            completion_markers=expected_completion_markers(case.family),
        )
        result.validate(case)
        return result


__all__ = ["FlexibleDenseReleaseAdapter"]
