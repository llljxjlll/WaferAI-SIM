"""Versioned, digest-bound intra-die timing calibration table."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .common import stable_artifact_id, validate_nonempty, validate_uint64


INTRA_DIE_TIMING_MODEL_SCHEMA_VERSION = (
    "wafer_frontend.intra_die_timing_model/v1alpha2"
)
INTRA_DIE_TIMING_MODEL_VERSION = "resource_timeline_cost/v5"


def _validate_sha256(value: object, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError("must be a lowercase SHA256 digest", path=path)


@dataclass(frozen=True, slots=True)
class IntraDieTimingModel:
    """Calibrated resource costs bound to one hardware/simulation pair."""

    schema_version: str
    producer_pass: str
    id: str
    model_version: str
    hardware_digest: str
    simulation_digest: str
    compute_setup_cycles: int
    effective_gemm_ops_per_cycle: int
    hbm_load_setup_cycles: int
    hbm_load_bytes_per_cycle: int
    local_transport_setup_cycles: int
    local_transport_bytes_per_cycle: int
    local_reduce_setup_cycles: int
    local_reduce_bytes_per_cycle: int
    sync_issue_cycles: int
    fixed_pipeline_cycles: int

    @classmethod
    def create(
        cls,
        *,
        hardware_digest: str,
        simulation_digest: str,
        compute_setup_cycles: int = 64,
        effective_gemm_ops_per_cycle: int = 20,
        hbm_load_setup_cycles: int = 220,
        hbm_load_bytes_per_cycle: int = 4,
        local_transport_setup_cycles: int = 232,
        local_transport_bytes_per_cycle: int = 256,
        local_reduce_setup_cycles: int = 512,
        local_reduce_bytes_per_cycle: int = 128,
        sync_issue_cycles: int = 8,
        fixed_pipeline_cycles: int = 326,
    ) -> "IntraDieTimingModel":
        semantic = {
            "model_version": INTRA_DIE_TIMING_MODEL_VERSION,
            "hardware_digest": hardware_digest,
            "simulation_digest": simulation_digest,
            "compute_setup_cycles": compute_setup_cycles,
            "effective_gemm_ops_per_cycle": effective_gemm_ops_per_cycle,
            "hbm_load_setup_cycles": hbm_load_setup_cycles,
            "hbm_load_bytes_per_cycle": hbm_load_bytes_per_cycle,
            "local_transport_setup_cycles": local_transport_setup_cycles,
            "local_transport_bytes_per_cycle": local_transport_bytes_per_cycle,
            "local_reduce_setup_cycles": local_reduce_setup_cycles,
            "local_reduce_bytes_per_cycle": local_reduce_bytes_per_cycle,
            "sync_issue_cycles": sync_issue_cycles,
            "fixed_pipeline_cycles": fixed_pipeline_cycles,
        }
        result = cls(
            schema_version=INTRA_DIE_TIMING_MODEL_SCHEMA_VERSION,
            producer_pass="intra_die_timing_calibration",
            id=stable_artifact_id(
                "intra_die_timing_model",
                semantic,
                schema_version=INTRA_DIE_TIMING_MODEL_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "model_version",
                "hardware_digest",
                "simulation_digest",
                "compute_setup_cycles",
                "effective_gemm_ops_per_cycle",
                "hbm_load_setup_cycles",
                "hbm_load_bytes_per_cycle",
                "local_transport_setup_cycles",
                "local_transport_bytes_per_cycle",
                "local_reduce_setup_cycles",
                "local_reduce_bytes_per_cycle",
                "sync_issue_cycles",
                "fixed_pipeline_cycles",
            )
        }

    def validate(self, path: str = "intra_die_timing_model") -> None:
        if self.schema_version != INTRA_DIE_TIMING_MODEL_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "intra_die_timing_calibration":
            raise SchemaError("unsupported producer", path=f"{path}.producer_pass")
        if self.model_version != INTRA_DIE_TIMING_MODEL_VERSION:
            raise SchemaError("unsupported timing model", path=f"{path}.model_version")
        _validate_sha256(self.hardware_digest, f"{path}.hardware_digest")
        _validate_sha256(self.simulation_digest, f"{path}.simulation_digest")
        for name in (
            "compute_setup_cycles",
            "effective_gemm_ops_per_cycle",
            "hbm_load_setup_cycles",
            "hbm_load_bytes_per_cycle",
            "local_transport_setup_cycles",
            "local_transport_bytes_per_cycle",
            "local_reduce_setup_cycles",
            "local_reduce_bytes_per_cycle",
            "sync_issue_cycles",
            "fixed_pipeline_cycles",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        for name in (
            "effective_gemm_ops_per_cycle",
            "hbm_load_bytes_per_cycle",
            "local_transport_bytes_per_cycle",
            "local_reduce_bytes_per_cycle",
        ):
            if getattr(self, name) == 0:
                raise SchemaError("must be positive", path=f"{path}.{name}")
        expected = stable_artifact_id(
            "intra_die_timing_model",
            self._semantic_key(),
            schema_version=INTRA_DIE_TIMING_MODEL_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")

    def validate_binding(
        self,
        *,
        hardware_digest: str,
        simulation_digest: str,
        path: str = "intra_die_timing_model",
    ) -> None:
        self.validate(path)
        _validate_sha256(hardware_digest, "hardware_digest")
        _validate_sha256(simulation_digest, "simulation_digest")
        if self.hardware_digest != hardware_digest:
            raise SchemaError("hardware digest mismatch", path=f"{path}.hardware_digest")
        if self.simulation_digest != simulation_digest:
            raise SchemaError("simulation digest mismatch", path=f"{path}.simulation_digest")


__all__ = [
    "INTRA_DIE_TIMING_MODEL_SCHEMA_VERSION",
    "INTRA_DIE_TIMING_MODEL_VERSION",
    "IntraDieTimingModel",
]
