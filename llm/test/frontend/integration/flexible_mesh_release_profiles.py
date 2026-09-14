"""Trusted family profiles used to derive canonical release case identities."""

from __future__ import annotations

from dataclasses import dataclass

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.flexible_mesh_release import (
    FlexibleMeshReleaseFamily,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest


@dataclass(frozen=True, slots=True)
class FlexibleMeshReleaseFamilyProfile:
    family: FlexibleMeshReleaseFamily
    producer: str
    profile_version: str
    adapter_inputs: tuple[str, ...]

    def validate(self, path: str = "release_family_profile") -> None:
        if type(self.family) is not FlexibleMeshReleaseFamily:
            raise SchemaError("must be a release family", path=f"{path}.family")
        if not self.producer or not self.profile_version:
            raise SchemaError("producer/profile is empty", path=path)
        if (
            type(self.adapter_inputs) is not tuple
            or not self.adapter_inputs
            or any(type(item) is not str or not item for item in self.adapter_inputs)
            or self.adapter_inputs != tuple(sorted(set(self.adapter_inputs)))
        ):
            raise SchemaError(
                "adapter inputs must be non-empty, unique and sorted",
                path=f"{path}.adapter_inputs",
            )

    @property
    def digest(self) -> str:
        self.validate()
        return canonical_digest(self)


_PROFILES = (
    FlexibleMeshReleaseFamilyProfile(
        FlexibleMeshReleaseFamily.DENSE_TRAIN,
        "flexible_mesh_release_dense.FlexibleDenseReleaseAdapter",
        "tiny_dense_dp_rows_tp_columns/v3",
        tuple(sorted((
            "dp_sync=row_major_binary_tree_reduce_broadcast_root0",
            "hardware=p5_large_hardware_shape_specialized_1mib_sram(rows,columns)",
            "release_layers=1",
            "operation=forward_backward_gradient_sync_sgd_state_store",
            "spec=_dense_release_spec(rows,columns)",
        ))),
    ),
    FlexibleMeshReleaseFamilyProfile(
        FlexibleMeshReleaseFamily.MOE_INFERENCE,
        "flexible_mesh_release_moe.FlexibleMoeReleaseAdapter",
        "static_top1_direct_xy_inference/v2",
        tuple(sorted((
            "builder=build_round_robin_flexible_moe_spec",
            "mode=inference",
            "routing_shift=0_if_rank1_else_1",
        ))),
    ),
    FlexibleMeshReleaseFamilyProfile(
        FlexibleMeshReleaseFamily.MOE_TRAIN,
        "flexible_mesh_release_moe.FlexibleMoeReleaseAdapter",
        "static_top1_direct_xy_gate_ar_sgd/v3",
        tuple(sorted((
            "builder=build_round_robin_flexible_moe_spec",
            "gate_sync=row_major_binary_tree_reduce_broadcast_root0",
            "mode=train",
            "routing_shift=0_if_rank1_else_1",
        ))),
    ),
    FlexibleMeshReleaseFamilyProfile(
        FlexibleMeshReleaseFamily.MESHSLICE_AG,
        "flexible_mesh_release_meshslice.FlexibleMeshSliceReleaseAdapter",
        "dense_infer_ag_gemm/v5",
        tuple(sorted((
            "operation=ag_gemm",
            "runtime_problem=m4_per_row_n4_per_column_k_factor=(4_if_rows_times_columns_eq_1_else_2)",
            "peer_schedule=circle_method_1factorization_complementary_local_order",
            "workload=FlexibleMeshWorkloadSpec.dense_infer(mesh)",
        ))),
    ),
    FlexibleMeshReleaseFamilyProfile(
        FlexibleMeshReleaseFamily.MESHSLICE_RS_FALLBACK,
        "flexible_mesh_release_meshslice.FlexibleMeshSliceReleaseAdapter",
        "dense_infer_gemm_rs_fallback/v5",
        tuple(sorted((
            "operation=gemm_rs",
            "runtime_problem=m4_per_row_n4_per_column_k_factor=(4_if_rows_times_columns_eq_1_else_2)",
            "peer_schedule=circle_method_1factorization_complementary_local_order",
            "workload=FlexibleMeshWorkloadSpec.dense_infer(mesh)",
        ))),
    ),
    FlexibleMeshReleaseFamilyProfile(
        FlexibleMeshReleaseFamily.MESHSLICE_AR_FALLBACK,
        "flexible_mesh_release_meshslice.FlexibleMeshSliceReleaseAdapter",
        "dense_infer_gemm_ar_fallback/v5",
        tuple(sorted((
            "operation=gemm_ar",
            "runtime_problem=m4_per_row_n4_per_column_k_factor=(4_if_rows_times_columns_eq_1_else_2)",
            "peer_schedule=circle_method_1factorization_complementary_local_order",
            "workload=FlexibleMeshWorkloadSpec.dense_infer(mesh)",
        ))),
    ),
)

if tuple(profile.family for profile in _PROFILES) != tuple(FlexibleMeshReleaseFamily):
    raise RuntimeError("release family profiles must follow canonical family order")
for _index, _profile in enumerate(_PROFILES):
    _profile.validate(f"release_family_profiles[{_index}]")


def release_family_profile(
    family: FlexibleMeshReleaseFamily,
) -> FlexibleMeshReleaseFamilyProfile:
    if type(family) is not FlexibleMeshReleaseFamily:
        raise SchemaError("must be a release family", path="family")
    return _PROFILES[tuple(FlexibleMeshReleaseFamily).index(family)]


def release_trace_model_digest(family: FlexibleMeshReleaseFamily) -> str:
    return release_family_profile(family).digest


def release_trace_model_digests(
) -> tuple[tuple[FlexibleMeshReleaseFamily, str], ...]:
    return tuple((profile.family, profile.digest) for profile in _PROFILES)


__all__ = [
    "FlexibleMeshReleaseFamilyProfile",
    "release_family_profile",
    "release_trace_model_digest",
    "release_trace_model_digests",
]
