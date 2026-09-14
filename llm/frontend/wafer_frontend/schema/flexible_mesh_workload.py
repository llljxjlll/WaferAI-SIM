"""V2 workload envelope for all rectangular Mesh workload families."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import DType
from .flexible_mesh_capacity import FlexibleMeshCapacityProfile
from .flexible_mesh_groups import FlexibleMeshAxisMapping
from .flexible_moe import FlexibleMoeMode, FlexibleMoeSpec
from .rect_mesh import RectMeshSpec
from .serde import canonical_digest


FLEXIBLE_MESH_WORKLOAD_SCHEMA_VERSION = (
    "wafer_frontend.flexible_mesh_workload/v1alpha1"
)


class FlexibleMeshWorkloadKind(str, Enum):
    DENSE_INFER = "dense_infer"
    DENSE_TRAIN = "dense_train"
    MOE_INFER = "moe_infer"
    MOE_TRAIN = "moe_train"


class FlexibleMeshOptimizer(str, Enum):
    SGD = "sgd"


class FlexibleMeshSliceOperation(str, Enum):
    AG_GEMM = "ag_gemm"
    GEMM_RS = "gemm_rs"
    GEMM_AR = "gemm_ar"


class FlexibleMeshSliceMode(str, Enum):
    FULL_2D = "full_2d"
    ROW_ONLY = "row_only"
    COLUMN_ONLY = "column_only"
    LOCAL = "local"


@dataclass(frozen=True, slots=True)
class FlexibleTrainSpec:
    micro_batch_count: int = 1
    optimizer: FlexibleMeshOptimizer = FlexibleMeshOptimizer.SGD
    gradient_accumulation_dtype: DType = DType.FP32
    parameter_dtype: DType = DType.FP16
    pp_degree: int = 1
    backward: bool = True

    def validate(self, path: str = "training") -> None:
        if type(self.micro_batch_count) is not int or self.micro_batch_count != 1:
            raise SchemaError(
                "v1 requires one microbatch", path=f"{path}.micro_batch_count"
            )
        if self.optimizer is not FlexibleMeshOptimizer.SGD:
            raise SchemaError("v1 requires SGD", path=f"{path}.optimizer")
        if self.gradient_accumulation_dtype is not DType.FP32:
            raise SchemaError(
                "v1 requires FP32 gradient accumulation",
                path=f"{path}.gradient_accumulation_dtype",
            )
        if self.parameter_dtype is not DType.FP16:
            raise SchemaError("v1 requires FP16 parameters", path=f"{path}.parameter_dtype")
        if type(self.pp_degree) is not int or self.pp_degree != 1:
            raise SchemaError("v1 requires PP=1", path=f"{path}.pp_degree")
        if self.backward is not True:
            raise SchemaError("training requires backward", path=f"{path}.backward")


@dataclass(frozen=True, slots=True)
class FlexibleMeshSliceSpec:
    operations: tuple[FlexibleMeshSliceOperation, ...] = (
        FlexibleMeshSliceOperation.AG_GEMM,
        FlexibleMeshSliceOperation.GEMM_RS,
        FlexibleMeshSliceOperation.GEMM_AR,
    )
    allow_fallback: bool = True

    def validate(self, path: str = "meshslice") -> None:
        if (
            type(self.operations) is not tuple
            or not self.operations
            or any(type(op) is not FlexibleMeshSliceOperation for op in self.operations)
            or len(set(self.operations)) != len(self.operations)
        ):
            raise SchemaError(
                "operations must be a unique non-empty tuple",
                path=f"{path}.operations",
            )
        canonical = tuple(
            op for op in FlexibleMeshSliceOperation if op in self.operations
        )
        if self.operations != canonical:
            raise SchemaError("operations must be canonical", path=f"{path}.operations")
        if type(self.allow_fallback) is not bool:
            raise SchemaError("must be a bool", path=f"{path}.allow_fallback")

    @staticmethod
    def mode_for(mesh: RectMeshSpec) -> FlexibleMeshSliceMode:
        mesh.validate()
        if mesh.rows == 1 and mesh.columns == 1:
            return FlexibleMeshSliceMode.LOCAL
        if mesh.rows == 1:
            return FlexibleMeshSliceMode.ROW_ONLY
        if mesh.columns == 1:
            return FlexibleMeshSliceMode.COLUMN_ONLY
        return FlexibleMeshSliceMode.FULL_2D


@dataclass(frozen=True, slots=True)
class FlexibleMeshWorkloadSpec:
    mesh: RectMeshSpec
    workload_kind: FlexibleMeshWorkloadKind
    axis_mapping: FlexibleMeshAxisMapping
    training: FlexibleTrainSpec | None = None
    moe: FlexibleMoeSpec | None = None
    meshslice: FlexibleMeshSliceSpec = FlexibleMeshSliceSpec()
    capacity: FlexibleMeshCapacityProfile = FlexibleMeshCapacityProfile()
    schema_version: str = FLEXIBLE_MESH_WORKLOAD_SCHEMA_VERSION

    @classmethod
    def dense_infer(
        cls, mesh: RectMeshSpec, *, transposed: bool = False
    ) -> "FlexibleMeshWorkloadSpec":
        return cls(
            mesh=mesh,
            workload_kind=FlexibleMeshWorkloadKind.DENSE_INFER,
            axis_mapping=FlexibleMeshAxisMapping.dense(mesh, transposed=transposed),
        )

    @classmethod
    def dense_train(
        cls, mesh: RectMeshSpec, *, transposed: bool = False
    ) -> "FlexibleMeshWorkloadSpec":
        return cls(
            mesh=mesh,
            workload_kind=FlexibleMeshWorkloadKind.DENSE_TRAIN,
            axis_mapping=FlexibleMeshAxisMapping.dense(mesh, transposed=transposed),
            training=FlexibleTrainSpec(),
        )

    @classmethod
    def moe_infer(cls, moe: FlexibleMoeSpec) -> "FlexibleMeshWorkloadSpec":
        if type(moe) is not FlexibleMoeSpec or moe.mode is not FlexibleMoeMode.INFERENCE:
            raise SchemaError("requires an inference FlexibleMoeSpec", path="moe")
        return cls(
            mesh=moe.mesh,
            workload_kind=FlexibleMeshWorkloadKind.MOE_INFER,
            axis_mapping=FlexibleMeshAxisMapping.moe(moe.mesh),
            moe=moe,
        )

    @classmethod
    def moe_train(cls, moe: FlexibleMoeSpec) -> "FlexibleMeshWorkloadSpec":
        if type(moe) is not FlexibleMoeSpec or moe.mode is not FlexibleMoeMode.TRAIN:
            raise SchemaError("requires a train FlexibleMoeSpec", path="moe")
        return cls(
            mesh=moe.mesh,
            workload_kind=FlexibleMeshWorkloadKind.MOE_TRAIN,
            axis_mapping=FlexibleMeshAxisMapping.moe(moe.mesh),
            training=FlexibleTrainSpec(),
            moe=moe,
        )

    @property
    def meshslice_mode(self) -> FlexibleMeshSliceMode:
        return self.meshslice.mode_for(self.mesh)

    @property
    def digest(self) -> str:
        self.validate()
        return canonical_digest(self)

    def validate(self, path: str = "flexible_mesh_workload") -> None:
        if self.schema_version != FLEXIBLE_MESH_WORKLOAD_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if type(self.mesh) is not RectMeshSpec:
            raise SchemaError("must be a RectMeshSpec", path=f"{path}.mesh")
        self.mesh.validate(f"{path}.mesh")
        if type(self.workload_kind) is not FlexibleMeshWorkloadKind:
            raise SchemaError("must be a workload kind", path=f"{path}.workload_kind")
        if type(self.axis_mapping) is not FlexibleMeshAxisMapping:
            raise SchemaError("must be an axis mapping", path=f"{path}.axis_mapping")
        dense = self.workload_kind in (
            FlexibleMeshWorkloadKind.DENSE_INFER,
            FlexibleMeshWorkloadKind.DENSE_TRAIN,
        )
        self.axis_mapping.validate(
            self.mesh,
            require_dense=dense,
            require_moe=not dense,
            path=f"{path}.axis_mapping",
        )
        is_train = self.workload_kind in (
            FlexibleMeshWorkloadKind.DENSE_TRAIN,
            FlexibleMeshWorkloadKind.MOE_TRAIN,
        )
        if is_train:
            if type(self.training) is not FlexibleTrainSpec:
                raise SchemaError("training spec is required", path=f"{path}.training")
            self.training.validate(f"{path}.training")
        elif self.training is not None:
            raise SchemaError("must be absent for inference", path=f"{path}.training")
        if dense:
            if self.moe is not None:
                raise SchemaError("must be absent for Dense", path=f"{path}.moe")
        else:
            if type(self.moe) is not FlexibleMoeSpec:
                raise SchemaError("MoE spec is required", path=f"{path}.moe")
            self.moe.validate_against_mesh(self.mesh, f"{path}.moe")
            expected_mode = (
                FlexibleMoeMode.TRAIN
                if self.workload_kind is FlexibleMeshWorkloadKind.MOE_TRAIN
                else FlexibleMoeMode.INFERENCE
            )
            if self.moe.mode is not expected_mode:
                raise SchemaError(
                    "MoE mode disagrees with workload kind", path=f"{path}.moe.mode"
                )
        if type(self.meshslice) is not FlexibleMeshSliceSpec:
            raise SchemaError("must be a MeshSlice spec", path=f"{path}.meshslice")
        self.meshslice.validate(f"{path}.meshslice")
        if type(self.capacity) is not FlexibleMeshCapacityProfile:
            raise SchemaError("must be a capacity profile", path=f"{path}.capacity")
        self.capacity.validate(f"{path}.capacity")


__all__ = [
    "FLEXIBLE_MESH_WORKLOAD_SCHEMA_VERSION",
    "FlexibleMeshOptimizer",
    "FlexibleMeshSliceMode",
    "FlexibleMeshSliceOperation",
    "FlexibleMeshSliceSpec",
    "FlexibleMeshWorkloadKind",
    "FlexibleMeshWorkloadSpec",
    "FlexibleTrainSpec",
]
