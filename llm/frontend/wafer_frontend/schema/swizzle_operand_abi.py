"""Exact typed operand ABI for executable Swizzle lowering."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import prod

from ..errors import SchemaError
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64
from .ir0 import FusionPattern
from .ir1 import IR1
from .swizzle import SwizzleActionKind, SwizzleAlgorithm
from .swizzle_abi import SwizzleCoreAddressABI
from .swizzle_ir2 import SwizzleIr2Projection
from .swizzle_lowering import validate_swizzle_plan_projection
from .swizzle_plan import SwizzleFusionPlan, SwizzleValueUse


SWIZZLE_OPERAND_ABI_SCHEMA_VERSION = "wafer_frontend.swizzle_operand_abi/v1alpha1"


def _dtype_bytes(dtype: DType) -> int:
    if dtype is DType.FP16:
        return 2
    if dtype in (DType.FP32, DType.INT32):
        return 4
    raise SchemaError("unsupported operand dtype", path="dtype")


class SwizzleDteDirection(str, Enum):
    REMOTE_SEND = "remote_send"
    REMOTE_RECV = "remote_recv"
    LOCAL_COPY = "local_copy"


@dataclass(frozen=True, slots=True)
class SwizzleTaskOperandView:
    task_ref: str
    ordinal: int
    use: SwizzleValueUse
    value_ref: str
    slot: int
    shape: tuple[int, ...]
    layout: str
    dtype: DType
    byte_offset: int
    byte_extent: int

    def validate(self, path: str) -> None:
        validate_nonempty(self.task_ref, f"{path}.task_ref")
        validate_uint64(self.ordinal, f"{path}.ordinal")
        if type(self.use) is not SwizzleValueUse:
            raise SchemaError("must be a SwizzleValueUse", path=f"{path}.use")
        validate_nonempty(self.value_ref, f"{path}.value_ref")
        validate_uint64(self.slot, f"{path}.slot")
        if not self.shape:
            raise SchemaError("shape must be nonempty", path=f"{path}.shape")
        for index, extent in enumerate(self.shape):
            validate_uint64(extent, f"{path}.shape[{index}]")
            if extent == 0:
                raise SchemaError("shape extent must be positive", path=f"{path}.shape[{index}]")
        validate_nonempty(self.layout, f"{path}.layout")
        if type(self.dtype) is not DType:
            raise SchemaError("must be a DType", path=f"{path}.dtype")
        validate_uint64(self.byte_offset, f"{path}.byte_offset")
        validate_uint64(self.byte_extent, f"{path}.byte_extent")
        if self.byte_extent != prod(self.shape) * _dtype_bytes(self.dtype):
            raise SchemaError("byte extent must exactly equal typed view", path=f"{path}.byte_extent")


@dataclass(frozen=True, slots=True)
class SwizzleMatmulContract:
    task_ref: str
    dtype: DType
    accumulation_dtype: DType
    m: int
    n: int
    k: int

    def validate(self, path: str) -> None:
        validate_nonempty(self.task_ref, f"{path}.task_ref")
        if type(self.dtype) is not DType or type(self.accumulation_dtype) is not DType:
            raise SchemaError("requires typed dtypes", path=path)
        for name in ("m", "n", "k"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
            if getattr(self, name) == 0:
                raise SchemaError("must be positive", path=f"{path}.{name}")


@dataclass(frozen=True, slots=True)
class SwizzleDteContract:
    task_ref: str
    direction: SwizzleDteDirection
    logical_bytes: int
    payload_bits: int
    hbm: bool
    source_operand_ordinal: int | None
    destination_operand_ordinal: int | None

    def validate(self, path: str) -> None:
        validate_nonempty(self.task_ref, f"{path}.task_ref")
        if type(self.direction) is not SwizzleDteDirection:
            raise SchemaError("must be a SwizzleDteDirection", path=f"{path}.direction")
        validate_uint64(self.logical_bytes, f"{path}.logical_bytes")
        validate_uint64(self.payload_bits, f"{path}.payload_bits")
        if self.logical_bytes == 0 or self.payload_bits != self.logical_bytes * 8:
            raise SchemaError("payload bits must exactly encode bytes", path=path)
        if type(self.hbm) is not bool or self.hbm:
            raise SchemaError("Swizzle V1 DTE is SRAM-only", path=f"{path}.hbm")
        for name in ("source_operand_ordinal", "destination_operand_ordinal"):
            value = getattr(self, name)
            if value is not None:
                validate_uint64(value, f"{path}.{name}")


@dataclass(frozen=True, slots=True)
class SwizzleReduceContract:
    task_ref: str
    input_count: int
    input_stride_bytes: int
    element_count: int
    input_dtype: DType
    accumulation_dtype: DType
    output_dtype: DType

    def validate(self, path: str) -> None:
        validate_nonempty(self.task_ref, f"{path}.task_ref")
        for name in ("input_count", "input_stride_bytes", "element_count"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
            if getattr(self, name) == 0:
                raise SchemaError("must be positive", path=f"{path}.{name}")
        if self.input_count != 2:
            raise SchemaError("Swizzle V1 reduction requires two inputs", path=f"{path}.input_count")
        for name in ("input_dtype", "accumulation_dtype", "output_dtype"):
            if type(getattr(self, name)) is not DType:
                raise SchemaError("must be a DType", path=f"{path}.{name}")


def _expected_semantics(
    ir1: IR1,
    plan: SwizzleFusionPlan,
    projection: SwizzleIr2Projection,
) -> tuple[
    tuple[SwizzleTaskOperandView, ...],
    tuple[SwizzleMatmulContract, ...],
    tuple[SwizzleDteContract, ...],
    tuple[SwizzleReduceContract, ...],
]:
    ir1_values = {value.id: value for value in ir1.values}
    projected_values = {
        value.id: value for dag in projection.rank_dags for value in dag.values
    }
    actions = {
        action.source_action.id: action
        for program in plan.rank_programs
        for action in program.actions
    }
    operands = []
    matmuls = []
    dtes = []
    reductions = []
    meshslice = plan.algorithm is SwizzleAlgorithm.MESHSLICE_2D_OS
    meshslice_views = {}
    meshslice_tile = None
    if meshslice:
        problem = plan.decision.problem
        rows = len(plan.candidate.topology_witness.row_orders)
        columns = len(plan.candidate.topology_witness.column_orders)
        chunks = plan.candidate.chunk_count
        meshslice_tile = (
            problem.gemm.m // rows,
            problem.gemm.n // columns,
            problem.gemm.k // chunks,
        )
        tile_m, tile_n, tile_k = meshslice_tile
        lhs_source = ir1_values[problem.gemm.lhs.value_ref]
        rhs_source = ir1_values[problem.gemm.rhs.value_ref]
        output_source = ir1_values[problem.gemm.output.value_ref]
        meshslice_views = {
            "lhs": ((tile_m, tile_k), lhs_source.logical_layout, problem.gemm.dtype),
            "rhs": ((tile_k, tile_n), rhs_source.logical_layout, problem.gemm.dtype),
            "output": (
                (tile_m, tile_n),
                output_source.logical_layout,
                problem.gemm.accumulation_dtype,
            ),
        }
    for dag in projection.rank_dags:
        for task in dag.tasks:
            action = actions[task.source_action_ref]
            slot_by_buffer = {use.buffer_ref: use.slot for use in task.buffer_uses}
            refs = task.read_value_refs + task.write_value_refs
            uses = (SwizzleValueUse.READ,) * len(task.read_value_refs) + (
                (SwizzleValueUse.WRITE,) * len(task.write_value_refs)
            )
            if meshslice and refs:
                typed = []
                for ref in refs:
                    value = projected_values[ref]
                    if value.buffer_ref is None:
                        raise SchemaError(
                            "MeshSlice operand must bind an explicit buffer",
                            path="projection.rank_dags",
                        )
                    role = value.buffer_ref.rsplit(".", 1)[-1]
                    view = meshslice_views.get(role)
                    if view is None:
                        raise SchemaError(
                            "unsupported MeshSlice buffer role",
                            path="projection.rank_dags",
                        )
                    typed.append(view)
                shapes = tuple(item[0] for item in typed)
                layouts = tuple(item[1] for item in typed)
                dtypes = tuple(item[2] for item in typed)
                if task.kind is SwizzleActionKind.COMP:
                    assert meshslice_tile is not None
                    rank_m, rank_n, rank_k = meshslice_tile
                    if task.flops != 2 * rank_m * rank_n * rank_k:
                        raise SchemaError(
                            "MeshSlice GEMM FLOPs disagree with tile",
                            path="projection.rank_dags",
                        )
                    if action.compute_origin is None:
                        raise SchemaError(
                            "COMP lacks typed compute origin",
                            path="plan.rank_programs",
                        )
                    matmuls.append(SwizzleMatmulContract(
                        task.id, problem.gemm.dtype, problem.gemm.accumulation_dtype,
                        rank_m, rank_n, rank_k,
                    )
                )
            elif task.kind is SwizzleActionKind.COMP:
                origin = action.compute_origin
                if origin is None:
                    raise SchemaError(
                        "COMP lacks typed compute origin",
                        path="plan.rank_programs",
                    )
                rank_m, rank_n, rank_k = origin.workload.rank_shape
                if action.chunk_origin is not None:
                    if len(action.chunk_origin.logical_shape) != 2:
                        raise SchemaError("V1 GEMM chunk must be rank two", path="plan.rank_programs")
                    chunk0, chunk1 = action.chunk_origin.logical_shape
                    if plan.pattern is FusionPattern.AG_GEMM:
                        rank_m, rank_k = chunk0, chunk1
                    else:
                        rank_m, rank_n = chunk0, chunk1
                if task.flops != 2 * rank_m * rank_n * rank_k:
                    raise SchemaError("GEMM FLOPs disagree with chunk witness", path="projection.rank_dags")
                shapes = ((rank_m, rank_k), (rank_k, rank_n), (rank_m, rank_n))
                source_values = tuple(
                    ir1_values[ref]
                    for ref in origin.ir1_input_refs[:2] + origin.ir1_output_refs[:1]
                )
                layouts = tuple(source.logical_layout for source in source_values)
                dtypes = tuple(source.dtype for source in source_values)
                matmuls.append(
                    SwizzleMatmulContract(
                        task.id,
                        origin.workload.dtype,
                        origin.math.accumulation_dtype,
                        rank_m,
                        rank_n,
                        rank_k,
                    )
                )
            elif refs:
                if action.chunk_origin is None:
                    raise SchemaError("non-COMP operand lacks chunk witness", path="plan.rank_programs")
                source = ir1_values[action.chunk_origin.source_value_ref]
                shapes = (action.chunk_origin.logical_shape,) * len(refs)
                source_values = (source,) * len(refs)
                layouts = tuple(source.logical_layout for source in source_values)
                dtypes = tuple(source.dtype for source in source_values)
            else:
                shapes = ()
                layouts = ()
                dtypes = ()
            for ordinal, (use, ref, shape, layout, dtype) in enumerate(
                zip(uses, refs, shapes, layouts, dtypes, strict=True)
            ):
                value = projected_values[ref]
                slot = slot_by_buffer.get(value.buffer_ref, 0)
                operands.append(
                    SwizzleTaskOperandView(
                        task.id,
                        ordinal,
                        use,
                        ref,
                        slot,
                        shape,
                        layout,
                        dtype,
                        0,
                        prod(shape) * _dtype_bytes(dtype),
                    )
                )
            if task.kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV, SwizzleActionKind.LOCAL_COPY):
                extent = operands[-1].byte_extent if task.kind is SwizzleActionKind.RECV else operands[-len(refs)].byte_extent
                if task.kind is SwizzleActionKind.SEND:
                    direction, source_ordinal, destination_ordinal = SwizzleDteDirection.REMOTE_SEND, 0, None
                elif task.kind is SwizzleActionKind.RECV:
                    direction, source_ordinal, destination_ordinal = SwizzleDteDirection.REMOTE_RECV, None, 0
                else:
                    direction, source_ordinal, destination_ordinal = SwizzleDteDirection.LOCAL_COPY, 0, 1
                if task.kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV) and (
                    task.logical_bytes > extent
                    if meshslice
                    else task.logical_bytes != extent
                ):
                    raise SchemaError("DTE payload disagrees with typed view", path="projection.rank_dags")
                logical_bytes = task.logical_bytes if meshslice else extent
                dtes.append(SwizzleDteContract(
                    task.id, direction, logical_bytes, logical_bytes * 8, False,
                    source_ordinal, destination_ordinal,
                ))
            if task.kind is SwizzleActionKind.REDUCE:
                if len(refs) != 3:
                    raise SchemaError("REDUCE requires two reads and one write", path="projection.rank_dags")
                origin = action.reduction_origin
                if origin is None:
                    raise SchemaError(
                        "REDUCE lacks typed reduction origin",
                        path="plan.rank_programs",
                    )
                extent = operands[-3].byte_extent
                reductions.append(
                    SwizzleReduceContract(
                        task.id,
                        2,
                        extent,
                        prod(operands[-3].shape),
                        operands[-3].dtype,
                        origin.math.accumulation_dtype,
                        operands[-1].dtype,
                    )
                )
    return tuple(operands), tuple(matmuls), tuple(dtes), tuple(reductions)


@dataclass(frozen=True, slots=True)
class SwizzleOperandABI:
    schema_version: str
    producer_pass: str
    id: str
    source_ir1_id: str
    source_plan_ref: str
    source_projection_ref: str
    operands: tuple[SwizzleTaskOperandView, ...]
    matmul_contracts: tuple[SwizzleMatmulContract, ...]
    dte_contracts: tuple[SwizzleDteContract, ...]
    reduce_contracts: tuple[SwizzleReduceContract, ...]

    @classmethod
    def create(cls, *, producer_pass: str, **semantic: object) -> "SwizzleOperandABI":
        result = cls(
            SWIZZLE_OPERAND_ABI_SCHEMA_VERSION,
            producer_pass,
            stable_artifact_id("swizzle_operand_abi", semantic, schema_version=SWIZZLE_OPERAND_ABI_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in (
            "source_ir1_id", "source_plan_ref", "source_projection_ref",
            "operands", "matmul_contracts", "dte_contracts", "reduce_contracts",
        )}

    def validate(self, path: str = "swizzle_operand_abi") -> None:
        if self.schema_version != SWIZZLE_OPERAND_ABI_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        for name in ("producer_pass", "source_ir1_id", "source_plan_ref", "source_projection_ref"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        for name in ("operands", "matmul_contracts", "dte_contracts", "reduce_contracts"):
            values = getattr(self, name)
            for index, value in enumerate(values):
                value.validate(f"{path}.{name}[{index}]")
        operand_keys = tuple((item.task_ref, item.ordinal) for item in self.operands)
        if len(operand_keys) != len(set(operand_keys)):
            raise SchemaError("duplicate task operand", path=f"{path}.operands")
        for name in ("matmul_contracts", "dte_contracts", "reduce_contracts"):
            refs = tuple(item.task_ref for item in getattr(self, name))
            if len(refs) != len(set(refs)):
                raise SchemaError("duplicate task contract", path=f"{path}.{name}")
        expected = stable_artifact_id("swizzle_operand_abi", self._semantic_key(), schema_version=SWIZZLE_OPERAND_ABI_SCHEMA_VERSION)
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")

    def validate_against(
        self,
        ir1: IR1,
        plan: SwizzleFusionPlan,
        projection: SwizzleIr2Projection,
        core_abi: SwizzleCoreAddressABI,
        path: str = "swizzle_operand_abi",
    ) -> None:
        self.validate(path)
        validate_swizzle_plan_projection(plan, projection, f"{path}.inputs")
        core_abi.validate_against(ir1, plan, projection, f"{path}.core_abi")
        if (self.source_ir1_id, self.source_plan_ref, self.source_projection_ref) != (ir1.id, plan.id, projection.id):
            raise SchemaError("operand ABI provenance is not exact", path=path)
        expected = _expected_semantics(ir1, plan, projection)
        actual = (self.operands, self.matmul_contracts, self.dte_contracts, self.reduce_contracts)
        if actual != expected:
            raise SchemaError("operand ABI disagrees with W7 origins/chunk witnesses", path=path)
        addresses = {(item.value_ref, item.slot): item for item in core_abi.value_bindings}
        for index, operand in enumerate(self.operands):
            binding = addresses[(operand.value_ref, operand.slot)]
            if operand.byte_offset > binding.size_bytes or operand.byte_extent > binding.size_bytes - operand.byte_offset:
                raise SchemaError("typed operand view exceeds address span", path=f"{path}.operands[{index}]")
        by_task = {}
        for operand in self.operands:
            by_task.setdefault(operand.task_ref, []).append(operand)
        for contract in self.reduce_contracts:
            views = by_task[contract.task_ref]
            first, accumulator, output = (addresses[(item.value_ref, item.slot)] for item in views)
            if first.address + contract.input_stride_bytes != accumulator.address:
                raise SchemaError("REDUCE inputs must be caller-allocated in typed order with exact stride", path=f"{path}.reduce_contracts")
            if (output.address, output.size_bytes) != (accumulator.address, accumulator.size_bytes):
                raise SchemaError("REDUCE output must alias input1 accumulator", path=f"{path}.reduce_contracts")


def build_swizzle_operand_abi(
    ir1: IR1,
    plan: SwizzleFusionPlan,
    projection: SwizzleIr2Projection,
) -> SwizzleOperandABI:
    ir1.validate("ir1")
    validate_swizzle_plan_projection(plan, projection)
    operands, matmuls, dtes, reductions = _expected_semantics(ir1, plan, projection)
    return SwizzleOperandABI.create(
        producer_pass="swizzle_operand_abi_builder",
        source_ir1_id=ir1.id,
        source_plan_ref=plan.id,
        source_projection_ref=projection.id,
        operands=operands,
        matmul_contracts=matmuls,
        dte_contracts=dtes,
        reduce_contracts=reductions,
    )


__all__ = [name for name in globals() if name.startswith("Swizzle") or name.startswith("SWIZZLE_") or name == "build_swizzle_operand_abi"]
