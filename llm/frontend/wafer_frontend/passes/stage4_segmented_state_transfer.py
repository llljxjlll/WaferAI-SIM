"""Canonical segmented KV transfer producer for heterogeneous-TP Stage 4 PD."""

from __future__ import annotations

from ..errors import SchemaError, UnsupportedFeatureError
from ..schema.n4 import Stage4InterDiePlannedIR1
from ..schema.stage4_pd import Stage4KvReshardKind, Stage4PdMode
from ..schema.state_transfer import SegmentedKvStateTransferContract
from .stage4_state_transfer import (
    build_stage4_state_transfers,
    validate_stage4_state_transfers,
)


_PRODUCER_PASS = "stage4_segmented_state_transfer"


def _validate_source(source: Stage4InterDiePlannedIR1) -> None:
    if type(source) is not Stage4InterDiePlannedIR1:
        raise SchemaError(
            "must be a Stage4InterDiePlannedIR1",
            path="source",
        )
    source.validate("source")
    if source.pd_plan.mode is not Stage4PdMode.SEPARATED:
        raise UnsupportedFeatureError(
            "segmented KV transfers require separated PD",
            path="source.pd_plan.mode",
        )
    if source.pd_plan.reshard not in (
        Stage4KvReshardKind.GATHER,
        Stage4KvReshardKind.SCATTER,
    ):
        raise UnsupportedFeatureError(
            "segmented KV transfers require gather or scatter reshard",
            path="source.pd_plan.reshard",
        )


def _derive(
    source: Stage4InterDiePlannedIR1,
) -> tuple[SegmentedKvStateTransferContract, ...]:
    _validate_source(source)
    logical_slices = build_stage4_state_transfers(
        source.graph,
        source.pd_plan,
    )
    validate_stage4_state_transfers(
        logical_slices,
        source.graph,
        source.pd_plan,
    )
    return tuple(
        SegmentedKvStateTransferContract.from_logical_slice(
            producer_pass=_PRODUCER_PASS,
            logical_slice=logical_slice,
        )
        for logical_slice in logical_slices
    )


def validate_stage4_segmented_state_transfers(
    contracts: tuple[SegmentedKvStateTransferContract, ...],
    source: Stage4InterDiePlannedIR1,
    *,
    path: str = "stage4_segmented_state_transfers",
) -> None:
    """Independently rebuild the exact PDR contract and segment order."""

    _validate_source(source)
    if type(contracts) is not tuple:
        raise SchemaError("must be an immutable tuple", path=path)
    for index, contract in enumerate(contracts):
        contract_path = f"{path}[{index}]"
        if type(contract) is not SegmentedKvStateTransferContract:
            raise SchemaError(
                "must be a SegmentedKvStateTransferContract",
                path=contract_path,
            )
        if contract.producer_pass != _PRODUCER_PASS:
            raise SchemaError(
                f"must be {_PRODUCER_PASS!r}",
                path=f"{contract_path}.producer_pass",
            )
        contract.validate_against(source.graph, contract_path)
    expected = _derive(source)
    if contracts != expected:
        raise SchemaError(
            "must exactly equal the planned segmented KV transfer set",
            path=path,
        )
    logical_slices = tuple(contract.logical_slice for contract in contracts)
    validate_stage4_state_transfers(
        logical_slices,
        source.graph,
        source.pd_plan,
        path=f"{path}.logical_slices",
    )
    if len({contract.id for contract in contracts}) != len(contracts):
        raise SchemaError("contract ids must be unique", path=path)
    if sum(contract.bytes for contract in contracts) != sum(
        handoff.logical_unique_bytes for handoff in source.pd_plan.handoffs
    ):
        raise SchemaError(
            "total bytes must equal the plan logical KV handoff bytes",
            path=path,
        )


def build_stage4_segmented_state_transfers(
    source: Stage4InterDiePlannedIR1,
) -> tuple[SegmentedKvStateTransferContract, ...]:
    """Build canonical per-token contiguous segments for PDR gather/scatter."""

    result = _derive(source)
    validate_stage4_segmented_state_transfers(result, source)
    return result


__all__ = [
    "build_stage4_segmented_state_transfers",
    "validate_stage4_segmented_state_transfers",
]
