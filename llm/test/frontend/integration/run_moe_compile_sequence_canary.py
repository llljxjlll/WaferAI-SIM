"""Standalone production compile canary for P3 MoE inference and training."""

from __future__ import annotations

from llm.frontend.wafer_frontend.passes.moe_compile_sequence import (
    compile_moe_sequence,
)
from llm.frontend.wafer_frontend.schema.flexible_moe import MoeRectActionKind
from llm.frontend.wafer_frontend.schema.workload_run import WorkloadFamily
from llm.test.frontend.unit.test_moe_compile_sequence import _manifest


def main() -> int:
    total_units = 0
    total_records = 0
    for family in (
        WorkloadFamily.MOE_INFERENCE,
        WorkloadFamily.MOE_TRAINING,
    ):
        sequence = compile_moe_sequence(_manifest(family))
        repeated = compile_moe_sequence(sequence.materialization)
        if repeated != sequence:
            raise RuntimeError(f"{family.value} compile sequence is not repeatable")
        if any(unit.runtime_verified for unit in sequence.units):
            raise RuntimeError(f"{family.value} compile sequence overclaims runtime")
        if family is WorkloadFamily.MOE_TRAINING and not all(
            any(
                action.kind is MoeRectActionKind.GATE_GRADIENT_ALL_REDUCE
                for action in unit.plan.actions
            )
            for unit in sequence.units
        ):
            raise RuntimeError("training units lack production gate sync")
        unit_records = sum(
            len(stream.records)
            for unit in sequence.units
            for fragment in unit.linked_manifest.fragments
            for stream in fragment.core_streams
        )
        if unit_records <= 0:
            raise RuntimeError(f"{family.value} produced an empty linked sequence")
        total_units += len(sequence.units)
        total_records += unit_records
        print(
            "MOE_COMPILE_SEQUENCE "
            f"family={family.value} units={len(sequence.units)} "
            f"records={unit_records} coverage={sequence.coverage.value} "
            f"runtime={sequence.runtime_status.value}"
        )
    print(
        "MOE_COMPILE_SEQUENCE_CANARY "
        f"status=PASS units={total_units} records={total_records}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
