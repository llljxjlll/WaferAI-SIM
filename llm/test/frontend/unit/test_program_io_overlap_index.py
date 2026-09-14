from __future__ import annotations

from dataclasses import dataclass
import random
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.program_io import (
    _reused_owned_root_ids,
)
from llm.frontend.wafer_frontend.schema.ir2 import BufferOwnership
from llm.frontend.wafer_frontend.schema.program_io import (
    _physical_allocation_overlap_pairs,
    _validate_allocation_nonoverlap,
)


@dataclass(frozen=True, slots=True)
class _Abi:
    id: str
    ownership: BufferOwnership
    region_ref: str
    region_offset_bytes: int
    size_bytes: int
    lifetime_start: int
    lifetime_end_exclusive: int


@dataclass(frozen=True, slots=True)
class _Root:
    abi: _Abi
    runtime_core_id: int


@dataclass(frozen=True, slots=True)
class _Allocation:
    abi: _Abi
    runtime_core_id: int
    region_symbol_ref: str
    absolute_start: int
    size_bytes: int


def _old_reused_owned_root_ids(roots: tuple[_Root, ...]) -> set[str]:
    return {
        item.abi.id
        for item in roots
        if item.abi.ownership is not BufferOwnership.BORROWED
        and any(
            other is not item
            and other.runtime_core_id == item.runtime_core_id
            and other.abi.region_ref == item.abi.region_ref
            and item.abi.region_offset_bytes
            < other.abi.region_offset_bytes + other.abi.size_bytes
            and other.abi.region_offset_bytes
            < item.abi.region_offset_bytes + item.abi.size_bytes
            and (
                other.abi.ownership is BufferOwnership.BORROWED
                or other.abi.lifetime_end_exclusive <= item.abi.lifetime_start
            )
            and (
                other.abi.lifetime_end_exclusive <= item.abi.lifetime_start
                or item.abi.lifetime_end_exclusive <= other.abi.lifetime_start
            )
            for other in roots
        )
    }


def _old_overlap_pairs(
    allocations: tuple[_Allocation, ...],
) -> set[frozenset[str]]:
    result: set[frozenset[str]] = set()
    for index, left in enumerate(allocations):
        for right in allocations[index + 1 :]:
            if (
                left.runtime_core_id == right.runtime_core_id
                and left.region_symbol_ref == right.region_symbol_ref
                and left.absolute_start < right.absolute_start + right.size_bytes
                and right.absolute_start < left.absolute_start + left.size_bytes
            ):
                result.add(frozenset((left.abi.id, right.abi.id)))
    return result


def _old_rejects(
    allocations: tuple[_Allocation, ...],
    producer_pass: str,
    core_stream_count: int,
) -> bool:
    reuse_authorized = (
        producer_pass in ("manifest_linker", "moe_swizzle_standard_linker")
        or (
            producer_pass == "unfused_comparison_standard_linker"
            and core_stream_count == 4
        )
    )
    for index, left in enumerate(allocations):
        for right in allocations[index + 1 :]:
            physical_overlap = (
                left.runtime_core_id == right.runtime_core_id
                and left.region_symbol_ref == right.region_symbol_ref
                and left.absolute_start < right.absolute_start + right.size_bytes
                and right.absolute_start < left.absolute_start + left.size_bytes
            )
            lifetime_disjoint = (
                left.abi.lifetime_end_exclusive <= right.abi.lifetime_start
                or right.abi.lifetime_end_exclusive <= left.abi.lifetime_start
            )
            if physical_overlap and not (reuse_authorized and lifetime_disjoint):
                return True
    return False


class ProgramIoOverlapIndexTest(unittest.TestCase):
    def _cases(self) -> tuple[tuple[_Root, ...], ...]:
        generator = random.Random(0x5EED)
        cases: list[tuple[_Root, ...]] = []
        for case_index in range(100):
            roots: list[_Root] = []
            for item_index in range(generator.randint(0, 35)):
                lifetime_start = generator.randint(0, 24)
                roots.append(_Root(
                    abi=_Abi(
                        id=f"abi_{case_index}_{item_index}",
                        ownership=generator.choice((
                            BufferOwnership.BORROWED,
                            BufferOwnership.OWNED,
                        )),
                        region_ref=f"region_{generator.randrange(3)}",
                        region_offset_bytes=generator.randrange(0, 64),
                        size_bytes=generator.randrange(1, 17),
                        lifetime_start=lifetime_start,
                        lifetime_end_exclusive=(
                            lifetime_start + generator.randrange(1, 10)
                        ),
                    ),
                    runtime_core_id=generator.randrange(4),
                ))
            generator.shuffle(roots)
            cases.append(tuple(roots))
        return tuple(cases)

    def test_reused_roots_match_old_ordered_all_pairs_oracle(self) -> None:
        touching = (
            _Root(_Abi("left", BufferOwnership.BORROWED, "r", 0, 8, 0, 4), 0),
            _Root(_Abi("right", BufferOwnership.OWNED, "r", 8, 8, 4, 8), 0),
        )
        self.assertEqual(_reused_owned_root_ids(touching), set())
        for roots in self._cases():
            self.assertEqual(
                _reused_owned_root_ids(roots),
                _old_reused_owned_root_ids(roots),
            )

    def test_allocation_pairs_and_validation_match_old_all_pairs_oracle(self) -> None:
        for roots in self._cases():
            allocations = tuple(
                _Allocation(
                    abi=root.abi,
                    runtime_core_id=root.runtime_core_id,
                    region_symbol_ref=root.abi.region_ref,
                    absolute_start=root.abi.region_offset_bytes,
                    size_bytes=root.abi.size_bytes,
                )
                for root in roots
            )
            actual_pairs = {
                frozenset((left.abi.id, right.abi.id))
                for left, right in _physical_allocation_overlap_pairs(allocations)
            }
            self.assertEqual(actual_pairs, _old_overlap_pairs(allocations))
            for producer_pass, core_stream_count in (
                ("manifest_linker", 1),
                ("moe_swizzle_standard_linker", 100),
                ("unfused_comparison_standard_linker", 4),
                ("unfused_comparison_standard_linker", 3),
                ("other_linker", 4),
            ):
                expected_rejection = _old_rejects(
                    allocations, producer_pass, core_stream_count,
                )
                try:
                    _validate_allocation_nonoverlap(
                        allocations, producer_pass, core_stream_count, "test",
                    )
                except SchemaError:
                    actual_rejection = True
                else:
                    actual_rejection = False
                self.assertEqual(actual_rejection, expected_rejection)


if __name__ == "__main__":
    unittest.main()
