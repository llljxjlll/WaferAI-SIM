#!/usr/bin/env python3
"""Deterministic mutation gate for the ISA-v1 program container/record parser."""

from __future__ import annotations

import argparse
import random
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


HEADER_SIZE = 64
DESCRIPTOR_SIZE = 40
SECTION_RECORD_STREAM = 6
WHOLE_CRC_OFFSET = 56
MAX_PROGRAM_BYTES = 64 * 1024 * 1024
FIXED_SEED = 0x4E505531


def fail(message: str) -> None:
    raise RuntimeError(f"[ISA MUTATION] FAIL: {message}")


def crc32c(payload: bytes) -> int:
    checksum = 0xFFFFFFFF
    for value in payload:
        checksum ^= value
        for _ in range(8):
            checksum = ((checksum >> 1) ^
                        (0x82F63B78 if checksum & 1 else 0))
    return (~checksum) & 0xFFFFFFFF


def u16(data: bytes | bytearray, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def u32(data: bytes | bytearray, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def u64(data: bytes | bytearray, offset: int) -> int:
    return struct.unpack_from("<Q", data, offset)[0]


def put16(data: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<H", data, offset, value)


def put32(data: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<I", data, offset, value)


def put64(data: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<Q", data, offset, value)


@dataclass(frozen=True)
class Descriptor:
    table_offset: int
    section_type: int
    offset: int
    size: int


def descriptors(data: bytes | bytearray) -> list[Descriptor]:
    if len(data) < HEADER_SIZE:
        fail("baseline artifact is shorter than its fixed header")
    count = u32(data, 32)
    table = u64(data, 40)
    if count != 7 or table != HEADER_SIZE:
        fail(f"baseline descriptor layout changed: count={count} table={table}")
    if table + count * DESCRIPTOR_SIZE > len(data):
        fail("baseline descriptor table is truncated")
    result: list[Descriptor] = []
    for index in range(count):
        entry = table + index * DESCRIPTOR_SIZE
        result.append(Descriptor(entry, u32(data, entry),
                                 u64(data, entry + 8),
                                 u64(data, entry + 16)))
    return result


def record_descriptor(data: bytes | bytearray) -> Descriptor:
    found = [entry for entry in descriptors(data)
             if entry.section_type == SECTION_RECORD_STREAM]
    if len(found) != 1:
        fail(f"baseline has {len(found)} external record sections")
    entry = found[0]
    if entry.size < 8 or entry.offset + entry.size > len(data):
        fail("baseline external record stream is invalid")
    return entry


def patch_whole_crc(data: bytearray) -> None:
    if len(data) < HEADER_SIZE:
        fail("cannot patch CRC of a truncated artifact")
    put32(data, WHOLE_CRC_OFFSET, 0)
    put32(data, WHOLE_CRC_OFFSET, crc32c(bytes(data)))


def patch_record_and_whole_crc(data: bytearray) -> None:
    entry = record_descriptor(data)
    payload = bytes(data[entry.offset:entry.offset + entry.size])
    put32(data, entry.table_offset + 32, crc32c(payload))
    patch_whole_crc(data)


def verify_baseline(data: bytes) -> None:
    if data[:8] != b"NPUPRG1\x00":
        fail("fixture magic changed")
    if len(data) != u64(data, 48):
        fail("fixture file_size does not match its bytes")
    if len(data) > MAX_PROGRAM_BYTES:
        fail("fixture exceeds the public 64 MiB limit")
    expected = u32(data, WHOLE_CRC_OFFSET)
    candidate = bytearray(data)
    put32(candidate, WHOLE_CRC_OFFSET, 0)
    if crc32c(bytes(candidate)) != expected:
        fail("fixture whole-file CRC is invalid")
    entry = record_descriptor(data)
    if crc32c(data[entry.offset:entry.offset + entry.size]) != \
            u32(data, entry.table_offset + 32):
        fail("fixture record-stream CRC is invalid")


Mutation = Callable[[bytearray], None]


def set_byte(offset: int, value: int) -> Mutation:
    def apply(data: bytearray) -> None:
        data[offset] = value
    return apply


def set16(offset: int, value: int) -> Mutation:
    def apply(data: bytearray) -> None:
        put16(data, offset, value)
    return apply


def set32(offset: int, value: int) -> Mutation:
    def apply(data: bytearray) -> None:
        put32(data, offset, value)
    return apply


def set64(offset: int, value: int) -> Mutation:
    def apply(data: bytearray) -> None:
        put64(data, offset, value)
    return apply


def flip_byte(offset: int, mask: int = 1) -> Mutation:
    def apply(data: bytearray) -> None:
        data[offset] ^= mask
    return apply


def mutate_descriptor(section_type: int, relative: int, value: int,
                      width: int, repair_whole: bool = True) -> Mutation:
    def apply(data: bytearray) -> None:
        selected = [entry for entry in descriptors(data)
                    if entry.section_type == section_type]
        if len(selected) != 1:
            fail(f"cannot find unique section {section_type}")
        offset = selected[0].table_offset + relative
        {2: put16, 4: put32, 8: put64}[width](data, offset, value)
        if repair_whole:
            patch_whole_crc(data)
    return apply


def mutate_first_record(relative: int, value: int, width: int) -> Mutation:
    def apply(data: bytearray) -> None:
        entry = record_descriptor(data)
        offset = entry.offset + relative
        {1: lambda d, o, v: d.__setitem__(o, v),
         2: put16, 4: put32}[width](data, offset, value)
        patch_record_and_whole_crc(data)
    return apply


def mutate_record_reserved(data: bytearray) -> None:
    entry = record_descriptor(data)
    # The reference fixture begins with SRAM_BIND. Its payload bytes [1,4)
    # are canonical reserved zeros.
    data[entry.offset + 8 + 1] = 1
    patch_record_and_whole_crc(data)


def append_canonical(data: bytearray) -> None:
    data.append(0xA5)
    put64(data, 48, len(data))
    patch_whole_crc(data)


def truncate_to(size: int) -> Mutation:
    def apply(data: bytearray) -> None:
        del data[size:]
    return apply


def mutation_matrix(baseline: bytes) -> list[tuple[str, Mutation, str | None]]:
    record = record_descriptor(baseline)
    table = u64(baseline, 40)
    cases: list[tuple[str, Mutation, str | None]] = [
        ("magic", flip_byte(0, 0x20), None),
        ("format-major", set16(8, 2), None),
        ("format-minor", set16(10, 1), None),
        ("isa-major", set16(12, 2), None),
        ("isa-minor", set16(14, 1), None),
        ("header-size", set16(16, 63), None),
        ("endianness", set_byte(18, 2), None),
        ("header-reserved", set_byte(19, 1), None),
        ("capability-bit", set64(24, 1 << 63), None),
        ("section-count-zero", set32(32, 0), None),
        ("section-count-max-plus-one", set32(32, 65), None),
        ("section-table-before-header", set64(40, 1), None),
        ("section-table-outside", set64(40, len(baseline) + 1), None),
        ("file-size-short", set64(48, len(baseline) - 1), None),
        ("file-size-long", set64(48, len(baseline) + 1), None),
        ("whole-crc", flip_byte(WHOLE_CRC_OFFSET), None),
        ("footer-reserved", set_byte(60, 1), None),
        ("record-section-flags", mutate_descriptor(6, 4, 0, 4), None),
        ("record-section-offset", mutate_descriptor(6, 8, 1, 8), None),
        ("record-section-size", mutate_descriptor(
            6, 16, record.size + len(baseline), 8), None),
        ("record-section-count", mutate_descriptor(6, 24, 0, 4), None),
        ("record-section-entry-size", mutate_descriptor(6, 28, 8, 4), None),
        ("record-section-crc", mutate_descriptor(
            6, 32, u32(baseline, record.table_offset + 32) ^ 1, 4), None),
        ("record-section-reserved", mutate_descriptor(6, 36, 1, 4), None),
        ("record-invalid-opcode", mutate_first_record(0, 0, 1),
         "opcode INVALID"),
        ("record-reserved-opcode", mutate_first_record(0, 0xFF, 1),
         "opcode"),
        ("record-version", mutate_first_record(1, 2, 1),
         "external record version"),
        ("record-flags", mutate_first_record(2, 1, 2),
         "record flags"),
        ("record-payload-size-plus-one", mutate_first_record(
            4, u32(baseline, record.offset + 4) + 1, 4), "payload_size"),
        ("record-payload-size-max", mutate_first_record(4, 0xFFFFFFFF, 4),
         "external record payload"),
        ("record-reserved-payload", mutate_record_reserved, "reserved"),
        ("trailing-byte", append_canonical, None),
    ]
    for size in sorted({0, 1, 7, 8, 31, 63, 64,
                        table + 1, len(baseline) // 2,
                        len(baseline) - 1}):
        cases.append((f"truncate-{size}", truncate_to(size), None))
    rng = random.Random(FIXED_SEED)
    candidates = list(range(HEADER_SIZE, len(baseline)))
    for index in range(min(16, len(candidates))):
        offset = candidates[rng.randrange(len(candidates))]
        bit = 1 << rng.randrange(8)
        cases.append((f"seeded-bitflip-{index:02d}-{offset}-{bit}",
                      flip_byte(offset, bit), None))
    names = [name for name, _, _ in cases]
    if len(names) != len(set(names)):
        fail("mutation names are not unique")
    return cases


def generate_fixture(program_fixture: Path, output: Path) -> None:
    proc = subprocess.run(
        [str(program_fixture), str(output)], cwd=output.parent, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        fail(f"fixture returned {proc.returncode}")
    if not output.is_file() or output.stat().st_size == 0:
        fail("fixture did not create a non-empty artifact")


def run_mutations(npusim: Path, program_fixture: Path,
                  runtime_root: Path | None) -> None:
    parent = runtime_root if runtime_root is not None else None
    with tempfile.TemporaryDirectory(prefix="isa-v1-mutation-", dir=parent) as tmp:
        root = Path(tmp)
        fixture_path = root / "baseline.npup"
        generate_fixture(program_fixture, fixture_path)
        baseline = fixture_path.read_bytes()
        verify_baseline(baseline)
        cases = mutation_matrix(baseline)
        for name, mutation, diagnostic in cases:
            candidate = bytearray(baseline)
            mutation(candidate)
            path = root / f"{name}.npup"
            path.write_bytes(candidate)
            proc = subprocess.run(
                [str(npusim), "--program", str(path)], cwd=root, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20,
            )
            if proc.returncode == 0:
                print(proc.stdout)
                fail(f"{name} was accepted")
            if "Loaded Program Format" in proc.stdout:
                print(proc.stdout)
                fail(f"{name} reached committed program state")
            if diagnostic is not None and diagnostic not in proc.stdout:
                print(proc.stdout)
                fail(f"{name} lacks diagnostic fragment {diagnostic!r}")
        print(f"[ISA MUTATION] PASS: {len(cases)} deterministic "
              f"container/record mutations seed=0x{FIXED_SEED:08x}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npusim", type=Path, required=True)
    parser.add_argument("--program-fixture", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path)
    args = parser.parse_args()
    try:
        run_mutations(args.npusim.resolve(), args.program_fixture.resolve(),
                      args.runtime_root.resolve() if args.runtime_root else None)
        return 0
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
