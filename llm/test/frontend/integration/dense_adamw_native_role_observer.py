"""Validate only real native ProbeExternal bytes for every AdamW role and version."""

from __future__ import annotations

import re

from llm.frontend.wafer_frontend.passes.dense_adamw_physical_authority import (
    PhysicalAdamwAuthority,
)


_PATTERN = re.compile(
    r"\[DENSE_ADAMW_EXTERNAL_ROLE_VALUE\] role=(\w+) version=(\d+) "
    r"bytes=(\d+) digest=([0-9a-f]{64}) state_count=(\d+) "
    r"pending=(\d+) functional=(\d+) pass=(\d+)"
)


def observe_native_adamw_role_values(
    stdout: str, source: PhysicalAdamwAuthority,
) -> dict[str, object]:
    rows = _PATTERN.findall(stdout)
    roles = {item.kind: item for item in source.roles}
    if len(rows) != 15 or len(roles) != 5 or not all(
        {row[0] for row in rows if row[1] == str(version)} == set(roles)
        for version in (0, 1, 2)
    ):
        raise RuntimeError("native AdamW 5-role ProbeExternal/version coverage incomplete")
    by_version = {}
    for role, version, byte_count, digest, state_count, pending, functional, passed in rows:
        expected = roles[role]
        if (
            int(byte_count) != expected.size_bytes
            or int(state_count) != expected.state_count
            or pending != "0" or functional != "0" or passed != "1"
            # The unchanged digest is a timing-only observation, never a
            # claim that AdamW m/v values were numerically updated.
            or digest != expected.seed_digest
        ):
            raise RuntimeError(f"native AdamW external role bytes differ: {role=} {version=}")
        by_version.setdefault(version, {})[role] = digest
    tokens = (
        "[DENSE_ADAMW_EXTERNAL_ROLE_VALUE] role=optimizer_master version=0",
        "[DENSE_TRAINING_SEQUENCE_STATE] version=0",
        "[DENSE_ADAMW_PAGED_DMA_EVENT] index=165 step=0",
        "[DENSE_ADAMW_EXTERNAL_ROLE_VALUE] role=optimizer_master version=1",
        "[DENSE_TRAINING_SEQUENCE_STATE] version=1",
        "[DENSE_ADAMW_SEQUENCE_STEP] index=0",
        "[DENSE_ADAMW_PAGED_DMA_EVENT] index=331 step=1",
        "[DENSE_ADAMW_EXTERNAL_ROLE_VALUE] role=optimizer_master version=2",
        "[DENSE_TRAINING_SEQUENCE_STATE] version=2",
        "[DENSE_ADAMW_SEQUENCE_STEP] index=1",
        "[DENSE_ADAMW_PAGED_DMA_DRAIN]",
        "[SIM_RESULT]",
    )
    positions = [stdout.find(token) for token in tokens]
    if -1 in positions or positions != sorted(positions):
        raise RuntimeError(f"external state restore/compute/dirty writeback ordering drifted: {positions}")
    return {
        "versions": (0, 1, 2),
        "roles": by_version,
        "role_bytes": {role: roles[role].size_bytes for role in sorted(roles)},
        "physical_state_count": len(source.states),
        "pending_requests": 0,
        "functional": False,
        "numerical_optimizer_correctness_verified": False,
    }


__all__ = ["observe_native_adamw_role_values"]
