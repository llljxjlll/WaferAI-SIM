from __future__ import annotations

from enum import Enum


class IntraDieOptimizationMode(str, Enum):
    """Product selection semantics for intra-die optimization."""

    OFF = "off"
    AUTO = "auto"
    FORCE = "force"


__all__ = ["IntraDieOptimizationMode"]
