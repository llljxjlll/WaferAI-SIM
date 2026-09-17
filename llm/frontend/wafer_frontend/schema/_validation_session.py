"""Private once-per-builder validation memo for frozen schema objects."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


_SESSION: ContextVar[dict[tuple[str, int], object] | None] = ContextVar(
    "wafer_frontend_validation_session", default=None
)


@contextmanager
def builder_validation_session() -> Iterator[None]:
    if _SESSION.get() is not None:
        yield
        return

    token = _SESSION.set({})
    try:
        yield
    finally:
        _SESSION.reset(token)


def validation_seen(value: object, domain: str) -> bool:
    session = _SESSION.get()
    return session is not None and session.get((domain, id(value))) is value


def mark_validation_complete(value: object, domain: str) -> None:
    session = _SESSION.get()
    if session is not None:
        session[(domain, id(value))] = value

def cached_validation_aux(owner: object, domain: str) -> object | None:
    session = _SESSION.get()
    if session is None:
        return None
    cached = session.get(("validation_aux:" + domain, id(owner)))
    if type(cached) is tuple and len(cached) == 2 and cached[0] is owner:
        return cached[1]
    return None


def cache_validation_aux(owner: object, domain: str, value: object) -> None:
    session = _SESSION.get()
    if session is not None:
        session[("validation_aux:" + domain, id(owner))] = (owner, value)


def cached_dataclass_primitive(value: object) -> object | None:
    session = _SESSION.get()
    if session is None:
        return None
    cached = session.get(("canonical_primitive", id(value)))
    if type(cached) is tuple and len(cached) == 2 and cached[0] is value:
        return cached[1]
    return None


def cache_dataclass_primitive(value: object, primitive: object) -> None:
    session = _SESSION.get()
    if session is not None:
        session[("canonical_primitive", id(value))] = (value, primitive)

__all__ = [
    "builder_validation_session",
    "cache_validation_aux",
    "cached_validation_aux",
    "cache_dataclass_primitive",
    "cached_dataclass_primitive",
    "mark_validation_complete",
    "validation_seen",
]
