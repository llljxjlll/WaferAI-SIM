"""Strict dataclass decoding and deterministic JSON encoding."""

from __future__ import annotations

import hashlib
import json
import math
import types
from collections.abc import Mapping
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import TypeVar, Union, get_args, get_origin, get_type_hints

from ..errors import FrontendError, SchemaError


T = TypeVar("T")


class _JSONObject:
    """Lossless object pairs used only while checking strict JSON input."""

    __slots__ = ("pairs",)

    def __init__(self, pairs: list[tuple[str, object]]) -> None:
        self.pairs = pairs


class _NonFiniteJSONConstant:
    __slots__ = ("token",)

    def __init__(self, token: str) -> None:
        self.token = token


def _child_path(path: str, child: str) -> str:
    return f"{path}.{child}" if path else child


def to_primitive(value: object, *, path: str = "$") -> object:
    """Convert supported immutable schema values to JSON primitives."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_primitive(
                getattr(value, field.name), path=_child_path(path, field.name)
            )
            for field in fields(value)
        }
    if isinstance(value, tuple) or isinstance(value, list):
        return [
            to_primitive(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaError("JSON object keys must be strings", path=path)
            result[key] = to_primitive(item, path=_child_path(path, key))
        return result
    if type(value) is float and not math.isfinite(value):
        raise SchemaError("non-finite floats are not valid JSON", path=path)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise SchemaError(
        f"unsupported value type {type(value).__name__!r}", path=path
    )


def canonical_json(value: object) -> str:
    try:
        primitive = to_primitive(value)
    except RecursionError as error:
        raise SchemaError("value nesting is too deep or recursive", path="$") from error
    try:
        return json.dumps(
            primitive,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (ValueError, OverflowError) as error:
        raise SchemaError(str(error), path="$") from error


def canonical_json_bytes(value: object) -> bytes:
    return canonical_json(value).encode("utf-8")


def canonical_digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _materialize_strict_json(value: object, path: str) -> object:
    if isinstance(value, _NonFiniteJSONConstant):
        raise SchemaError(
            f"non-finite JSON constant {value.token!r} is not allowed",
            path=path,
        )
    if isinstance(value, _JSONObject):
        result: dict[str, object] = {}
        for key, child in value.pairs:
            child_path = _child_path(path, key)
            if key in result:
                raise SchemaError(f"duplicate object key {key!r}", path=child_path)
            result[key] = _materialize_strict_json(child, child_path)
        return result
    if isinstance(value, list):
        return [
            _materialize_strict_json(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    return value


def _decode_union(expected_type: object, value: object, path: str) -> object:
    alternatives = get_args(expected_type)
    if value is None and type(None) in alternatives:
        return None
    non_null_alternatives = tuple(
        alternative for alternative in alternatives if alternative is not type(None)
    )
    if len(non_null_alternatives) == 1:
        return from_data(non_null_alternatives[0], value, path=path)
    failures: list[FrontendError] = []
    for alternative in alternatives:
        if alternative is type(None):
            continue
        try:
            return from_data(alternative, value, path=path)
        except FrontendError as error:
            failures.append(error)
    if failures:
        raise SchemaError(
            f"value does not match any allowed type: {failures[0].message}", path=path
        )
    raise SchemaError("value does not match any allowed type", path=path)


def from_data(expected_type: type[T] | object, value: object, *, path: str = "$") -> T:
    """Decode one value, rejecting unknown fields and implicit coercions."""

    origin = get_origin(expected_type)
    if origin in (Union, types.UnionType):
        return _decode_union(expected_type, value, path)  # type: ignore[return-value]

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise SchemaError("expected an array", path=path)
        arguments = get_args(expected_type)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            item_type = arguments[0]
            return tuple(
                from_data(item_type, item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            )  # type: ignore[return-value]
        if len(arguments) != len(value):
            raise SchemaError(
                f"expected {len(arguments)} array entries, got {len(value)}", path=path
            )
        return tuple(
            from_data(item_type, item, path=f"{path}[{index}]")
            for index, (item_type, item) in enumerate(zip(arguments, value))
        )  # type: ignore[return-value]

    if expected_type is object:
        return value  # type: ignore[return-value]
    if expected_type is type(None):
        if value is not None:
            raise SchemaError("expected null", path=path)
        return None  # type: ignore[return-value]

    if isinstance(expected_type, type) and issubclass(expected_type, Enum):
        enum_values = tuple(item.value for item in expected_type)
        if enum_values and all(type(item) is int for item in enum_values):
            if type(value) is not int:
                raise SchemaError("expected an enum integer", path=path)
        elif type(value) is not str:
            raise SchemaError("expected an enum string", path=path)
        try:
            return expected_type(value)  # type: ignore[return-value]
        except ValueError as error:
            allowed = ", ".join(repr(item.value) for item in expected_type)
            raise SchemaError(
                f"unknown value {value!r}; expected one of {allowed}", path=path
            ) from error

    if isinstance(expected_type, type) and is_dataclass(expected_type):
        if not isinstance(value, Mapping):
            raise SchemaError("expected an object", path=path)
        if any(not isinstance(key, str) for key in value):
            raise SchemaError("object keys must be strings", path=path)
        schema_fields = {field.name: field for field in fields(expected_type)}
        unknown = sorted(set(value) - set(schema_fields))
        if unknown:
            raise SchemaError(
                f"unknown field {unknown[0]!r}", path=_child_path(path, unknown[0])
            )
        type_hints = get_type_hints(expected_type)
        decoded: dict[str, object] = {}
        for name, field in schema_fields.items():
            if name not in value:
                if field.default is MISSING and field.default_factory is MISSING:
                    raise SchemaError("missing required field", path=_child_path(path, name))
                continue
            decoded[name] = from_data(
                type_hints[name], value[name], path=_child_path(path, name)
            )
        try:
            result = expected_type(**decoded)
            validator = getattr(result, "validate", None)
            if validator is not None:
                validator(path)
            return result
        except FrontendError:
            raise
        except (TypeError, ValueError) as error:
            raise SchemaError(str(error), path=path) from error

    if expected_type in (str, int, float, bool):
        if type(value) is not expected_type:
            raise SchemaError(
                f"expected {expected_type.__name__}, got {type(value).__name__}", path=path
            )
        return value  # type: ignore[return-value]

    raise SchemaError(f"unsupported schema type {expected_type!r}", path=path)


def loads_dataclass(expected_type: type[T], text: str, *, path: str = "$") -> T:
    raw = loads_json_value(text, path=path)
    return from_data(expected_type, raw, path=path)


def loads_json_value(text: str, *, path: str = "$") -> object:
    """Decode raw JSON while preserving the frontend's strict lexical rules.

    Hardware JSON is owned by the simulator and therefore is not decoded into
    one frontend dataclass.  It must still reject duplicate keys and non-finite
    constants exactly like versioned frontend artifacts do.
    """

    if type(text) is not str:
        raise SchemaError("expected JSON text as str", path=path)
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_JSONObject,
            parse_constant=_NonFiniteJSONConstant,
        )
        return _materialize_strict_json(parsed, path)
    except SchemaError:
        raise
    except (json.JSONDecodeError, ValueError) as error:
        raise SchemaError(str(error), path=path) from error
    except RecursionError as error:
        raise SchemaError("JSON nesting is too deep", path=path) from error


def load_json_dataclass(expected_type: type[T], source: Path, *, path: str = "$") -> T:
    raw = load_json_value(source, path=path)
    return from_data(expected_type, raw, path=path)


def load_json_value(source: Path, *, path: str = "$") -> object:
    """Read a UTF-8 JSON file using :func:`loads_json_value`."""

    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise SchemaError(str(error), path=path) from error
    return loads_json_value(text, path=path)
