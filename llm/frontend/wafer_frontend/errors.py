"""Stable, path-aware frontend diagnostics."""

from __future__ import annotations


class FrontendError(Exception):
    """Base class for errors that may be shown directly by the CLI."""

    default_code = "frontend_error"

    def __init__(
        self,
        message: str,
        *,
        path: str = "$",
        code: str | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.path = path
        self.code = code or self.default_code
        self.hint = hint

    def __str__(self) -> str:
        result = f"{self.code} at {self.path}: {self.message}"
        if self.hint:
            result += f"; hint: {self.hint}"
        return result


class SchemaError(FrontendError):
    default_code = "schema_error"


class UnsupportedFeatureError(FrontendError):
    default_code = "unsupported_feature"


class PassOrderError(FrontendError):
    default_code = "pass_order_error"


class InputMutationError(FrontendError):
    default_code = "input_mutation"


class RegistryError(FrontendError):
    default_code = "registry_error"


class StageNotImplementedError(FrontendError):
    default_code = "stage_not_implemented"
