"""
A thin, dependency-light JSON-Schema validator for param grounding.

The ONLY public entry point is `validate_params(params, schema)`: it returns
`None` when `params` satisfy `schema`, otherwise a short human-readable error
string. It NEVER raises — grounding turns a failure into a DroppedStep, so the
validator must always answer with None-or-message.

Primary implementation wraps `jsonschema` (a declared dependency, v4.25.x). A
minimal hand-rolled fallback is used only if that import fails, so the dependency
swap stays trivial and the engine degrades gracefully rather than crashing.

Schema conventions in CORE's registry (kept non-strict):
- top-level is `{"type": "object", "properties": {...}, "required": [...]}`
- an empty/absent schema (`{}`) accepts ANY params (vacuously valid) — this is
  correct for git.* whose params_schema is `{}`.
- extra/unknown keys are allowed (no `additionalProperties: false`).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

try:  # pragma: no cover - import path is environment-dependent
    import jsonschema as _jsonschema
    from jsonschema import ValidationError as _ValidationError

    _HAVE_JSONSCHEMA = True
except Exception:  # pragma: no cover - fallback path
    _jsonschema = None
    _ValidationError = Exception  # type: ignore[assignment]
    _HAVE_JSONSCHEMA = False


# JSON-Schema leaf type name -> acceptable Python types.
_TYPE_MAP = {
    "string": str,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def validate_params(params: Dict[str, Any], schema: Dict[str, Any]) -> Optional[str]:
    """Return None if `params` are valid under `schema`, else a short error string.

    An empty/falsy schema accepts anything. Never raises.
    """
    if not schema:
        return None

    if _HAVE_JSONSCHEMA:
        try:
            _jsonschema.validate(instance=params, schema=schema)
            return None
        except _ValidationError as err:  # type: ignore[misc]
            return str(getattr(err, "message", err))
        except Exception as err:  # malformed schema etc. — never propagate
            return f"schema validation error: {err}"

    return _validate_params_fallback(params, schema)


def _validate_params_fallback(
    params: Dict[str, Any], schema: Dict[str, Any]
) -> Optional[str]:
    """Minimal validator: required-key presence + leaf JSON type checks.

    Only used when `jsonschema` is unavailable. Covers the schema subset CORE
    actually emits (object with leaf-typed properties + required list).
    """
    if schema.get("type") not in (None, "object"):
        # We only know how to check object schemas; accept anything else.
        return None

    required = schema.get("required", []) or []
    for key in required:
        if key not in params:
            return f"'{key}' is a required property"

    properties = schema.get("properties", {}) or {}
    for key, value in params.items():
        spec = properties.get(key)
        if not isinstance(spec, dict):
            continue  # extra/unknown key — non-strict, ignore
        expected = spec.get("type")
        if expected is None:
            continue

        if expected == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return f"{value!r} is not of type 'number'"
            continue
        if expected == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                return f"{value!r} is not of type 'integer'"
            continue

        py_type = _TYPE_MAP.get(expected)
        if py_type is None:
            continue  # unknown leaf type — accept
        # bool is a subclass of int but only valid for 'boolean'
        if expected != "boolean" and isinstance(value, bool):
            return f"{value!r} is not of type '{expected}'"
        if not isinstance(value, py_type):
            return f"{value!r} is not of type '{expected}'"

    return None
