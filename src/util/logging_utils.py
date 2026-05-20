from __future__ import annotations

import json
import logging
import re
from contextlib import contextmanager
from contextvars import ContextVar, Token, copy_context
from copy import copy
from typing import Any, Callable, Generator, Mapping, ParamSpec, TypeVar, cast

DEFAULT_TRACE_ID = "-"
DEFAULT_LOG_FORMAT = "text"
DEFAULT_TEXT_LOG_TEMPLATE = "%(asctime)s %(levelname)-8s %(name)-32s [%(source_location)s] %(message)s%(context_suffix)s"

_COLOR_LEVELS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[41m",
}
_COLOR_DIM = "\033[90m"
_COLOR_LINK = "\033[34m"
_COLOR_RESET = "\033[0m"
_SIMPLE_VALUE_RE = re.compile(r"^[A-Za-z0-9_.:/-]+$")
_VALID_LOG_FORMATS = {"text", "json"}
_LOG_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("src_log_context", default={})
_P = ParamSpec("_P")
_R = TypeVar("_R")


def _level_names_mapping() -> dict[str, int]:
    get_mapping = getattr(logging, "getLevelNamesMapping", None)
    if callable(get_mapping):
        mapping_any = get_mapping()
        if isinstance(mapping_any, Mapping):
            mapping = cast(Mapping[object, object], mapping_any)
            resolved: dict[str, int] = {}
            for key, value in mapping.items():
                if isinstance(key, str) and isinstance(value, int):
                    resolved[key] = value
            return resolved

    legacy_mapping_any = getattr(logging, "_nameToLevel", {})
    if isinstance(legacy_mapping_any, Mapping):
        legacy_mapping = cast(Mapping[object, object], legacy_mapping_any)
        resolved: dict[str, int] = {}
        for key, value in legacy_mapping.items():
            if isinstance(key, str) and isinstance(value, int):
                resolved[key] = value
        return resolved
    return {}


def _normalize_context_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in fields.items():
        token = key.strip()
        if not token or value is None:
            continue
        if isinstance(value, str):
            trimmed = value.strip()
            if not trimmed:
                continue
            normalized[token] = trimmed
            continue
        normalized[token] = value
    return normalized


def _default_level(default: str | int) -> int:
    if isinstance(default, int):
        return default
    candidate = _level_names_mapping().get(str(default).strip().upper())
    return candidate if isinstance(candidate, int) else logging.INFO


def _format_context_value(value: Any) -> str:
    if isinstance(value, str):
        return value if _SIMPLE_VALUE_RE.match(value) else json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(value)


def normalize_log_format(value: str | None, default: str = DEFAULT_LOG_FORMAT) -> str:
    token = (value or "").strip().lower()
    if token in _VALID_LOG_FORMATS:
        return token
    fallback = (default or DEFAULT_LOG_FORMAT).strip().lower()
    return fallback if fallback in _VALID_LOG_FORMATS else DEFAULT_LOG_FORMAT


def parse_log_level(value: str | int | None, default: str | int = logging.INFO) -> int:
    if isinstance(value, int):
        return value

    fallback = _default_level(default)
    if value is None:
        return fallback

    token = str(value).strip()
    if not token:
        return fallback

    if token.lstrip("-").isdigit():
        return int(token)

    resolved = _level_names_mapping().get(token.upper())
    return resolved if isinstance(resolved, int) else fallback


def parse_logger_level_overrides(raw: str | None) -> dict[str, int]:
    overrides: dict[str, int] = {}
    if raw is None:
        return overrides

    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue

        if "=" in token:
            logger_name, level_name = token.split("=", 1)
        elif ":" in token:
            logger_name, level_name = token.split(":", 1)
        else:
            continue

        logger_name = logger_name.strip()
        if not logger_name:
            continue

        parsed_level = parse_log_level(level_name, default=-1)
        if parsed_level < 0:
            continue
        overrides[logger_name] = parsed_level

    return overrides


def get_log_context() -> dict[str, Any]:
    return dict(_LOG_CONTEXT.get())


def capture_log_context() -> dict[str, Any]:
    return get_log_context()


def bind_current_context(func: Callable[_P, _R]) -> Callable[_P, _R]:
    context = copy_context()

    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        return context.run(func, *args, **kwargs)

    return wrapped


def clear_log_context() -> None:
    _LOG_CONTEXT.set({})


def set_log_context(**fields: Any) -> Token[dict[str, Any]]:
    merged = get_log_context()
    merged.update(_normalize_context_fields(fields))
    return _LOG_CONTEXT.set(merged)


def replace_log_context(fields: Mapping[str, Any]) -> Token[dict[str, Any]]:
    return _LOG_CONTEXT.set(_normalize_context_fields(fields))


def restore_log_context(token: Token[dict[str, Any]]) -> None:
    _LOG_CONTEXT.reset(token)


@contextmanager
def log_context(**fields: Any) -> Generator[dict[str, Any], None, None]:
    token = set_log_context(**fields)
    try:
        yield get_log_context()
    finally:
        restore_log_context(token)


@contextmanager
def applied_log_context(fields: Mapping[str, Any]) -> Generator[dict[str, Any], None, None]:
    token = replace_log_context(fields)
    try:
        yield get_log_context()
    finally:
        restore_log_context(token)


def format_context_suffix(context: Mapping[str, Any]) -> str:
    if not context:
        return ""

    rendered: list[str] = []
    for key in sorted(context):
        value = context[key]
        if key == "trace_id" and str(value).strip() == DEFAULT_TRACE_ID:
            continue
        rendered.append(f"{key}={_format_context_value(value)}")

    return f" [{' , '.join(rendered).replace(' , ', ', ')}]" if rendered else ""


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        merged = get_log_context()

        extra_context = getattr(record, "log_context", None)
        if isinstance(extra_context, Mapping):
            merged.update(_normalize_context_fields(cast(Mapping[str, Any], extra_context)))

        explicit_trace_id = getattr(record, "trace_id", None)
        if explicit_trace_id is not None:
            normalized_trace_id = str(explicit_trace_id).strip()
            if normalized_trace_id:
                merged["trace_id"] = normalized_trace_id

        trace_id = str(merged.get("trace_id") or "").strip()
        if trace_id:
            merged["trace_id"] = trace_id
        else:
            merged.pop("trace_id", None)

        setattr(record, "trace_id", trace_id or DEFAULT_TRACE_ID)
        setattr(record, "log_context", merged)
        setattr(record, "context_suffix", format_context_suffix(merged))
        setattr(record, "source_location", f"{record.pathname}:{record.lineno}")
        return True


class TextFormatter(logging.Formatter):
    def __init__(self, use_color: bool = False):
        super().__init__(DEFAULT_TEXT_LOG_TEMPLATE)
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        local_record = copy(record)
        context_suffix = getattr(local_record, "context_suffix", "")
        if not isinstance(context_suffix, str):
            context_suffix = ""
        setattr(local_record, "context_suffix", context_suffix)

        source_location_any = getattr(local_record, "source_location", None)
        source_location = source_location_any if isinstance(source_location_any, str) else f"{local_record.pathname}:{local_record.lineno}"

        if self._use_color:
            level_name = local_record.levelname
            color = _COLOR_LEVELS.get(level_name)
            if color:
                local_record.levelname = f"{color}{level_name}{_COLOR_RESET}"
            local_record.name = f"{_COLOR_DIM}{local_record.name}{_COLOR_RESET}"
            source_location = f"{_COLOR_LINK}{source_location}{_COLOR_RESET}"

        setattr(local_record, "source_location", source_location)

        return super().format(local_record)


class ColorFormatter(TextFormatter):
    def __init__(self):
        super().__init__(use_color=True)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        context_any = getattr(record, "log_context", None)
        context = _normalize_context_fields(cast(Mapping[str, Any], context_any)) if isinstance(context_any, Mapping) else {}
        trace_id = str(context.get("trace_id") or getattr(record, "trace_id", DEFAULT_TRACE_ID)).strip() or DEFAULT_TRACE_ID

        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": trace_id,
            "source": {
                "path": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            },
        }
        if context:
            payload["context"] = context
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = record.stack_info

        return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
