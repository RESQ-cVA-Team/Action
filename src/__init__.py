import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TypedDict

from src.util import env
from src.util.logging_utils import (
    ColorFormatter,
    JsonFormatter,
    RequestContextFilter,
    TextFormatter,
    normalize_log_format,
    parse_log_level,
    parse_logger_level_overrides,
)


def _env_str(name: str, default: str) -> str:
    raw = env.get_env(name)
    if raw is None:
        return default
    value = raw.strip()
    return value or default


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = _env_str(name, str(default))
    try:
        return max(minimum, int(raw))
    except Exception:
        return default


def _env_csv(name: str, default: str) -> list[str]:
    raw = _env_str(name, default)
    values = [part.strip() for part in raw.split(",")]
    return [v for v in values if v]


def _stream_supports_color() -> bool:
    try:
        return bool(sys.stderr.isatty())
    except Exception:
        return False


_MANAGED_LOGGER_NAMES: set[str] = set()


class LoggingSettings(TypedDict):
    root_level: int
    root_level_name: str
    log_format: str
    log_to_file: bool
    log_color: bool
    log_file_dir: str
    log_file_level: int
    log_file_format: str
    log_file_session: bool
    log_file_rotate: bool
    log_file_max_bytes: int
    log_file_backup_count: int
    log_file_retention_days: int
    log_noisy_lib_level: int
    log_noisy_lib_loggers: list[str]
    log_module_levels: dict[str, int]


def _cleanup_old_logs(log_dir: Path, retention_days: int) -> None:
    if retention_days <= 0:
        return
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    try:
        patterns = ["actions-*.log*", "actions.log*"]
        for pattern in patterns:
            for path in log_dir.glob(pattern):
                if not path.is_file():
                    continue
                modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                if modified < cutoff:
                    path.unlink(missing_ok=True)
    except Exception:
        logging.getLogger(__name__).warning("Failed log retention cleanup", exc_info=True)


def _build_log_file_path(log_dir: Path, session_mode: bool) -> Path:
    if session_mode:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return log_dir / f"actions-{stamp}-pid{os.getpid()}.log"
    return log_dir / "actions.log"


def _load_logging_settings() -> LoggingSettings:
    root_level = parse_log_level(_env_str("LOGLEVEL", "INFO"), default=logging.INFO)
    root_level_name = str(logging.getLevelName(root_level))
    log_format = normalize_log_format(_env_str("LOG_FORMAT", "text"))
    return {
        "root_level": root_level,
        "root_level_name": root_level_name,
        "log_format": log_format,
        "log_to_file": env.env_flag("LOG_TO_FILE", default=False),
        "log_color": env.env_flag("LOG_COLOR", default=_stream_supports_color()),
        "log_file_dir": _env_str("LOG_FILE_DIR", ".tmp/logs"),
        "log_file_level": parse_log_level(_env_str("LOG_FILE_LEVEL", root_level_name), default=root_level),
        "log_file_format": normalize_log_format(_env_str("LOG_FILE_FORMAT", log_format), default=log_format),
        "log_file_session": env.env_flag("LOG_FILE_SESSION", default=False),
        "log_file_rotate": env.env_flag("LOG_FILE_ROTATE", default=True),
        "log_file_max_bytes": _env_int("LOG_FILE_MAX_BYTES", default=10 * 1024 * 1024, minimum=1024),
        "log_file_backup_count": _env_int("LOG_FILE_BACKUP_COUNT", default=3, minimum=0),
        "log_file_retention_days": _env_int("LOG_FILE_RETENTION_DAYS", default=7, minimum=0),
        "log_noisy_lib_level": parse_log_level(_env_str("LOG_NOISY_LIB_LEVEL", "WARNING"), default=logging.WARNING),
        "log_noisy_lib_loggers": _env_csv("LOG_NOISY_LIB_LOGGERS", ""),
        "log_module_levels": parse_logger_level_overrides(_env_str("LOG_MODULE_LEVELS", "")),
    }


def _build_formatter(output_format: str, use_color: bool) -> logging.Formatter:
    if output_format == "json":
        return JsonFormatter()
    if use_color:
        return ColorFormatter()
    return TextFormatter()


def _configure_handler(handler: logging.Handler, level: int, output_format: str, use_color: bool) -> None:
    handler.setLevel(level)
    handler.addFilter(RequestContextFilter())
    handler.setFormatter(_build_formatter(output_format=output_format, use_color=use_color))


def _reset_managed_logger_levels() -> None:
    for logger_name in _MANAGED_LOGGER_NAMES:
        logging.getLogger(logger_name).setLevel(logging.NOTSET)
    _MANAGED_LOGGER_NAMES.clear()


def _configure_package_logging() -> None:
    settings = _load_logging_settings()
    root_logger = logging.getLogger()
    for existing_handler in root_logger.handlers[:]:
        root_logger.removeHandler(existing_handler)
        try:
            existing_handler.close()
        except Exception:
            pass

    _reset_managed_logger_levels()

    stream_handler = logging.StreamHandler()
    _configure_handler(
        stream_handler,
        level=settings["root_level"],
        output_format=settings["log_format"],
        use_color=settings["log_color"] and settings["log_format"] == "text",
    )
    root_logger.addHandler(stream_handler)

    if settings["log_to_file"]:
        try:
            log_dir = Path(settings["log_file_dir"])
            log_dir.mkdir(parents=True, exist_ok=True)
            _cleanup_old_logs(log_dir, settings["log_file_retention_days"])

            file_path = _build_log_file_path(log_dir, settings["log_file_session"])
            if settings["log_file_rotate"]:
                file_handler: logging.Handler = RotatingFileHandler(
                    filename=str(file_path),
                    maxBytes=settings["log_file_max_bytes"],
                    backupCount=settings["log_file_backup_count"],
                    encoding="utf-8",
                )
            else:
                file_handler = logging.FileHandler(filename=str(file_path), encoding="utf-8")

            _configure_handler(
                file_handler,
                level=settings["log_file_level"],
                output_format=settings["log_file_format"],
                use_color=False,
            )
            root_logger.addHandler(file_handler)
        except Exception:
            logging.getLogger(__name__).warning("Failed to initialize file logging", exc_info=True)

    root_logger.setLevel(settings["root_level"])

    managed_logger_names: set[str] = set()
    for noisy_logger_name in settings["log_noisy_lib_loggers"]:
        logging.getLogger(noisy_logger_name).setLevel(settings["log_noisy_lib_level"])
        managed_logger_names.add(noisy_logger_name)

    for logger_name, logger_level in settings["log_module_levels"].items():
        logging.getLogger(logger_name).setLevel(logger_level)
        managed_logger_names.add(logger_name)

    _MANAGED_LOGGER_NAMES.clear()
    _MANAGED_LOGGER_NAMES.update(managed_logger_names)

    logger = logging.getLogger(__name__)
    logger.debug(
        "Logging configured",
        extra={
            "log_context": {
                "root_level": str(settings["root_level_name"]),
                "format": settings["log_format"],
                "file_logging": settings["log_to_file"],
                "module_overrides": len(settings["log_module_levels"]),
            }
        },
    )


_configure_package_logging()
