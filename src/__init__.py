import logging
import os
import sys
from copy import copy
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.util import env


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


LOG_LEVEL = _env_str("LOGLEVEL", "INFO").upper()


LOG_TO_FILE = env.env_flag("LOG_TO_FILE", default=False)
LOG_COLOR = env.env_flag("LOG_COLOR", default=_stream_supports_color())
LOG_FILE_DIR = _env_str("LOG_FILE_DIR", ".tmp/logs")
LOG_FILE_LEVEL = _env_str("LOG_FILE_LEVEL", LOG_LEVEL).upper()
LOG_FILE_SESSION = env.env_flag("LOG_FILE_SESSION", default=False)
LOG_FILE_ROTATE = env.env_flag("LOG_FILE_ROTATE", default=True)
LOG_FILE_MAX_BYTES = _env_int("LOG_FILE_MAX_BYTES", default=10 * 1024 * 1024, minimum=1024)
LOG_FILE_BACKUP_COUNT = _env_int("LOG_FILE_BACKUP_COUNT", default=3, minimum=0)
LOG_FILE_RETENTION_DAYS = _env_int("LOG_FILE_RETENTION_DAYS", default=7, minimum=0)
LOG_NOISY_LIB_LEVEL = _env_str("LOG_NOISY_LIB_LEVEL", "WARNING").upper()
LOG_NOISY_LIB_LOGGERS = _env_csv("LOG_NOISY_LIB_LOGGERS", "")


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


root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

handler = logging.StreamHandler()
handler.setLevel(LOG_LEVEL)


class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m",  # Red background
    }
    GREY = "\033[90m"
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        local_record = copy(record)
        levelname = local_record.levelname
        if levelname in self.COLORS:
            local_record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        local_record.name = f"{self.GREY}{local_record.name}{self.RESET}"
        local_record.link = f"\033[34m{local_record.pathname}:{local_record.lineno}\033[0m"
        return super().format(local_record)


if LOG_COLOR:
    formatter: logging.Formatter = ColorFormatter("%(asctime)s %(levelname)-16s %(name)-32s [%(link)s] \n%(message)s\n")
else:
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(name)-32s [%(pathname)s:%(lineno)d] %(message)s")
handler.setFormatter(formatter)
root_logger.addHandler(handler)

if LOG_TO_FILE:
    try:
        log_dir = Path(LOG_FILE_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        _cleanup_old_logs(log_dir, LOG_FILE_RETENTION_DAYS)

        file_path = _build_log_file_path(log_dir, LOG_FILE_SESSION)
        if LOG_FILE_ROTATE:
            file_handler = RotatingFileHandler(
                filename=str(file_path),
                maxBytes=LOG_FILE_MAX_BYTES,
                backupCount=LOG_FILE_BACKUP_COUNT,
                encoding="utf-8",
            )
        else:
            file_handler = logging.FileHandler(filename=str(file_path), encoding="utf-8")
        file_handler.setLevel(LOG_FILE_LEVEL)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s [%(pathname)s:%(lineno)d] %(message)s"))
        root_logger.addHandler(file_handler)
    except Exception:
        logging.getLogger(__name__).warning("Failed to initialize file logging", exc_info=True)

root_logger.setLevel(LOG_LEVEL)

for noisy_logger_name in LOG_NOISY_LIB_LOGGERS:
    logging.getLogger(noisy_logger_name).setLevel(LOG_NOISY_LIB_LEVEL)

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

logger.info(f"Package logger initialized with level: {logging.getLevelName(logger.level)}")
