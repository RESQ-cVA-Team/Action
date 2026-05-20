import os
from typing import Tuple, overload

from dotenv import load_dotenv

_loaded = False


def _ensure_loaded() -> None:
    """Load .env once (if present) to populate os.environ.

    This keeps behavior consistent across the codebase: any env access via
    this module will see values from a local .env file without requiring each
    caller to remember to call load_dotenv.
    """

    global _loaded
    if not _loaded:
        load_dotenv(override=True)
        _loaded = True


def env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment flag.

    Truthy values: 1, true, yes, y, on (case-insensitive)
    Falsy values:  0, false, no, n, off, empty

    Any other non-empty value falls back to `default` to avoid surprising
    behavior from typos.
    """

    _ensure_loaded()
    val = os.getenv(name)
    if val is None:
        return default
    norm = val.strip().lower()
    if norm in {"1", "true", "yes", "y", "on"}:
        return True
    if norm in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def get_env(name: str, default: str | None = None) -> str | None:
    """Get an optional environment variable with dotenv loading applied."""

    _ensure_loaded()
    value = os.getenv(name)
    return value if value is not None else default


@overload
def require_all_env(key: str) -> str: ...
@overload
def require_all_env(*keys: str) -> Tuple[str, ...]: ...


def require_all_env(*keys: str) -> str | Tuple[str, ...]:
    _ensure_loaded()

    values: list[str] = []
    missing: list[str] = []

    for key in keys:
        val = os.getenv(key)
        if val is None:
            missing.append(key)
        else:
            values.append(val)

    if missing:
        raise OSError(f"Missing required environment variable(s): {', '.join(missing)}")

    return tuple(values) if len(values) > 1 else values[0]


@overload
def require_any_env(key: str) -> str | None: ...
@overload
def require_any_env(*keys: str) -> Tuple[str | None, ...]: ...


def require_any_env(*keys: str) -> str | None | Tuple[str | None, ...]:
    _ensure_loaded()

    values: list[str | None] = []
    for key in keys:
        val = os.getenv(key)
        values.append(val)

    if all(v is None for v in values):
        raise OSError(f"At least one of the required environment variables must be set: {', '.join(keys)}")

    return tuple(values) if len(values) > 1 else values[0]
