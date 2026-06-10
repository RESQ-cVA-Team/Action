from typing import Any, Callable, Dict, Iterable, List, Protocol, TypeVar


class CommandHandler(Protocol):
    def __call__(self, dispatcher: Any, tracker: Any, domain: Any, args: List[str], opts: Dict[str, Any]) -> List[Any]: ...


_REGISTRY: Dict[str, CommandHandler] = {}


def register(name: str, handler: CommandHandler) -> None:
    _REGISTRY[name] = handler


def get(name: str) -> CommandHandler | None:
    return _REGISTRY.get(name)


def names() -> Iterable[str]:
    return _REGISTRY.keys()


# Optional decorator for ergonomic registration
F = TypeVar("F", bound=Callable[..., List[Any]])


def command(name: str) -> Callable[[F], F]:
    def _decorator(fn: F) -> F:
        register(name, fn)
        return fn

    return _decorator


from . import core as _core  # noqa: E402
from .test import analytics as _test_analytics  # noqa: E402
from .test import area as _test_area  # noqa: E402
from .test import bar as _test_bar  # noqa: E402
from .test import box as _test_box  # noqa: E402
from .test import charts as _test_charts  # noqa: E402
from .test import graphql as _test_graphql  # noqa: E402
from .test import histogram as _test_histogram  # noqa: E402
from .test import line as _test_line  # noqa: E402
from .test import pie as _test_pie  # noqa: E402
from .test import radar as _test_radar  # noqa: E402
from .test import scatter as _test_scatter  # noqa: E402
from .test import stream as _test_stream  # noqa: E402
from .test import waterfall as _test_waterfall  # noqa: E402

_ = (
    _core,
    _test_graphql,
    _test_analytics,
    _test_bar,
    _test_pie,
    _test_histogram,
    _test_box,
    _test_scatter,
    _test_radar,
    _test_waterfall,
    _test_area,
    _test_line,
    _test_stream,
    _test_charts,
)
