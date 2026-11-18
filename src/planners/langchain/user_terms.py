"""Words for the user's language shared by chart descriptions and value answers."""

from typing import Any, List

from src.shared.ssot_loader import get_enum_option_label

LANGUAGE_NAMES = {"en": "English", "cs": "Czech", "el": "Greek"}

STATISTICS = {
    "MEDIAN": {"en": "median", "cs": "medián", "el": "διάμεση τιμή"},
    "MEAN": {"en": "mean", "cs": "průměr", "el": "μέση τιμή"},
    "COUNT": {"en": "number of cases", "cs": "počet případů", "el": "αριθμός περιστατικών"},
    "RATE": {"en": "share of cases (%)", "cs": "podíl případů (%)", "el": "ποσοστό περιστατικών (%)"},
    "DISTRIBUTION": {"en": "number of cases per value", "cs": "počet případů podle hodnoty", "el": "αριθμός περιστατικών ανά τιμή"},
}


def language_name(language: str) -> str:
    return LANGUAGE_NAMES.get(language, "English")


def statistic_name(measure: str, language: str) -> Any:
    names = STATISTICS.get((measure or "").upper()) or {}
    return names.get(language) or names.get("en")


def option_names(metric: Any, language: str) -> List[str]:
    """The names of a chart or value metric's chosen options, in `language`."""
    return [get_enum_option_label(metric.metric, key, language) or key for key in metric.categories]
