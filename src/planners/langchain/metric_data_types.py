import json
import re
from typing import Any, Dict

from src.shared.ssot_loader import get_enum_options, get_metric_data_type

_CODE = re.compile(r"\b[A-Z][A-Z0-9_]*\b")


def metric_data_types(question: str, entities: Dict[str, Any]) -> Dict[str, str]:
    """SSOT data type of each metric the request mentions, as a detected entity
    or as a code in the question text (a resolved metric, a previous plan)."""
    values = (entities or {}).get("metric")
    codes = {str(v).strip().upper() for v in (values if isinstance(values, list) else [values]) if isinstance(v, str)}
    codes.update(_CODE.findall(question or ""))
    found = {code: get_metric_data_type(code) for code in sorted(codes)}
    return {code: data_type for code, data_type in found.items() if data_type}


def metric_data_types_json(question: str, entities: Dict[str, Any]) -> str:
    return json.dumps(metric_data_types(question, entities))


def metric_options(question: str, entities: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Option keys and names of each Enum metric the request mentions, other
    than yes/no metrics, for choosing which options a chart counts together."""
    out: Dict[str, Dict[str, str]] = {}
    for code, data_type in metric_data_types(question, entities).items():
        options = get_enum_options(code) if data_type == "Enum" else {}
        if options and set(options) != {"yes", "no"}:
            out[code] = options
    return out


def metric_options_json(question: str, entities: Dict[str, Any]) -> str:
    return json.dumps(metric_options(question, entities))
