from __future__ import annotations

from typing import Any, Optional

from src.domain.graphql.request import DateFilter as GQLDateFilter
from src.domain.graphql.request import IntegerFilter, LogicalFilter, SexFilter, StrokeFilter
from src.domain.graphql.ssot_enums import Operator, SexType, StrokeType
from src.domain.langchain import schema as S


def _predicate_to_gql(node: S.PredicateFilter) -> Optional[Any]:
    field = (node.field or "").strip().upper()
    op = Operator(node.operator)
    value = node.value

    if field in {"AGE"}:
        if value is None:
            return None
        return IntegerFilter(property="AGE", operator=op, value=int(value))

    if field in {"NIHSS", "ADMISSION_NIHSS"}:
        if value is None:
            return None
        return IntegerFilter(property="ADMISSION_NIHSS", operator=op, value=int(value))

    if field in {"SEX", "SEX_TYPE"}:
        if not isinstance(value, str):
            return None
        return SexFilter(sexType=SexType(value.upper()))

    if field in {"STROKE", "STROKE_TYPE"}:
        if not isinstance(value, str):
            return None
        return StrokeFilter(strokeType=StrokeType(value.upper()))

    if field in {"DISCHARGE_DATE", "DATE"}:
        if not isinstance(value, str):
            return None
        return GQLDateFilter(property="DISCHARGE_DATE", operator=op, value=value)

    return None


def to_gql_filter(node: Optional[S.FilterNode]) -> Optional[Any]:
    match node:
        case None:
            return None
        case S.AndFilter(clauses=children):
            converted = [f for f in (to_gql_filter(c) for c in (children or [])) if f is not None]
            return LogicalFilter(operator="AND", children=converted)  # type: ignore[arg-type]
        case S.OrFilter(clauses=children):
            converted = [f for f in (to_gql_filter(c) for c in (children or [])) if f is not None]
            return LogicalFilter(operator="OR", children=converted)  # type: ignore[arg-type]
        case S.NotFilter(clause=inner):
            child = to_gql_filter(inner)
            if child is None:
                return None
            return LogicalFilter(operator="NOT", children=[child])
        case S.PredicateFilter():
            return _predicate_to_gql(node)
        case _:
            return None
