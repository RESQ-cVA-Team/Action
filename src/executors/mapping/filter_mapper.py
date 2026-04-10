from __future__ import annotations

from typing import Any, Optional

from src.domain.graphql.request import DateFilter as GQLDateFilter
from src.domain.graphql.request import IntegerFilter, LogicalFilter, SexFilter, StrokeFilter
from src.domain.graphql.ssot_enums import Operator, SexType, StrokeType
from src.domain.langchain import schema as S


def to_gql_filter(node: Optional[S.FilterNode]) -> Optional[Any]:
    match node:
        case None:
            return None
        case S.AndFilter(and_=children):
            converted = [f for f in (to_gql_filter(c) for c in (children or [])) if f is not None]
            return LogicalFilter(operator="AND", children=converted)  # type: ignore[arg-type]
        case S.OrFilter(or_=children):
            converted = [f for f in (to_gql_filter(c) for c in (children or [])) if f is not None]
            return LogicalFilter(operator="OR", children=converted)  # type: ignore[arg-type]
        case S.NotFilter(not_=inner):
            child = to_gql_filter(inner)
            if child is None:
                return None
            return LogicalFilter(operator="NOT", children=[child])
        case S.AgeFilter(operator=op, value=val):
            return IntegerFilter(property="AGE", operator=Operator(op), value=int(val))
        case S.NIHSSFilter(operator=op, value=val):
            return IntegerFilter(property="ADMISSION_NIHSS", operator=Operator(op), value=int(val))
        case S.SexFilter(value=val):
            return SexFilter(sexType=SexType(val))
        case S.StrokeFilter(value=val):
            return StrokeFilter(strokeType=StrokeType(val))
        case S.BooleanFilter():
            return None
        case S.DateFilter(operator=op, value=val):
            return GQLDateFilter(property="DISCHARGE_DATE", operator=Operator(op), value=val)
        case _:
            return None
