import asyncio
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

from src.domain.dto.charts import BarChart, ChartDTO, LineChart, union
from src.domain.dto.charts.types import ChartAxis, ChartMetadata, ChartPoint, ChartSeries, ChartType
from src.domain.dto.response import VisualizationResponse
from src.domain.graphql.request import (
    DataOrigin,
    DateFilter,
    GraphQLQueryRequest,
    IntegerFilter,
    LogicalFilter,
    MetricRequest,
    SexFilter,
    StrokeFilter,
    TimePeriod,
)
from src.domain.graphql.request import DateFilter as GQLDateFilter
from src.domain.graphql.ssot_enums import GroupByType, MetricType, Operator, SexType, StrokeType
from src.domain.langchain import schema as S
from src.domain.langchain.schema import (
    AnalysisPlan,
    DistributionSpec,
    GroupByAge,
    GroupByCanonicalField,
    GroupByNIHSS,
    GroupBySex,
    GroupBySpec,
    GroupByStrokeType,
    GroupByTime,
)
from src.executors.graphql.client import GraphQLProxyClient, GraphQLProxyError
from src.shared.ssot_loader import get_canonical_display_name, get_enum_option_label, get_metric_display_name, get_metric_metadata, get_sex_label, get_stroke_label
from src.util import env as env_util
from src.util.coalesce import coalesce

logger = logging.getLogger(__name__)
# Privacy/safety defaults:
# - Avoid logging raw GraphQL queries by default.
_LOG_GRAPHQL_QUERY = env_util.env_flag("EXECUTOR_LOG_GRAPHQL_QUERY", default=False)

proxy_url, action_server_token = env_util.require_all_env("RASA_PROXY_URL", "ACTION_SERVER_TOKEN")
graphql_target = env_util.require_any_env("RASA_PROXY_GRAPHQL_TARGET")
client = GraphQLProxyClient(
    proxy_url=proxy_url,
    action_server_token=action_server_token,
    target=graphql_target if isinstance(graphql_target, str) and graphql_target.strip() else "graphql",
)


METRIC_METADATA: Dict[str, Any] = get_metric_metadata()


class VisualizationExecutionError(RuntimeError):
    def __init__(self, user_message: str, reason: str = "unknown"):
        super().__init__(user_message)
        self.user_message = user_message
        self.reason = reason


def _get_metric_meta(metric_code: str) -> Dict[str, Any]:
    code = (metric_code or "").upper()
    meta: Dict[str, Any] = METRIC_METADATA.get(code) or {}
    return meta


def _metric_label_from_alias(metric_alias: str) -> str:
    code = metric_alias
    if code.lower().startswith("metric_"):
        code = code[len("metric_") :]
    code_up = code.upper()
    return get_metric_display_name(code_up)


def _derive_distribution_defaults(metric_code: str) -> tuple[int, int, int]:
    """Return (bins, min_value, max_value) using SSOT metadata with sensible fallbacks.

    Priority:
    - Use top-level promoted keys when present: range_min/range_max and distribution_default_buckets
    - Else use nested numeric block fields if present: numeric.default_buckets (and optional range_min/range_max if provided)
    - Else fallback to known safe ranges per metric; else general default (20, 0, 200)
    """
    meta = _get_metric_meta(metric_code)

    bins_any: Any = meta.get("distribution_default_buckets")
    numeric_block: Dict[str, Any] = cast(Dict[str, Any], meta.get("numeric") or {})
    bins = bins_any or numeric_block.get("default_buckets") or 20

    rmin: Any = meta.get("range_min")
    rmax: Any = meta.get("range_max")
    if rmin is None or rmax is None:
        n: Dict[str, Any] = cast(Dict[str, Any], meta.get("numeric") or {})
        rmin = rmin if rmin is not None else n.get("range_min")
        rmax = rmax if rmax is not None else n.get("range_max")

    if rmin is None or rmax is None:
        defaults: dict[str, tuple[int, int]] = {
            "AGE": (18, 95),
            "ADMISSION_NIHSS": (0, 42),
            "DTN": (0, 120),
        }
        if metric_code.upper() in defaults:
            rmin, rmax = defaults[metric_code.upper()]
        else:
            rmin = rmin if rmin is not None else 0
            rmax = rmax if rmax is not None else 200

    try:
        bins = int(bins)
    except Exception:
        bins = 20
    try:
        rmin = int(rmin)
        rmax = int(rmax)
    except Exception:
        rmin, rmax = 0, 200
    if rmin > rmax:
        rmin, rmax = rmax, rmin
    return bins, rmin, rmax


def _axis_from_meta(metric_code: str, x_min: int, x_max: int) -> tuple[ChartAxis, ChartAxis]:
    meta = _get_metric_meta(metric_code)
    display = get_metric_display_name(metric_code)
    unit_any: Any = meta.get("unit")
    if unit_any is None:
        unit_any = cast(Dict[str, Any], meta.get("numeric") or {}).get("unit")
    unit: Optional[str] = cast(Optional[str], unit_any)
    x_label = f"{display} ({unit})" if unit else display
    x_axis = ChartAxis(label=x_label, min_value=x_min, max_value=x_max)
    y_axis = ChartAxis(label="Cases")
    return x_axis, y_axis


class Dimension:
    """Represents one grouping dimension and how to enumerate categories/filters."""

    def __init__(self, spec: GroupBySpec):
        self.spec = spec
        self.kind = type(spec)

    def is_canonical(self) -> bool:
        return isinstance(self.spec, GroupByCanonicalField)

    def categories(self) -> Sequence[Any]:
        if isinstance(self.spec, GroupBySex):
            return list(self.spec.categories or list(SexType))
        if isinstance(self.spec, GroupByStrokeType):
            return list(self.spec.categories or list(StrokeType))
        if isinstance(self.spec, GroupByTime):
            window = self.spec.window
            if isinstance(window, S.TimeWindow) and str(window.unit).upper() == "MONTH":
                from datetime import date

                today = date.today()
                buckets: list[tuple[date, date]] = []
                year = today.year
                month = today.month
                for i in range(window.last_n):
                    m = month - i
                    y = year
                    while m <= 0:
                        m += 12
                        y -= 1
                    from calendar import monthrange

                    start_day = 1
                    end_day = monthrange(y, m)[1]
                    buckets.append((date(y, m, start_day), date(y, m, end_day)))
                buckets.reverse()
                return buckets
            return []
        if isinstance(self.spec, GroupByAge):
            return list(self.spec.buckets)
        if isinstance(self.spec, GroupByNIHSS):
            return list(self.spec.buckets)
        return []

    def label_for(self, cat: Any) -> str:
        if isinstance(self.spec, GroupBySex):
            val = cat if isinstance(cat, SexType) else SexType(cat)
            raw = getattr(val, "value", str(val))
            return get_sex_label(str(raw).upper())
        if isinstance(self.spec, GroupByStrokeType):
            val = cat if isinstance(cat, StrokeType) else StrokeType(cat)
            raw = getattr(val, "value", str(val))
            return get_stroke_label(str(raw).upper())
        if isinstance(self.spec, (GroupByAge, GroupByNIHSS)):
            return f"{cat.min}-{cat.max}"
        if isinstance(self.spec, GroupByCanonicalField):
            return self.spec.field
        if isinstance(self.spec, GroupByTime):
            try:
                start, end = cat
                return f"{start.isoformat()} to {end.isoformat()}"
            except Exception:
                return self.spec.grain
        return str(cat)

    def filter_for(self, cat: Any) -> Optional[Any]:
        if isinstance(self.spec, GroupBySex):
            val = cat if isinstance(cat, SexType) else SexType(cat)
            return SexFilter(sexType=val)
        if isinstance(self.spec, GroupByStrokeType):
            val = cat if isinstance(cat, StrokeType) else StrokeType(cat)
            return StrokeFilter(strokeType=val)
        if isinstance(self.spec, GroupByAge):
            return LogicalFilter(
                operator="AND",
                children=[
                    IntegerFilter(property="AGE", operator=Operator("GE"), value=cat.min),
                    IntegerFilter(property="AGE", operator=Operator("LT"), value=cat.max),
                ],
            )
        if isinstance(self.spec, GroupByNIHSS):
            return LogicalFilter(
                operator="AND",
                children=[
                    IntegerFilter(property="ADMISSION_NIHSS", operator=Operator("GE"), value=cat.min),
                    IntegerFilter(property="ADMISSION_NIHSS", operator=Operator("LT"), value=cat.max),
                ],
            )
        if isinstance(self.spec, GroupByTime):
            try:
                start, end = cat
            except Exception:
                return None
            return LogicalFilter(
                operator="AND",
                children=[
                    DateFilter(property="DISCHARGE_DATE", operator=Operator("GE"), value=start.isoformat()),
                    DateFilter(property="DISCHARGE_DATE", operator=Operator("LE"), value=end.isoformat()),
                ],
            )
        return None


def execute_plan(plan: AnalysisPlan, user_sub: str) -> VisualizationResponse:
    """Sync wrapper that delegates to the async implementation with concurrency=1.

    - If no event loop is running, run the coroutine directly with asyncio.run.
    - If an event loop is already running, offload to a new thread and run a fresh loop there.
    """
    coro = execute_plan_async(plan, user_sub, max_concurrency=1)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(asyncio.run, coro)
        return fut.result()


ProgressCallback = Callable[[str], None]


async def execute_plan_async(
    plan: AnalysisPlan,
    user_sub: str,
    max_concurrency: int = 4,
    progress_cb: Optional[ProgressCallback] = None,
) -> VisualizationResponse:
    """Async version that runs GraphQL requests concurrently.

    - Uses asyncio.to_thread to run the existing synchronous client in a thread pool.
    - Limits concurrency via a semaphore to avoid overloading the proxy/backend.
    - Produces one chart per canonical GroupBy (or one overall if none), matching sync behavior.
    """
    planCharts = coalesce(plan.charts, [])
    if plan.statistical_tests:
        logger.warning("Statistical tests are defined in the plan but not implemented in this executor yet. They will be ignored.")
    response: VisualizationResponse = VisualizationResponse()

    sem = asyncio.Semaphore(max_concurrency)

    def _to_execution_error(failure_reasons: List[str]) -> VisualizationExecutionError:
        reason_set = set(failure_reasons)
        if "timeout" in reason_set:
            return VisualizationExecutionError(
                user_message="The data service timed out while generating the visualization. Please try again.",
                reason="timeout",
            )
        if "service_unavailable" in reason_set:
            return VisualizationExecutionError(
                user_message="The analytics service is temporarily unavailable. Please try again in a moment.",
                reason="service_unavailable",
            )
        if "graphql_error" in reason_set:
            return VisualizationExecutionError(
                user_message="The analytics service returned an error while generating the visualization.",
                reason="graphql_error",
            )
        return VisualizationExecutionError(
            user_message="Could not fetch analytics data for this visualization. Please try again.",
            reason="data_fetch_failed",
        )

    def _to_gql_filter(node: Optional[S.FilterNode]) -> Optional[Any]:
        match node:
            case None:
                return None
            case S.AndFilter(and_=children):
                converted = [f for f in (_to_gql_filter(c) for c in (children or [])) if f is not None]
                return LogicalFilter(operator="AND", children=converted)  # type: ignore[arg-type]
            case S.OrFilter(or_=children):
                converted = [f for f in (_to_gql_filter(c) for c in (children or [])) if f is not None]
                return LogicalFilter(operator="OR", children=converted)  # type: ignore[arg-type]
            case S.NotFilter(not_=inner):
                child = _to_gql_filter(inner)
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

    def _collect_date_bounds(filter_obj: Optional[Any]) -> tuple[Optional[str], Optional[str]]:
        if filter_obj is None:
            return None, None

        from src.domain.graphql.request import DateFilter as GQLDateFilter
        from src.domain.graphql.request import LogicalFilter as GQLLogicalFilter

        min_start: Optional[str] = None
        max_end: Optional[str] = None

        def visit(node: Any) -> None:
            nonlocal min_start, max_end
            if isinstance(node, GQLLogicalFilter):
                for child in node.children:
                    visit(child)
                return
            if isinstance(node, GQLDateFilter) and node.property == "DISCHARGE_DATE":
                op = node.operator.value
                val = node.value
                if op in ("GE", "GT"):
                    if min_start is None or val < min_start:
                        min_start = val
                if op in ("LE", "LT"):
                    if max_end is None or val > max_end:
                        max_end = val

        visit(filter_obj)
        return min_start, max_end

    async def _run_one_request(
        req: GraphQLQueryRequest,
        label_parts: List[str],
        include_metric_alias: bool,
        group_by_field: Optional[str],
        request_failures: List[str],
    ) -> List[ChartSeries]:
        async with sem:
            query_str = req.to_graphql_string()
            q_hash = hashlib.sha256(query_str.encode("utf-8")).hexdigest()[:12]
            if _LOG_GRAPHQL_QUERY:
                logger.debug(
                    "[plan_executor] GraphQL query for chart (groupBy=%s, labels=%s, hash=%s):\n%s",
                    group_by_field,
                    " | ".join(label_parts),
                    q_hash,
                    query_str,
                )
            else:
                logger.info(
                    "[plan_executor] GraphQL query for chart (groupBy=%s, labels=%s, hash=%s, len=%s)",
                    group_by_field,
                    " | ".join(label_parts),
                    q_hash,
                    len(query_str),
                )
            try:
                resp = await asyncio.to_thread(client.query, query_str, user_sub, None, True)
            except GraphQLProxyError as exc:
                if exc.kind == "timeout":
                    request_failures.append("timeout")
                elif exc.kind == "http_error" and exc.status_code in {429, 500, 502, 503, 504}:
                    request_failures.append("service_unavailable")
                else:
                    request_failures.append("upstream_error")
                logger.error(
                    "[plan_executor] GraphQL request failed (groupBy=%s, labels=%s, kind=%s, status=%s)",
                    group_by_field,
                    " - ".join([p for p in label_parts if p]) or "(none)",
                    exc.kind,
                    exc.status_code,
                )
                return []
        series: List[ChartSeries] = []
        if resp is None:
            request_failures.append("upstream_error")
            logger.error("[plan_executor] GraphQL returned empty response")
            return series
        metrics_payload = None
        if (x := resp) and (x := x.data) and (x := x.get_metrics) and (x := x.metrics):
            metrics_payload = x
        if getattr(resp, "errors", None):
            request_failures.append("graphql_error")
            error_count = len(resp.errors or [])
            logger.error("[plan_executor] GraphQL errors returned (count=%s)", error_count)
        if metrics_payload:
            for metricName, metric in metrics_payload.items():
                for kpi in metric.kpi_group:
                    if not kpi.kpi1.d1:
                        continue
                    server_label = kpi.grouped_by.group_item_name if kpi.grouped_by else None
                    parts: List[str] = []
                    if include_metric_alias:
                        parts.append(_metric_label_from_alias(metricName))
                    parts.extend([p for p in label_parts if p])
                    if server_label:
                        if group_by_field:
                            mapped = get_enum_option_label(group_by_field, server_label)
                        else:
                            mapped = None
                        parts.append(mapped or server_label)
                    series_name = " — ".join(parts) if parts else _metric_label_from_alias(metricName)
                    series.append(
                        ChartSeries(
                            name=series_name,
                            data=[ChartPoint(x=x, y=y) for x, y in zip(kpi.kpi1.d1.edges, kpi.kpi1.d1.case_count)],
                        )
                    )
        return series

    for planChart in planCharts:
        metric_requests: List[MetricRequest] = []
        derived_axes: Optional[tuple[ChartAxis, ChartAxis]] = None
        for metric in planChart.metrics:
            if metric.distribution is not None:
                distribution = metric.distribution
            else:
                bins, rmin, rmax = _derive_distribution_defaults(metric.metric)
                distribution = DistributionSpec(num_buckets=bins, min_value=rmin, max_value=rmax)
                if len(planChart.metrics) == 1:
                    derived_axes = _axis_from_meta(metric.metric, rmin, rmax)
            metric_requests.append(MetricRequest(metricType=MetricType(metric.metric)).with_distribution(distribution.num_buckets, distribution.min_value, distribution.max_value))

        collected_groups: List[GroupBySpec] = list(coalesce(planChart.group_by, []))

        seen: set[GroupBySpec] = set()
        uniq_groups: List[GroupBySpec] = []
        for g in collected_groups:
            if g not in seen:
                seen.add(g)
                uniq_groups.append(g)
        dims: List[Dimension] = [Dimension(g) for g in uniq_groups]

        server_dims: List[Optional[Dimension]] = [d for d in dims if d.is_canonical()]
        if not server_dims:
            server_dims = [None]

        for server_dim in server_dims:
            request_failures: List[str] = []
            filter_dims: List[Dimension] = [d for d in dims if d is not server_dim]

            filter_categories: List[Sequence[Any]] = []
            for d in filter_dims:
                cats = d.categories()
                if not cats:
                    logger.warning("Skipping non-enumerable filter dimension: %s", d.kind.__name__)
                    continue
                filter_categories.append(cats)

            if not filter_categories:
                combos_list: List[Tuple[Any, ...]] = [tuple()]
            else:
                combos_list = list(product(*filter_categories))

            total_requests = len(combos_list)
            completed_requests = 0

            if progress_cb is not None and total_requests > 0:
                progress_cb(f"Fetching data (0/{total_requests})")

            async def run_one(
                req: GraphQLQueryRequest,
                label_parts: List[str],
                include_metric_alias: bool,
                group_by_field: Optional[str],
                request_failures: List[str],
            ) -> List[ChartSeries]:
                nonlocal completed_requests
                result = await _run_one_request(req, label_parts, include_metric_alias, group_by_field, request_failures)
                if progress_cb is not None:
                    completed_requests += 1
                    if total_requests > 0:
                        progress_cb(f"Fetching data ({completed_requests}/{total_requests})")
                    else:
                        progress_cb("Fetching data…")
                return result

            tasks: List[asyncio.Task[List[ChartSeries]]] = []
            include_metric_alias = len(planChart.metrics) > 1
            gb_field = server_dim.spec.field if server_dim and isinstance(server_dim.spec, GroupByCanonicalField) else None

            chart_filter = _to_gql_filter(coalesce(planChart.filters, None))

            for combo in combos_list:
                combo_filters: List[Any] = []
                label_parts: List[str] = []
                for dim, cat in zip(filter_dims, combo):
                    f = dim.filter_for(cat)
                    if f is not None:
                        combo_filters.append(f)
                    label_parts.append(dim.label_for(cat))

                if len(combo_filters) == 0:
                    case_filter: Optional[Any] = chart_filter
                elif len(combo_filters) == 1 and chart_filter is None:
                    case_filter = combo_filters[0]
                else:
                    merged_children: List[Any] = []
                    if chart_filter is not None:
                        merged_children.append(chart_filter)
                    merged_children.extend(combo_filters)
                    case_filter = LogicalFilter(operator="AND", children=merged_children)  # type: ignore[arg-type]

                start_bound, end_bound = _collect_date_bounds(case_filter)

                req = GraphQLQueryRequest(
                    metrics=metric_requests,
                    timePeriod=TimePeriod(startDate=start_bound, endDate=end_bound),
                    dataOrigin=DataOrigin(providerGroupId=[1]),
                    includeGeneralStats=True,
                    caseFilter=case_filter,
                    groupBy=(GroupByType(server_dim.spec.field) if server_dim and isinstance(server_dim.spec, GroupByCanonicalField) else None),
                )

                tasks.append(asyncio.create_task(run_one(req, label_parts, include_metric_alias, gb_field, request_failures)))

            all_series: List[ChartSeries] = []
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=False)
                for lst in results:
                    all_series.extend(lst)

            if not all_series:
                logger.warning(
                    "[test2] No series generated for chart '%s'%s. This often indicates a backend error or empty results.",
                    planChart.title or "Chart",
                    "",
                )
                if request_failures:
                    raise _to_execution_error(request_failures)
            metric_codes = [m.metric for m in planChart.metrics]
            metric_names: List[str] = []
            for code in metric_codes:
                metric_names.append(get_metric_display_name(code))
            dim_names: List[str] = []
            for d in dims:
                if isinstance(d.spec, GroupBySex):
                    dim_names.append("Sex")
                elif isinstance(d.spec, GroupByNIHSS):
                    dim_names.append(get_canonical_display_name("ADMISSION_NIHSS"))
                elif isinstance(d.spec, GroupByAge):
                    dim_names.append(get_canonical_display_name("AGE"))
                elif isinstance(d.spec, GroupByCanonicalField):
                    dim_names.append(get_canonical_display_name(d.spec.field))
            dims_phrase = f" by {' and '.join(dim_names)}" if dim_names else ""

            if planChart.title:
                base = planChart.title
                title_text = base if (" by " in base.lower()) or not dims_phrase else base + dims_phrase
            else:
                if len(metric_names) == 0:
                    base_title = f"{planChart.chart_type.title()} Chart"
                elif len(metric_names) == 1:
                    base_title = f"{metric_names[0]} Distribution"
                elif len(metric_names) == 2:
                    base_title = f"{metric_names[0]} and {metric_names[1]}"
                else:
                    base_title = ", ".join(metric_names[:-1]) + f" and {metric_names[-1]}"
                title_text = base_title + dims_phrase

            meta_kwargs: dict[str, Any] = {
                "title": title_text,
                "description": planChart.description,
            }
            if derived_axes is not None:
                meta_kwargs["x_axis"], meta_kwargs["y_axis"] = derived_axes

            chart_type_upper = (planChart.chart_type or "").upper()
            vis_chart: ChartDTO
            if chart_type_upper == ChartType.LINE.value:
                vis_chart = LineChart(metadata=ChartMetadata(**meta_kwargs), series=all_series)
            elif chart_type_upper == ChartType.BAR.value:
                vis_chart = BarChart(metadata=ChartMetadata(**meta_kwargs), series=all_series)
            elif chart_type_upper == ChartType.AREA.value:
                vis_chart = union.AreaChart(metadata=ChartMetadata(**meta_kwargs), series=all_series)
            else:
                logger.warning("Chart type %s not yet implemented; defaulting to LINE rendering", planChart.chart_type)
                vis_chart = LineChart(metadata=ChartMetadata(**meta_kwargs), series=all_series)

            response.charts.append(vis_chart)

    return response
