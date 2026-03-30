import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, cast

from src.domain.dto.charts.types import ChartAxis, ChartSeries
from src.domain.dto.execution_summary import ExecutionBatchSummary, ExecutionSummary
from src.domain.dto.response import VisualizationResponse
from src.domain.langchain.schema import AnalysisPlan
from src.executors.graphql.client import GraphQLProxyClient
from src.executors.mapping.chart_builder import build_chart_dto
from src.executors.mapping.filter_mapper import to_gql_filter
from src.executors.mapping.series_mapper import merge_series_by_name
from src.executors.mapping.summary_builder import make_batch_summary, make_execution_summary
from src.executors.planning.metric_request_factory import build_metric_requests
from src.executors.planning.query_compiler import Dimension, compile_chart_grouping, estimate_query_count_for_plan
from src.executors.planning.request_plan import RequestSpec, build_fallback_request_specs, build_primary_request_specs, should_retry_unbatched_time
from src.executors.transport.request_runner import run_graphql_request
from src.planners.langchain.semantic_adapter import normalize_analysis_plan_with_diagnostics
from src.shared.ssot_loader import get_metric_display_name, get_metric_metadata
from src.util import env as env_util
from src.util.coalesce import coalesce

logger = logging.getLogger(__name__)
# Privacy/safety defaults:
# - Avoid logging raw GraphQL queries by default.
_LOG_GRAPHQL_QUERY = env_util.env_flag("EXECUTOR_LOG_GRAPHQL_QUERY", default=False)
_EMIT_COMPILER_DIAGNOSTICS = env_util.env_flag("EXECUTOR_EMIT_COMPILER_DIAGNOSTICS", default=False)
_ENABLE_UNBATCHED_TIME_FALLBACK = env_util.env_flag("EXECUTOR_ENABLE_UNBATCHED_TIME_FALLBACK", default=True)
_STRICT_MODE = env_util.env_flag("ANALYTICS_STRICT_MODE", default=False) or env_util.env_flag("EXECUTOR_STRICT_MODE", default=False)

_executor_default_concurrency_raw = env_util.get_env("EXECUTOR_DEFAULT_MAX_CONCURRENCY", default="4") or "4"
try:
    _executor_default_concurrency = max(1, int(_executor_default_concurrency_raw))
except Exception:
    _executor_default_concurrency = 4
_EXECUTOR_DEFAULT_MAX_CONCURRENCY = _executor_default_concurrency

_executor_sync_concurrency_raw = env_util.get_env("EXECUTOR_SYNC_MAX_CONCURRENCY", default="1") or "1"
try:
    _executor_sync_concurrency = max(1, int(_executor_sync_concurrency_raw))
except Exception:
    _executor_sync_concurrency = 1
_EXECUTOR_SYNC_MAX_CONCURRENCY = _executor_sync_concurrency

proxy_url, action_server_token = env_util.require_all_env("RASA_PROXY_URL", "ACTION_SERVER_TOKEN")
graphql_target = env_util.require_any_env("RASA_PROXY_GRAPHQL_TARGET")
client = GraphQLProxyClient(
    proxy_url=proxy_url,
    action_server_token=action_server_token,
    target=graphql_target if isinstance(graphql_target, str) and graphql_target.strip() else "graphql",
)


METRIC_METADATA: Dict[str, Any] = get_metric_metadata()


def _emit_compiler_diagnostics(progress_cb: Optional[Callable[[str], None]], payload: Dict[str, Any]) -> None:
    logger.info("[plan_executor] compiler_diagnostics=%s", json.dumps(payload, default=str, sort_keys=True))
    if progress_cb is not None:
        progress_cb(f"Compiler diagnostics: {json.dumps(payload, default=str, sort_keys=True)}")


class VisualizationExecutionError(RuntimeError):
    def __init__(self, user_message: str, reason: str = "unknown"):
        super().__init__(user_message)
        self.user_message = user_message
        self.reason = reason


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


def _get_metric_meta(metric_code: str) -> Dict[str, Any]:
    code = (metric_code or "").upper()
    meta: Dict[str, Any] = METRIC_METADATA.get(code) or {}
    return meta


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


def execute_plan(plan: AnalysisPlan, user_sub: str) -> VisualizationResponse:
    """Sync wrapper that delegates to the async implementation with concurrency=1.

    - If no event loop is running, run the coroutine directly with asyncio.run.
    - If an event loop is already running, offload to a new thread and run a fresh loop there.
    """
    trace_id = plan.metadata.trace_id if getattr(plan, "metadata", None) is not None else None
    coro = execute_plan_async(plan, user_sub, max_concurrency=_EXECUTOR_SYNC_MAX_CONCURRENCY, trace_id=trace_id)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(asyncio.run, coro)
        return fut.result()


ProgressCallback = Callable[[str], None]
SummaryCallback = Callable[[ExecutionSummary], None]


@dataclass(frozen=True)
class ExecutionContext:
    user_sub: str
    semaphore: asyncio.Semaphore
    progress_cb: Optional[ProgressCallback]
    log_graphql_query: bool


def _emit_progress(context: ExecutionContext, completed: int, total: int, prefix: str = "Fetching data") -> None:
    if context.progress_cb is None:
        return
    if total > 0:
        context.progress_cb(f"{prefix} ({completed}/{total})")
    else:
        context.progress_cb(f"{prefix}…")


async def _execute_request_spec(
    spec: RequestSpec,
    request_failures: List[str],
    context: ExecutionContext,
) -> List[ChartSeries]:
    return await run_graphql_request(
        req=spec.req,
        label_parts=spec.label_parts,
        include_metric_alias=spec.include_metric_alias,
        group_by_field=spec.group_by_field,
        add_time_period_labels=spec.add_time_period_labels,
        request_failures=request_failures,
        client=client,
        user_sub=context.user_sub,
        semaphore=context.semaphore,
        log_graphql_query=context.log_graphql_query,
    )


async def _execute_specs_concurrent(
    specs: List[RequestSpec],
    request_failures: List[str],
    context: ExecutionContext,
    total_requests: int,
    progress_prefix: str = "Fetching data",
) -> List[ChartSeries]:
    _emit_progress(context, completed=0, total=total_requests, prefix=progress_prefix)
    if not specs:
        return []

    tasks = [asyncio.create_task(_execute_request_spec(spec=spec, request_failures=request_failures, context=context)) for spec in specs]

    all_series: List[ChartSeries] = []
    completed = 0
    for task in asyncio.as_completed(tasks):
        result = await task
        all_series.extend(result)
        completed += 1
        _emit_progress(context, completed=completed, total=total_requests, prefix=progress_prefix)

    return all_series


async def _execute_specs_sequential(
    specs: List[RequestSpec],
    request_failures: List[str],
    context: ExecutionContext,
    total_requests: int,
    progress_prefix: str,
) -> List[ChartSeries]:
    _emit_progress(context, completed=0, total=total_requests, prefix=progress_prefix)
    all_series: List[ChartSeries] = []
    completed = 0

    for spec in specs:
        result = await _execute_request_spec(spec=spec, request_failures=request_failures, context=context)
        all_series.extend(result)
        completed += 1
        _emit_progress(context, completed=completed, total=total_requests, prefix=progress_prefix)

    return all_series


async def execute_plan_async(
    plan: AnalysisPlan,
    user_sub: str,
    max_concurrency: Optional[int] = None,
    progress_cb: Optional[ProgressCallback] = None,
    summary_cb: Optional[SummaryCallback] = None,
    trace_id: Optional[str] = None,
) -> VisualizationResponse:
    """Async version that runs GraphQL requests concurrently.

    - Uses asyncio.to_thread to run the existing synchronous client in a thread pool.
    - Limits concurrency via a semaphore to avoid overloading the proxy/backend.
    - Produces one chart per canonical GroupBy (or one overall if none), matching sync behavior.
    """
    if not trace_id and getattr(plan, "metadata", None) is not None:
        trace_id = plan.metadata.trace_id

    plan, normalization_summary = normalize_analysis_plan_with_diagnostics(plan)
    plan_charts = coalesce(plan.charts, [])
    if plan.statistical_tests:
        logger.warning("Statistical tests are defined in the plan but not implemented in this executor yet. They will be ignored.")
    response: VisualizationResponse = VisualizationResponse()
    estimated_queries = estimate_query_count_for_plan(plan)
    actual_queries = 0
    summary_batches: List[ExecutionBatchSummary] = []

    resolved_concurrency = _EXECUTOR_DEFAULT_MAX_CONCURRENCY if max_concurrency is None else max(1, int(max_concurrency))
    sem = asyncio.Semaphore(resolved_concurrency)
    execution_context = ExecutionContext(
        user_sub=user_sub,
        semaphore=sem,
        progress_cb=progress_cb,
        log_graphql_query=_LOG_GRAPHQL_QUERY,
    )

    for planChart in plan_charts:
        metric_requests, derived_axes = build_metric_requests(
            plan_chart=planChart,
            derive_defaults_fn=_derive_distribution_defaults,
            axis_from_meta_fn=_axis_from_meta,
        )

        compiled_grouping = compile_chart_grouping(planChart)
        dims: List[Dimension] = compiled_grouping.dimensions

        for batch in compiled_grouping.batches:
            request_failures: List[str] = []
            filter_dims = batch.filter_dims
            batched_time_enabled = batch.batched_time_enabled
            batched_time_periods = batch.batched_time_periods
            combos_list = batch.combos_list

            total_requests = batch.request_count
            gb_field = batch.server_groupby
            actual_queries += total_requests

            summary_batches.append(
                make_batch_summary(
                    chart_title=planChart.title or "",
                    chart_type=planChart.chart_type,
                    server_groupby=gb_field,
                    filter_dimensions=[d.kind.__name__ for d in filter_dims],
                    batched_time_period_count=len(batched_time_periods) if batched_time_enabled else 0,
                    query_count=total_requests,
                )
            )

            if _EMIT_COMPILER_DIAGNOSTICS:
                _emit_compiler_diagnostics(
                    progress_cb,
                    {
                        "chart_title": planChart.title or "",
                        "chart_type": planChart.chart_type,
                        "server_groupby": gb_field,
                        "batched_time_enabled": batched_time_enabled,
                        "batched_time_period_count": len(batched_time_periods),
                        "filter_dimensions": [d.kind.__name__ for d in filter_dims],
                        "query_count_estimate": total_requests,
                    },
                )
            include_metric_alias = len(planChart.metrics) > 1

            chart_filter = to_gql_filter(coalesce(planChart.filters, None))

            primary_specs, combo_contexts = build_primary_request_specs(
                metric_requests=metric_requests,
                chart_filter=chart_filter,
                filter_dims=filter_dims,
                combos_list=combos_list,
                batched_time_enabled=batched_time_enabled,
                batched_time_periods=batched_time_periods,
                include_metric_alias=include_metric_alias,
                group_by_field=gb_field,
            )

            all_series = await _execute_specs_concurrent(
                specs=primary_specs,
                request_failures=request_failures,
                context=execution_context,
                total_requests=total_requests,
                progress_prefix="Fetching data",
            )

            if (
                _ENABLE_UNBATCHED_TIME_FALLBACK
                and not _STRICT_MODE
                and should_retry_unbatched_time(
                    all_series=all_series,
                    request_failures=request_failures,
                    batched_time_enabled=batched_time_enabled,
                    batched_time_periods=batched_time_periods,
                )
            ):
                logger.warning(
                    "[plan_executor] Batched multi-period request timed out; retrying with per-period requests (period_count=%s, combos=%s)",
                    len(batched_time_periods),
                    len(combo_contexts),
                )
                request_failures.clear()
                all_series = []

                fallback_specs = build_fallback_request_specs(
                    metric_requests=metric_requests,
                    combo_contexts=combo_contexts,
                    batched_time_periods=batched_time_periods,
                )

                retry_count = max(1, len(fallback_specs))
                actual_queries += retry_count
                all_series = await _execute_specs_sequential(
                    specs=fallback_specs,
                    request_failures=request_failures,
                    context=execution_context,
                    total_requests=retry_count,
                    progress_prefix="Retrying with per-period requests",
                )

            all_series = merge_series_by_name(all_series)

            if not all_series:
                logger.warning(
                    "[plan_executor] No series generated for chart '%s'%s. This often indicates a backend error or empty results.",
                    planChart.title or "Chart",
                    "",
                )
                if request_failures:
                    raise _to_execution_error(request_failures)
            vis_chart = build_chart_dto(
                plan_chart=planChart,
                dimensions=dims,
                series=all_series,
                derived_axes=derived_axes,
            )
            response.charts.append(vis_chart)

    if summary_cb is not None:
        payload = make_execution_summary(
            trace_id=trace_id,
            chart_count=len(plan_charts),
            estimated_queries=estimated_queries,
            actual_queries=actual_queries,
            batches=summary_batches,
            normalization=normalization_summary,
        )
        try:
            summary_cb(payload)
        except Exception:
            logger.debug("Failed to emit execution summary callback", exc_info=True)

    return response
