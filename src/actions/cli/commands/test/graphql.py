from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, cast

from rasa_sdk.events import EventType  # type: ignore

from src.domain.graphql.request import (
    DataOrigin,
    GraphQLQueryRequest,
    MetricRequest,
    TimePeriod,
)
from src.executors.graphql.client import GraphQLProxyClient
from src.util import env as env_util

from .. import command

# Privacy/safety default: don't log raw query strings.
_LOG_CLI_GRAPHQL_QUERY = env_util.env_flag("CLI_LOG_GRAPHQL_QUERY", default=False)


@command("test_gql")
def test_graphql(dispatcher: Any, tracker: Any, domain: Any, args: List[str], opts: Dict[str, Any]) -> List[EventType]:
    """Run a minimal GraphQL smoke test via the proxy and report status."""
    logger = logging.getLogger(__name__)
    try:
        proxy_url, graphql_url = env_util.require_all_env("GRAPHQL_PROXY_URL", "GRAPHQL_API_URL")
        session_token = tracker.sender_id
        provider_ids: List[int] = [1]

        end_dt = datetime.now(timezone.utc).date()
        start_dt = end_dt - timedelta(days=30)

        metric = MetricRequest(metricType=cast(Any, "AGE")).with_stats()
        gql_req = GraphQLQueryRequest(
            metrics=[metric],
            timePeriod=TimePeriod(startDate=start_dt.isoformat(), endDate=end_dt.isoformat()),
            dataOrigin=DataOrigin(providerGroupId=provider_ids),
            includeGeneralStats=True,
        )

        query_str = gql_req.to_graphql_string()
        q_hash = hashlib.sha256(query_str.encode("utf-8")).hexdigest()[:12]
        if _LOG_CLI_GRAPHQL_QUERY:
            logger.debug("GraphQL CLI test query (hash=%s): %s", q_hash, query_str)
        else:
            logger.debug("GraphQL CLI test query prepared (hash=%s, len=%s)", q_hash, len(query_str))

        client = GraphQLProxyClient(proxy_url=proxy_url, graphql_url=graphql_url)
        result = client.query(query_str=query_str, session_token=session_token)

        if result is None:
            dispatcher.utter_message(text="❌ GraphQL test request failed (no response or non-200). Check logs for details.")
            return []

        # If GraphQL returned errors, surface the first one clearly
        errs_any: Any = getattr(result, "errors", None)
        if isinstance(errs_any, list) and errs_any and isinstance(errs_any[0], dict):
            first = cast(Dict[str, Any], errs_any[0])
            msg = first.get("message")
            path_val = first.get("path")
            text = "❌ GraphQL returned errors"
            if isinstance(msg, str):
                text += f": {msg}"
            if isinstance(path_val, list):
                try:
                    path_list = cast(List[Any], path_val)
                    parts = [str(p) for p in path_list]
                    path_str = ".".join(parts)
                    text += f" (path: {path_str})"
                except Exception:
                    pass
            dispatcher.utter_message(text=text)
            return []
        elif errs_any:
            dispatcher.utter_message(text="❌ GraphQL returned errors (details unavailable).")
            return []

        # If no data and no explicit errors
        if getattr(result, "data", None) is None:
            dispatcher.utter_message(text="⚠️ GraphQL returned no data (data: null) and no explicit errors.")
            return []

        # Success summary: include basic general stats when available
        summary = "✅ GraphQL test OK"
        try:
            data_obj = getattr(result, "data", None)
            gm = getattr(data_obj, "get_metrics", None) if data_obj is not None else None
            gs = getattr(gm, "general_stats_group", None) if gm is not None else None
            stats = None
            if isinstance(gs, list):
                gs_list = cast(List[Any], gs)
                if gs_list:
                    stats = getattr(gs_list[0], "general_statistics", None)
            elif gs is not None:
                stats = getattr(gs, "general_statistics", None)
            if stats is not None:
                cases = getattr(stats, "cases_in_period", None)
                filtered = getattr(stats, "filtered_cases_in_period", None)
                info_parts: List[str] = []
                if cases is not None:
                    info_parts.append(f"casesInPeriod={cases}")
                if filtered is not None:
                    info_parts.append(f"filteredCasesInPeriod={filtered}")
                if info_parts:
                    summary += " | " + " ".join(info_parts)
        except Exception:
            pass
        dispatcher.utter_message(text=summary)
    except Exception as e:
        logger.exception("GraphQL CLI test error: %s", e)
        dispatcher.utter_message(text=f"❌ GraphQL test error: {e}")
    return []
