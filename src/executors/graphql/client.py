import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, TypedDict

import requests

import src.domain.graphql.response as gqlr
from src.util import env as env_util


class GraphQLPayload(TypedDict):
    query: str
    variables: Dict[str, Any]


class ProxyHttpRequestPayload(TypedDict):
    path: str
    method: str
    body: GraphQLPayload


class ProxyRequestPayload(TypedDict):
    senderId: str
    target: str
    request: ProxyHttpRequestPayload


logger = logging.getLogger(__name__)
# Privacy/safety defaults: avoid logging raw GraphQL payloads.
_LOG_GRAPHQL_QUERY = env_util.env_flag("GRAPHQL_LOG_QUERY", default=False)
_LOG_GRAPHQL_BODY = env_util.env_flag("GRAPHQL_LOG_BODY", default=False)


@dataclass
class GraphQLProxyError(Exception):
    kind: str
    message: str
    status_code: Optional[int] = None
    transient: bool = False

    def __str__(self) -> str:
        return self.message


class GraphQLProxyClient:
    def __init__(
        self,
        proxy_url: str,
        action_server_token: str,
        target: str = "graphql",
        path: str = "/api/graphql/aggregation",
        timeout_seconds: int = 30,
        retry_attempts: int = 2,
        retry_backoff_seconds: float = 0.6,
    ):
        self.proxy_url = proxy_url
        self.action_server_token = action_server_token
        self.target = target
        self.path = path
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = max(0, int(retry_attempts))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))

    @staticmethod
    def _is_transient_status(status_code: int) -> bool:
        return status_code in {408, 429, 500, 502, 503, 504}

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = self.retry_backoff_seconds * (2**attempt)
        if delay > 0:
            time.sleep(delay)

    @staticmethod
    def _require_trace_id(trace_id: str, operation: str) -> str:
        token = (trace_id or "").strip()
        if not token:
            raise ValueError(f"trace_id is required for {operation}")
        return token

    def query(
        self,
        query_str: str,
        user_sub: str,
        trace_id: str,
        variables: Optional[Dict[str, Any]] = None,
        raise_on_error: bool = False,
    ) -> gqlr.MetricsQueryResponse | None:
        trace_label = self._require_trace_id(trace_id, "query")
        headers = {
            "Content-Type": "application/json",
            "x-action-server-token": self.action_server_token,
        }
        headers["x-trace-id"] = trace_label

        proxy_payload: ProxyRequestPayload = {
            # Preserve the exact Rasa sender_id value for proxy token lookup.
            "senderId": user_sub,
            "target": self.target,
            "request": {
                "path": self.path,
                "method": "POST",
                "body": {"query": query_str, "variables": variables or {}},
            },
        }

        q_hash = hashlib.sha256(query_str.encode("utf-8")).hexdigest()[:12]
        attempts_total = self.retry_attempts + 1
        last_error: Optional[GraphQLProxyError] = None

        for attempt in range(attempts_total):
            try:
                logger.debug(
                    "[GraphQLProxyClient] Outbound request (trace_id=%s, attempt=%s/%s, hash=%s, target=%s)",
                    trace_label,
                    attempt + 1,
                    attempts_total,
                    q_hash,
                    self.target,
                )
                response = requests.post(
                    self.proxy_url,
                    headers=headers,
                    json=proxy_payload,
                    timeout=self.timeout_seconds,
                )
                logger.info(
                    "[GraphQLProxyClient] Outbound response (trace_id=%s, attempt=%s/%s, hash=%s, status=%s)",
                    trace_label,
                    attempt + 1,
                    attempts_total,
                    q_hash,
                    response.status_code,
                )
            except requests.Timeout as exc:
                last_error = GraphQLProxyError(
                    kind="timeout",
                    message="GraphQL request timed out",
                    transient=True,
                )
                logger.warning(
                    "[GraphQLProxyClient] Timeout (trace_id=%s, attempt=%s/%s, hash=%s): %s",
                    trace_label,
                    attempt + 1,
                    attempts_total,
                    q_hash,
                    exc,
                )
                if attempt < self.retry_attempts:
                    self._sleep_before_retry(attempt)
                    continue
                break
            except requests.RequestException as exc:
                last_error = GraphQLProxyError(
                    kind="request_error",
                    message="GraphQL request failed",
                    transient=True,
                )
                logger.warning(
                    "[GraphQLProxyClient] Request exception (trace_id=%s, attempt=%s/%s, hash=%s): %s",
                    trace_label,
                    attempt + 1,
                    attempts_total,
                    q_hash,
                    exc,
                )
                if attempt < self.retry_attempts:
                    self._sleep_before_retry(attempt)
                    continue
                break

            if response.status_code == 200:
                try:
                    return gqlr.MetricsQueryResponse.model_validate(response.json())
                except Exception as exc:
                    if _LOG_GRAPHQL_BODY:
                        logger.error("[GraphQLProxyClient] Validation error (trace_id=%s, hash=%s): %s. Raw: %s", trace_label, q_hash, exc, response.text)
                    else:
                        logger.error(
                            "[GraphQLProxyClient] Validation error (trace_id=%s, hash=%s): %s (body_len=%s)",
                            trace_label,
                            q_hash,
                            exc,
                            len(response.text or ""),
                        )
                    last_error = GraphQLProxyError(
                        kind="invalid_response",
                        message="GraphQL response validation failed",
                        status_code=200,
                        transient=False,
                    )
                    break

            content_type = response.headers.get("Content-Type", "")
            transient_status = self._is_transient_status(response.status_code)
            if _LOG_GRAPHQL_BODY or _LOG_GRAPHQL_QUERY:
                body_preview = response.text[:1000] if _LOG_GRAPHQL_BODY else "(body logging disabled)"
                query_preview = query_str[:300] if _LOG_GRAPHQL_QUERY else "(query logging disabled)"
                logger.error(
                    "[GraphQLProxyClient] Proxy error %s (trace_id=%s, attempt=%s/%s, Content-Type=%s, target=%s, hash=%s). Body preview: %s. Query preview: %s",
                    response.status_code,
                    trace_label,
                    attempt + 1,
                    attempts_total,
                    content_type,
                    self.target,
                    q_hash,
                    body_preview,
                    query_preview,
                )
            else:
                logger.error(
                    "[GraphQLProxyClient] Proxy error %s (trace_id=%s, attempt=%s/%s, Content-Type=%s, target=%s, hash=%s, body_len=%s, query_len=%s)",
                    response.status_code,
                    trace_label,
                    attempt + 1,
                    attempts_total,
                    content_type,
                    self.target,
                    q_hash,
                    len(response.text or ""),
                    len(query_str),
                )

            last_error = GraphQLProxyError(
                kind="http_error",
                message=f"GraphQL proxy returned HTTP {response.status_code}",
                status_code=response.status_code,
                transient=transient_status,
            )
            if transient_status and attempt < self.retry_attempts:
                self._sleep_before_retry(attempt)
                continue
            break

        if raise_on_error and last_error is not None:
            raise last_error
        return None

    def query_raw(
        self,
        query_str: str,
        user_sub: str,
        trace_id: str,
        variables: Optional[Dict[str, Any]] = None,
        raise_on_error: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Execute a GraphQL query and return the raw JSON object.

        This is useful for endpoints whose shape differs from getMetrics, such
        as statistical test queries.
        """
        trace_label = self._require_trace_id(trace_id, "query_raw")
        headers = {
            "Content-Type": "application/json",
            "x-action-server-token": self.action_server_token,
        }
        headers["x-trace-id"] = trace_label

        proxy_payload: ProxyRequestPayload = {
            # Preserve the exact Rasa sender_id value for proxy token lookup.
            "senderId": user_sub,
            "target": self.target,
            "request": {
                "path": self.path,
                "method": "POST",
                "body": {"query": query_str, "variables": variables or {}},
            },
        }

        q_hash = hashlib.sha256(query_str.encode("utf-8")).hexdigest()[:12]
        attempts_total = self.retry_attempts + 1
        last_error: Optional[GraphQLProxyError] = None

        for attempt in range(attempts_total):
            try:
                logger.debug(
                    "[GraphQLProxyClient] Outbound raw request (trace_id=%s, attempt=%s/%s, hash=%s, target=%s)",
                    trace_label,
                    attempt + 1,
                    attempts_total,
                    q_hash,
                    self.target,
                )
                response = requests.post(
                    self.proxy_url,
                    headers=headers,
                    json=proxy_payload,
                    timeout=self.timeout_seconds,
                )
                logger.info(
                    "[GraphQLProxyClient] Outbound raw response (trace_id=%s, attempt=%s/%s, hash=%s, status=%s)",
                    trace_label,
                    attempt + 1,
                    attempts_total,
                    q_hash,
                    response.status_code,
                )
            except requests.Timeout as exc:
                last_error = GraphQLProxyError(
                    kind="timeout",
                    message="GraphQL request timed out",
                    transient=True,
                )
                logger.warning(
                    "[GraphQLProxyClient] Timeout (trace_id=%s, attempt=%s/%s, hash=%s): %s",
                    trace_label,
                    attempt + 1,
                    attempts_total,
                    q_hash,
                    exc,
                )
                if attempt < self.retry_attempts:
                    self._sleep_before_retry(attempt)
                    continue
                break
            except requests.RequestException as exc:
                last_error = GraphQLProxyError(
                    kind="request_error",
                    message="GraphQL request failed",
                    transient=True,
                )
                logger.warning(
                    "[GraphQLProxyClient] Request exception (trace_id=%s, attempt=%s/%s, hash=%s): %s",
                    trace_label,
                    attempt + 1,
                    attempts_total,
                    q_hash,
                    exc,
                )
                if attempt < self.retry_attempts:
                    self._sleep_before_retry(attempt)
                    continue
                break

            if response.status_code == 200:
                try:
                    payload_any = response.json()
                    if isinstance(payload_any, dict):
                        return payload_any
                    last_error = GraphQLProxyError(
                        kind="invalid_response",
                        message="GraphQL response is not a JSON object",
                        status_code=200,
                        transient=False,
                    )
                    break
                except Exception as exc:
                    if _LOG_GRAPHQL_BODY:
                        logger.error("[GraphQLProxyClient] JSON parse error (trace_id=%s, hash=%s): %s. Raw: %s", trace_label, q_hash, exc, response.text)
                    else:
                        logger.error(
                            "[GraphQLProxyClient] JSON parse error (trace_id=%s, hash=%s): %s (body_len=%s)",
                            trace_label,
                            q_hash,
                            exc,
                            len(response.text or ""),
                        )
                    last_error = GraphQLProxyError(
                        kind="invalid_response",
                        message="GraphQL response JSON parsing failed",
                        status_code=200,
                        transient=False,
                    )
                    break

            content_type = response.headers.get("Content-Type", "")
            transient_status = self._is_transient_status(response.status_code)
            if _LOG_GRAPHQL_BODY or _LOG_GRAPHQL_QUERY:
                body_preview = response.text[:1000] if _LOG_GRAPHQL_BODY else "(body logging disabled)"
                query_preview = query_str[:300] if _LOG_GRAPHQL_QUERY else "(query logging disabled)"
                logger.error(
                    "[GraphQLProxyClient] Proxy error %s (trace_id=%s, attempt=%s/%s, Content-Type=%s, target=%s, hash=%s). Body preview: %s. Query preview: %s",
                    response.status_code,
                    trace_label,
                    attempt + 1,
                    attempts_total,
                    content_type,
                    self.target,
                    q_hash,
                    body_preview,
                    query_preview,
                )
            else:
                logger.error(
                    "[GraphQLProxyClient] Proxy error %s (trace_id=%s, attempt=%s/%s, Content-Type=%s, target=%s, hash=%s, body_len=%s, query_len=%s)",
                    response.status_code,
                    trace_label,
                    attempt + 1,
                    attempts_total,
                    content_type,
                    self.target,
                    q_hash,
                    len(response.text or ""),
                    len(query_str),
                )

            last_error = GraphQLProxyError(
                kind="http_error",
                message=f"GraphQL proxy returned HTTP {response.status_code}",
                status_code=response.status_code,
                transient=transient_status,
            )
            if transient_status and attempt < self.retry_attempts:
                self._sleep_before_retry(attempt)
                continue
            break

        if raise_on_error and last_error is not None:
            raise last_error
        return None
