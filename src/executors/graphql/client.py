import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, TypedDict, cast
from urllib.parse import urlsplit

import requests

import src.domain.graphql.response as gqlr
from src.util import env as env_util
from src.util.logging_utils import log_context


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


def _proxy_endpoint_label(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    if parsed.netloc:
        return parsed.netloc
    return url


def _mapping_to_dict(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}

    mapping = cast(Mapping[object, object], value)
    result: Dict[str, Any] = {}
    for raw_key, raw_value in mapping.items():
        if isinstance(raw_key, str):
            result[raw_key] = raw_value
    return result


def _elapsed_ms(started_at: float) -> int:
    return max(0, int((time.monotonic() - started_at) * 1000))


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
        connect_timeout_seconds: float = 5.0,
        max_total_timeout_seconds: Optional[float] = None,
        retry_attempts: int = 2,
        retry_backoff_seconds: float = 0.6,
    ):
        self.proxy_url = proxy_url
        self.action_server_token = action_server_token
        self.target = target
        self.path = path
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.connect_timeout_seconds = max(1.0, float(connect_timeout_seconds))
        self.retry_attempts = max(0, int(retry_attempts))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        default_budget = self.timeout_seconds * (self.retry_attempts + 1)
        if self.retry_attempts > 0:
            default_budget += self.retry_backoff_seconds * ((2**self.retry_attempts) - 1)
        if max_total_timeout_seconds is None:
            self.max_total_timeout_seconds = default_budget
        else:
            self.max_total_timeout_seconds = max(self.timeout_seconds, float(max_total_timeout_seconds))

    @staticmethod
    def _is_transient_status(status_code: int) -> bool:
        return status_code in {408, 429, 500, 502, 503, 504, 524}

    def _sleep_before_retry(self, attempt: int, remaining_budget: Optional[float] = None) -> None:
        delay = self.retry_backoff_seconds * (2**attempt)
        if remaining_budget is not None:
            delay = min(delay, max(0.0, remaining_budget))
        if delay > 0:
            time.sleep(delay)

    @staticmethod
    def _require_trace_id(trace_id: str, operation: str) -> str:
        token = (trace_id or "").strip()
        if not token:
            raise ValueError(f"trace_id is required for {operation}")
        return token

    @staticmethod
    def _transport_log_extra(
        *,
        attempt: int,
        attempts_total: int,
        retry: bool,
        error_category: str,
        exc: Exception,
        started_at: float,
    ) -> Dict[str, Dict[str, Any]]:
        return {
            "log_context": {
                "attempt": attempt + 1,
                "attempts_total": attempts_total,
                "retry": retry,
                "error_category": error_category,
                "error_type": type(exc).__name__,
                "elapsed_ms": _elapsed_ms(started_at),
            }
        }

    def _log_transport_failure(
        self,
        *,
        operation: str,
        label: str,
        error_category: str,
        exc: Exception,
        attempt: int,
        attempts_total: int,
        started_at: float,
        retry: bool,
    ) -> None:
        extra = self._transport_log_extra(
            attempt=attempt,
            attempts_total=attempts_total,
            retry=retry,
            error_category=error_category,
            exc=exc,
            started_at=started_at,
        )
        if retry:
            logger.warning(
                "[GraphQLProxyClient] %s during %s (attempt=%s/%s): %s",
                label,
                operation,
                attempt + 1,
                attempts_total,
                exc,
                extra=extra,
            )
            return
        logger.error(
            "[GraphQLProxyClient] %s during %s after %s attempt(s): %s",
            label,
            operation,
            attempt + 1,
            exc,
            extra=extra,
        )

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
        started_at = time.monotonic()

        def parse_error_payload() -> Dict[str, Any]:
            try:
                payload_any: Any = response.json()
            except ValueError:
                logger.debug(
                    "[GraphQLProxyClient] Failed to parse proxy error payload as JSON",
                    extra={
                        "log_context": {
                            "event": "graphql.error_payload_parse_failed",
                            "operation": "query",
                            "outcome": "degraded",
                            "status_code": response.status_code,
                            "body_len": len(response.text or ""),
                            "elapsed_ms": _elapsed_ms(started_at),
                        }
                    },
                    exc_info=True,
                )
                return {}
            return _mapping_to_dict(payload_any)

        with log_context(
            trace_id=trace_label,
            user_sub=user_sub,
            graphql_target=self.target,
            graphql_path=self.path,
            graphql_hash=q_hash,
            graphql_operation="query",
            proxy_endpoint=_proxy_endpoint_label(self.proxy_url),
        ):
            for attempt in range(attempts_total):
                elapsed = time.monotonic() - started_at
                remaining_budget = self.max_total_timeout_seconds - elapsed
                if remaining_budget <= 0:
                    last_error = GraphQLProxyError(
                        kind="timeout",
                        message=f"GraphQL request exceeded total timeout budget ({self.max_total_timeout_seconds:.1f}s)",
                        transient=True,
                    )
                    logger.warning(
                        "[GraphQLProxyClient] Budget exhausted before attempt (attempt=%s/%s, budget=%.1fs)",
                        attempt + 1,
                        attempts_total,
                        self.max_total_timeout_seconds,
                    )
                    break
                read_timeout = max(1.0, min(self.timeout_seconds, remaining_budget))
                try:
                    logger.debug(
                        "[GraphQLProxyClient] Outbound request (attempt=%s/%s, timeout=(%.1f, %.1f), budget_left=%.1fs)",
                        attempt + 1,
                        attempts_total,
                        self.connect_timeout_seconds,
                        read_timeout,
                        remaining_budget,
                    )
                    response = requests.post(
                        self.proxy_url,
                        headers=headers,
                        json=proxy_payload,
                        timeout=(self.connect_timeout_seconds, read_timeout),
                    )
                    logger.debug(
                        "[GraphQLProxyClient] Outbound response (attempt=%s/%s, status=%s)",
                        attempt + 1,
                        attempts_total,
                        response.status_code,
                        extra={"log_context": {"elapsed_ms": _elapsed_ms(started_at)}},
                    )
                except requests.Timeout as exc:
                    last_error = GraphQLProxyError(
                        kind="timeout",
                        message="GraphQL request timed out",
                        transient=True,
                    )
                    should_retry = attempt < self.retry_attempts
                    self._log_transport_failure(
                        operation="request",
                        label="Timeout",
                        error_category="timeout",
                        exc=exc,
                        attempt=attempt,
                        attempts_total=attempts_total,
                        started_at=started_at,
                        retry=should_retry,
                    )
                    if should_retry:
                        retry_budget = self.max_total_timeout_seconds - (time.monotonic() - started_at)
                        if retry_budget <= 0:
                            break
                        self._sleep_before_retry(attempt, remaining_budget=retry_budget)
                        continue
                    break
                except requests.ConnectionError as exc:
                    last_error = GraphQLProxyError(
                        kind="request_error",
                        message="GraphQL request failed",
                        transient=True,
                    )
                    should_retry = attempt < self.retry_attempts
                    self._log_transport_failure(
                        operation="request",
                        label="Connection failure",
                        error_category="connection_error",
                        exc=exc,
                        attempt=attempt,
                        attempts_total=attempts_total,
                        started_at=started_at,
                        retry=should_retry,
                    )
                    if should_retry:
                        retry_budget = self.max_total_timeout_seconds - (time.monotonic() - started_at)
                        if retry_budget <= 0:
                            break
                        self._sleep_before_retry(attempt, remaining_budget=retry_budget)
                        continue
                    break
                except requests.RequestException as exc:
                    last_error = GraphQLProxyError(
                        kind="request_error",
                        message="GraphQL request failed",
                        transient=True,
                    )
                    should_retry = attempt < self.retry_attempts
                    self._log_transport_failure(
                        operation="request",
                        label="Request exception",
                        error_category="request_error",
                        exc=exc,
                        attempt=attempt,
                        attempts_total=attempts_total,
                        started_at=started_at,
                        retry=should_retry,
                    )
                    if should_retry:
                        retry_budget = self.max_total_timeout_seconds - (time.monotonic() - started_at)
                        if retry_budget <= 0:
                            break
                        self._sleep_before_retry(attempt, remaining_budget=retry_budget)
                        continue
                    break

                if response.status_code == 200:
                    try:
                        return gqlr.MetricsQueryResponse.model_validate(response.json())
                    except Exception as exc:
                        if _LOG_GRAPHQL_BODY:
                            logger.error(
                                "[GraphQLProxyClient] Validation error: %s. Raw: %s",
                                exc,
                                response.text,
                                extra={
                                    "log_context": {
                                        "event": "graphql.invalid_response_validation",
                                        "operation": "query",
                                        "outcome": "failure",
                                        "status_code": 200,
                                        "body_len": len(response.text or ""),
                                        "elapsed_ms": _elapsed_ms(started_at),
                                    }
                                },
                            )
                        else:
                            logger.error(
                                "[GraphQLProxyClient] Validation error: %s (body_len=%s)",
                                exc,
                                len(response.text or ""),
                                extra={
                                    "log_context": {
                                        "event": "graphql.invalid_response_validation",
                                        "operation": "query",
                                        "outcome": "failure",
                                        "status_code": 200,
                                        "body_len": len(response.text or ""),
                                        "elapsed_ms": _elapsed_ms(started_at),
                                    }
                                },
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
                error_payload = parse_error_payload()
                proxy_info = _mapping_to_dict(error_payload.get("proxy"))
                proxy_reason_any = proxy_info.get("reason")
                proxy_reason = proxy_reason_any.strip() if isinstance(proxy_reason_any, str) and proxy_reason_any.strip() else None
                log_context_fields: Dict[str, Any] = {
                    "status_code": response.status_code,
                    "attempt": attempt + 1,
                    "attempts_total": attempts_total,
                    "content_type": content_type,
                    "body_len": len(response.text or ""),
                    "query_len": len(query_str),
                    "proxy_reason": proxy_reason or "-",
                    "elapsed_ms": _elapsed_ms(started_at),
                }
                if _LOG_GRAPHQL_BODY:
                    log_context_fields["body_preview"] = response.text[:1000]
                if _LOG_GRAPHQL_QUERY:
                    log_context_fields["query_preview"] = query_str[:300]

                logger.error(
                    "[GraphQLProxyClient] Proxy error %s during %s (attempt=%s/%s, content_type=%s, body_len=%s)",
                    response.status_code,
                    "query",
                    attempt + 1,
                    attempts_total,
                    content_type,
                    len(response.text or ""),
                    extra={"log_context": log_context_fields},
                )

                last_error = GraphQLProxyError(
                    kind="http_error",
                    message=f"GraphQL proxy returned HTTP {response.status_code}",
                    status_code=response.status_code,
                    transient=transient_status,
                )
                if transient_status and attempt < self.retry_attempts:
                    retry_budget = self.max_total_timeout_seconds - (time.monotonic() - started_at)
                    if retry_budget <= 0:
                        break
                    self._sleep_before_retry(attempt, remaining_budget=retry_budget)
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
        started_at = time.monotonic()

        def parse_error_payload() -> Dict[str, Any]:
            try:
                payload_any: Any = response.json()
            except ValueError:
                logger.debug(
                    "[GraphQLProxyClient] Failed to parse proxy error payload as JSON",
                    extra={
                        "log_context": {
                            "event": "graphql.error_payload_parse_failed",
                            "operation": "query_raw",
                            "outcome": "degraded",
                            "status_code": response.status_code,
                            "body_len": len(response.text or ""),
                            "elapsed_ms": _elapsed_ms(started_at),
                        }
                    },
                    exc_info=True,
                )
                return {}
            return _mapping_to_dict(payload_any)

        with log_context(
            trace_id=trace_label,
            user_sub=user_sub,
            graphql_target=self.target,
            graphql_path=self.path,
            graphql_hash=q_hash,
            graphql_operation="query_raw",
            proxy_endpoint=_proxy_endpoint_label(self.proxy_url),
        ):
            for attempt in range(attempts_total):
                elapsed = time.monotonic() - started_at
                remaining_budget = self.max_total_timeout_seconds - elapsed
                if remaining_budget <= 0:
                    last_error = GraphQLProxyError(
                        kind="timeout",
                        message=f"GraphQL request exceeded total timeout budget ({self.max_total_timeout_seconds:.1f}s)",
                        transient=True,
                    )
                    logger.warning(
                        "[GraphQLProxyClient] Budget exhausted before raw attempt (attempt=%s/%s, budget=%.1fs)",
                        attempt + 1,
                        attempts_total,
                        self.max_total_timeout_seconds,
                    )
                    break
                read_timeout = max(1.0, min(self.timeout_seconds, remaining_budget))
                try:
                    logger.debug(
                        "[GraphQLProxyClient] Outbound raw request (attempt=%s/%s, timeout=(%.1f, %.1f), budget_left=%.1fs)",
                        attempt + 1,
                        attempts_total,
                        self.connect_timeout_seconds,
                        read_timeout,
                        remaining_budget,
                    )
                    response = requests.post(
                        self.proxy_url,
                        headers=headers,
                        json=proxy_payload,
                        timeout=(self.connect_timeout_seconds, read_timeout),
                    )
                    logger.debug(
                        "[GraphQLProxyClient] Outbound raw response (attempt=%s/%s, status=%s)",
                        attempt + 1,
                        attempts_total,
                        response.status_code,
                        extra={"log_context": {"elapsed_ms": _elapsed_ms(started_at)}},
                    )
                except requests.Timeout as exc:
                    last_error = GraphQLProxyError(
                        kind="timeout",
                        message="GraphQL request timed out",
                        transient=True,
                    )
                    should_retry = attempt < self.retry_attempts
                    self._log_transport_failure(
                        operation="raw request",
                        label="Timeout",
                        error_category="timeout",
                        exc=exc,
                        attempt=attempt,
                        attempts_total=attempts_total,
                        started_at=started_at,
                        retry=should_retry,
                    )
                    if should_retry:
                        retry_budget = self.max_total_timeout_seconds - (time.monotonic() - started_at)
                        if retry_budget <= 0:
                            break
                        self._sleep_before_retry(attempt, remaining_budget=retry_budget)
                        continue
                    break
                except requests.ConnectionError as exc:
                    last_error = GraphQLProxyError(
                        kind="request_error",
                        message="GraphQL request failed",
                        transient=True,
                    )
                    should_retry = attempt < self.retry_attempts
                    self._log_transport_failure(
                        operation="raw request",
                        label="Connection failure",
                        error_category="connection_error",
                        exc=exc,
                        attempt=attempt,
                        attempts_total=attempts_total,
                        started_at=started_at,
                        retry=should_retry,
                    )
                    if should_retry:
                        retry_budget = self.max_total_timeout_seconds - (time.monotonic() - started_at)
                        if retry_budget <= 0:
                            break
                        self._sleep_before_retry(attempt, remaining_budget=retry_budget)
                        continue
                    break
                except requests.RequestException as exc:
                    last_error = GraphQLProxyError(
                        kind="request_error",
                        message="GraphQL request failed",
                        transient=True,
                    )
                    should_retry = attempt < self.retry_attempts
                    self._log_transport_failure(
                        operation="raw request",
                        label="Request exception",
                        error_category="request_error",
                        exc=exc,
                        attempt=attempt,
                        attempts_total=attempts_total,
                        started_at=started_at,
                        retry=should_retry,
                    )
                    if should_retry:
                        retry_budget = self.max_total_timeout_seconds - (time.monotonic() - started_at)
                        if retry_budget <= 0:
                            break
                        self._sleep_before_retry(attempt, remaining_budget=retry_budget)
                        continue
                    break

                if response.status_code == 200:
                    try:
                        payload = _mapping_to_dict(response.json())
                        if payload:
                            return payload
                        last_error = GraphQLProxyError(
                            kind="invalid_response",
                            message="GraphQL response is not a JSON object",
                            status_code=200,
                            transient=False,
                        )
                        break
                    except Exception as exc:
                        if _LOG_GRAPHQL_BODY:
                            logger.error(
                                "[GraphQLProxyClient] JSON parse error: %s. Raw: %s",
                                exc,
                                response.text,
                                extra={
                                    "log_context": {
                                        "event": "graphql.invalid_response_json",
                                        "operation": "query_raw",
                                        "outcome": "failure",
                                        "status_code": 200,
                                        "body_len": len(response.text or ""),
                                        "elapsed_ms": _elapsed_ms(started_at),
                                    }
                                },
                            )
                        else:
                            logger.error(
                                "[GraphQLProxyClient] JSON parse error: %s (body_len=%s)",
                                exc,
                                len(response.text or ""),
                                extra={
                                    "log_context": {
                                        "event": "graphql.invalid_response_json",
                                        "operation": "query_raw",
                                        "outcome": "failure",
                                        "status_code": 200,
                                        "body_len": len(response.text or ""),
                                        "elapsed_ms": _elapsed_ms(started_at),
                                    }
                                },
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
                error_payload = parse_error_payload()
                proxy_info = _mapping_to_dict(error_payload.get("proxy"))
                proxy_reason_any = proxy_info.get("reason")
                proxy_reason = proxy_reason_any.strip() if isinstance(proxy_reason_any, str) and proxy_reason_any.strip() else None
                log_context_fields: Dict[str, Any] = {
                    "status_code": response.status_code,
                    "attempt": attempt + 1,
                    "attempts_total": attempts_total,
                    "content_type": content_type,
                    "body_len": len(response.text or ""),
                    "query_len": len(query_str),
                    "proxy_reason": proxy_reason or "-",
                    "elapsed_ms": _elapsed_ms(started_at),
                }
                if _LOG_GRAPHQL_BODY:
                    log_context_fields["body_preview"] = response.text[:1000]
                if _LOG_GRAPHQL_QUERY:
                    log_context_fields["query_preview"] = query_str[:300]

                logger.error(
                    "[GraphQLProxyClient] Proxy error %s during %s (attempt=%s/%s, content_type=%s, body_len=%s)",
                    response.status_code,
                    "query_raw",
                    attempt + 1,
                    attempts_total,
                    content_type,
                    len(response.text or ""),
                    extra={"log_context": log_context_fields},
                )

                last_error = GraphQLProxyError(
                    kind="http_error",
                    message=f"GraphQL proxy returned HTTP {response.status_code}",
                    status_code=response.status_code,
                    transient=transient_status,
                )
                if transient_status and attempt < self.retry_attempts:
                    retry_budget = self.max_total_timeout_seconds - (time.monotonic() - started_at)
                    if retry_budget <= 0:
                        break
                    self._sleep_before_retry(attempt, remaining_budget=retry_budget)
                    continue
                break

        if raise_on_error and last_error is not None:
            raise last_error
        return None
