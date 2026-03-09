import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict, cast

import requests

from src.util import env as env_util

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsCenterError(Exception):
    kind: str
    message: str
    status_code: Optional[int] = None
    transient: bool = False

    def __str__(self) -> str:
        return self.message


class ProxyHttpRequestPayload(TypedDict):
    path: str
    method: str
    query: Dict[str, Any]


class ProxyRequestPayload(TypedDict):
    userSub: str
    target: str
    request: ProxyHttpRequestPayload


class ProviderCollectionResult(TypedDict):
    results: List[Dict[str, Any]]
    count: int
    limit: int
    offset: int


class CountryCollectionResult(TypedDict):
    results: List[Dict[str, Any]]


class AnalyticsCenterClient:
    def __init__(
        self,
        proxy_url: str,
        action_server_token: str,
        target: str = "analytics",
        timeout_seconds: int = 30,
        retry_attempts: int = 2,
        retry_backoff_seconds: float = 0.6,
    ):
        self.proxy_url = proxy_url
        self.action_server_token = action_server_token
        self.target = target
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

    def _request_via_proxy(
        self,
        user_sub: str,
        path: str,
        query: Dict[str, Any],
        request_name: str,
        raise_on_error: bool = False,
    ) -> Optional[Dict[str, Any]]:
        headers = {
            "Content-Type": "application/json",
            "x-action-server-token": self.action_server_token,
        }

        request_payload: ProxyRequestPayload = {
            "userSub": user_sub,
            "target": self.target,
            "request": {
                "path": path,
                "method": "GET",
                "query": query,
            },
        }

        attempts_total = self.retry_attempts + 1
        last_error: Optional[AnalyticsCenterError] = None

        for attempt in range(attempts_total):
            try:
                response = requests.post(
                    self.proxy_url,
                    headers=headers,
                    json=request_payload,
                    timeout=self.timeout_seconds,
                )
            except requests.Timeout as exc:
                last_error = AnalyticsCenterError(
                    kind="timeout",
                    message=f"{request_name} request timed out",
                    transient=True,
                )
                logger.warning(
                    "[AnalyticsCenterClient] Timeout during %s (attempt=%s/%s): %s",
                    request_name,
                    attempt + 1,
                    attempts_total,
                    exc,
                )
                if attempt < self.retry_attempts:
                    self._sleep_before_retry(attempt)
                    continue
                break
            except requests.RequestException as exc:
                last_error = AnalyticsCenterError(
                    kind="request_error",
                    message=f"{request_name} request failed",
                    transient=True,
                )
                logger.warning(
                    "[AnalyticsCenterClient] Request exception during %s (attempt=%s/%s): %s",
                    request_name,
                    attempt + 1,
                    attempts_total,
                    exc,
                )
                if attempt < self.retry_attempts:
                    self._sleep_before_retry(attempt)
                    continue
                break

            if response.status_code != 200:
                logger.error(
                    "[AnalyticsCenterClient] Proxy error %s during %s (attempt=%s/%s, target=%s, body_len=%s)",
                    response.status_code,
                    request_name,
                    attempt + 1,
                    attempts_total,
                    self.target,
                    len(response.text or ""),
                )
                transient = self._is_transient_status(response.status_code)
                last_error = AnalyticsCenterError(
                    kind="http_error",
                    message=f"Proxy returned HTTP {response.status_code} during {request_name}",
                    status_code=response.status_code,
                    transient=transient,
                )
                if transient and attempt < self.retry_attempts:
                    self._sleep_before_retry(attempt)
                    continue
                break

            response_payload_any: Any = response.json()
            if isinstance(response_payload_any, dict):
                return cast(Dict[str, Any], response_payload_any)

            last_error = AnalyticsCenterError(
                kind="invalid_response",
                message=f"Unexpected response format during {request_name}",
                transient=False,
            )
            logger.error("[AnalyticsCenterClient] Unexpected response shape during %s", request_name)
            break

        if raise_on_error and last_error is not None:
            raise last_error
        return None

    def list_providers(
        self,
        user_sub: str,
        limit: int = 50,
        offset: int = 0,
        country_code: Optional[str] = None,
        sort: Optional[str] = None,
        user: Optional[str] = None,
        group: Optional[int] = None,
        raise_on_error: bool = False,
    ) -> Optional[ProviderCollectionResult]:
        query: Dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }
        if isinstance(country_code, str) and country_code.strip():
            query["country-code"] = country_code.strip()
        if isinstance(sort, str) and sort.strip():
            query["sort"] = sort.strip()
        if isinstance(user, str) and user.strip():
            query["user"] = user.strip()
        if isinstance(group, int) and group > 0:
            query["group"] = group

        payload_dict = self._request_via_proxy(
            user_sub=user_sub,
            path="/api/rest/analytics-center/providers",
            query=query,
            request_name="list_providers",
            raise_on_error=raise_on_error,
        )
        if payload_dict is None:
            return None

        results_any = payload_dict.get("results")
        if isinstance(results_any, list):
            results_list = cast(List[Any], results_any)
            provider_results: List[Dict[str, Any]] = [cast(Dict[str, Any], r) for r in results_list if isinstance(r, dict)]

            pagination_any = payload_dict.get("pagination")
            count = len(provider_results)
            out_limit = limit
            out_offset = offset
            if isinstance(pagination_any, dict):
                pagination_dict = cast(Dict[str, Any], pagination_any)
                c_any = pagination_dict.get("count")
                l_any = pagination_dict.get("limit")
                o_any = pagination_dict.get("offset")
                if isinstance(c_any, int) and c_any >= 0:
                    count = c_any
                if isinstance(l_any, int) and l_any >= 0:
                    out_limit = l_any
                if isinstance(o_any, int) and o_any >= 0:
                    out_offset = o_any

            return {
                "results": provider_results,
                "count": count,
                "limit": out_limit,
                "offset": out_offset,
            }

        logger.error("[AnalyticsCenterClient] Unexpected response format from providers list")
        if raise_on_error:
            raise AnalyticsCenterError(
                kind="invalid_response",
                message="Unexpected providers response format",
                transient=False,
            )
            return None

    def list_countries(
        self,
        user_sub: str,
        limit: int = 300,
        offset: int = 0,
        code: Optional[str] = None,
        raise_on_error: bool = False,
    ) -> Optional[CountryCollectionResult]:
        query: Dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }
        if isinstance(code, str) and code.strip():
            query["code"] = code.strip().upper()

        payload_dict = self._request_via_proxy(
            user_sub=user_sub,
            path="/api/rest/analytics-center/countries",
            query=query,
            request_name="list_countries",
            raise_on_error=raise_on_error,
        )
        if payload_dict is None:
            return None

        results_any = payload_dict.get("results")
        if isinstance(results_any, list):
            results_list = cast(List[Any], results_any)
            country_results: List[Dict[str, Any]] = [cast(Dict[str, Any], r) for r in results_list if isinstance(r, dict)]
            return {"results": country_results}

        logger.error("[AnalyticsCenterClient] Unexpected response format from countries list")
        if raise_on_error:
            raise AnalyticsCenterError(
                kind="invalid_response",
                message="Unexpected countries response format",
                transient=False,
            )
            return None

    def resolve_country_code(self, user_sub: str, country_input: str, raise_on_error: bool = False) -> Optional[str]:
        raw = (country_input or "").strip()
        if not raw:
            return None
        if len(raw) == 2 and raw.isalpha():
            return raw.upper()

        aliases: Dict[str, str] = {
            "spain": "ES",
            "espana": "ES",
            "españa": "ES",
            "mexico": "MX",
            "méxico": "MX",
            "czech republic": "CZ",
            "czechia": "CZ",
            "united kingdom": "GB",
            "uk": "GB",
            "great britain": "GB",
            "united states": "US",
            "usa": "US",
            "u.s.a": "US",
        }
        normalized = raw.lower()
        if normalized in aliases:
            return aliases[normalized]

        countries_page = self.list_countries(user_sub=user_sub, limit=300, offset=0, raise_on_error=raise_on_error)
        if not countries_page:
            return None

        for country in countries_page["results"]:
            code_any = country.get("countryCode")
            name_any = country.get("name")
            if not isinstance(code_any, str) or not code_any.strip():
                continue
            code = code_any.strip().upper()
            if normalized == code.lower():
                return code
            if isinstance(name_any, str) and name_any.strip() and normalized == name_any.strip().lower():
                return code

        return None


def get_analytics_center_client() -> AnalyticsCenterClient:
    proxy_url, action_server_token = env_util.require_all_env("RASA_PROXY_URL", "ACTION_SERVER_TOKEN")
    target = env_util.require_any_env("RASA_PROXY_ANALYTICS_TARGET")
    target_val = target if isinstance(target, str) and target.strip() else "analytics"
    return AnalyticsCenterClient(
        proxy_url=proxy_url,
        action_server_token=action_server_token,
        target=target_val,
    )
