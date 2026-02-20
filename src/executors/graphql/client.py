import hashlib
import logging
from typing import Any, Dict, Optional, TypedDict

import requests

import src.domain.graphql.response as gqlr
from src.util import env as env_util


class GraphQLPayload(TypedDict):
    query: str
    variables: Dict[str, Any]


class ProxyRequestPayload(TypedDict):
    operation: str
    target: str
    url: str
    payload: GraphQLPayload


logger = logging.getLogger(__name__)
# Privacy/safety defaults: avoid logging raw GraphQL payloads.
_LOG_GRAPHQL_QUERY = env_util.env_flag("GRAPHQL_LOG_QUERY", default=False)
_LOG_GRAPHQL_BODY = env_util.env_flag("GRAPHQL_LOG_BODY", default=False)


class GraphQLProxyClient:
    def __init__(self, proxy_url: str, graphql_url: str):
        self.proxy_url = proxy_url
        self.graphql_url = graphql_url

    def query(self, query_str: str, session_token: str, variables: Optional[Dict[str, Any]] = None) -> gqlr.MetricsQueryResponse | None:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {session_token}",
        }

        proxy_payload: ProxyRequestPayload = {
            "operation": "query",
            "target": "graphql",
            "url": self.graphql_url,
            "payload": {"query": query_str, "variables": variables or {}},
        }

        try:
            response = requests.post(self.proxy_url, headers=headers, json=proxy_payload)

            q_hash = hashlib.sha256(query_str.encode("utf-8")).hexdigest()[:12]

            if response.status_code == 200:
                try:
                    return gqlr.MetricsQueryResponse.model_validate(response.json())
                except Exception as e:
                    if _LOG_GRAPHQL_BODY:
                        logger.error("[GraphQLProxyClient] Validation error (hash=%s): %s. Raw: %s", q_hash, e, response.text)
                    else:
                        logger.error(
                            "[GraphQLProxyClient] Validation error (hash=%s): %s (body_len=%s)",
                            q_hash,
                            e,
                            len(response.text or ""),
                        )
                    return None
            else:
                content_type = response.headers.get("Content-Type", "")
                if _LOG_GRAPHQL_BODY or _LOG_GRAPHQL_QUERY:
                    body_preview = response.text[:1000] if _LOG_GRAPHQL_BODY else "(body logging disabled)"
                    query_preview = query_str[:300] if _LOG_GRAPHQL_QUERY else "(query logging disabled)"
                    logger.error(
                        "[GraphQLProxyClient] Error %s (Content-Type=%s, hash=%s). Body preview: %s. Query preview: %s",
                        response.status_code,
                        content_type,
                        q_hash,
                        body_preview,
                        query_preview,
                    )
                else:
                    logger.error(
                        "[GraphQLProxyClient] Error %s (Content-Type=%s, hash=%s, body_len=%s, query_len=%s)",
                        response.status_code,
                        content_type,
                        q_hash,
                        len(response.text or ""),
                        len(query_str),
                    )
                return None

        except Exception as e:
            logger.error("[GraphQLProxyClient] Request failed: %s", e)
            return None
