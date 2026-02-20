import logging
from typing import Any, Dict, List, Optional, cast

import requests

from src.util import env as env_util

logger = logging.getLogger(__name__)


class AnalyticsCenterClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def list_providers(
        self,
        session_token: str,
        limit: int = 50,
        offset: int = 0,
    ) -> Optional[List[Dict[str, Any]]]:
        url = f"{self.base_url}/providers"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {session_token}",
        }
        params = {
            "limit": limit,
            "offset": offset,
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            if response.status_code != 200:
                logger.error(
                    "[AnalyticsCenterClient] Error %s fetching providers (body_len=%s)",
                    response.status_code,
                    len(response.text or ""),
                )
                return None
            payload = response.json()
            if isinstance(payload, dict):
                payload_dict = cast(Dict[str, Any], payload)
            else:
                payload_dict = {}

            results_any = payload_dict.get("results")
            if isinstance(results_any, list):
                results_list = cast(List[Any], results_any)
                return [r for r in results_list if isinstance(r, dict)]
            logger.error("[AnalyticsCenterClient] Unexpected response format from providers list")
            return None
        except Exception as exc:
            logger.error("[AnalyticsCenterClient] Request failed: %s", exc)
            return None


def get_analytics_center_client() -> AnalyticsCenterClient:
    base_url = env_util.require_all_env("ANALYTICS_CENTER_BASE_URL")
    return AnalyticsCenterClient(base_url)
