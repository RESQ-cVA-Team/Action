import json
import logging
from typing import Any, Dict, List, Optional, Text, cast

from rasa_sdk import Action, Tracker  # type: ignore
from rasa_sdk import types as rasa_types  # type: ignore
from rasa_sdk.executor import CollectingDispatcher  # type: ignore

from src.actions.error_messages import friendly_hospital_error, friendly_metric_error, friendly_visualization_error
from src.actions.long_action.long_action import LongAction
from src.actions.long_action.long_action_context import LongActionContext
from src.domain.langchain import schema as lang_schema
from src.executors import plan_executor
from src.executors.analytics_center.client import get_analytics_center_client
from src.executors.langchain import pipeline as lang_pipeline
from src.executors.simple_planner import HeuristicVisualizationPlanner
from src.shared import ssot_loader
from src.util import env as env_util

logger = logging.getLogger(__name__)


# Privacy/safety defaults:
# - Do not log full user messages by default.
# - Do not echo internal exception messages to end users by default.
_LOG_USER_TEXT = env_util.env_flag("ACTIONS_LOG_USER_TEXT", default=False)
_ECHO_INTERNAL_ERRORS = env_util.env_flag("ACTIONS_ECHO_INTERNAL_ERRORS", default=False)

try:
    ssot_loader.validate_metric_metadata_complete(logger)
except Exception as _e:
    logger.debug("SSOT validation skipped due to: %s", _e)


# Warm up metric text lookup cache at import time so MetricType.yml is loaded
# once at startup.
_METRIC_TEXT_LOOKUP = ssot_loader.get_metric_text_lookup()


class ActionGenerateVisualization(LongAction):
    """Long action that uses the planner chain to generate visualizations and statistics.

    In callback mode this streams messages via the long-action callback URL;
    otherwise it behaves like a normal synchronous action and uses the
    dispatcher directly.
    """

    def name(self) -> str:
        return "action_generate_visualization"

    async def work(self, ctx: LongActionContext) -> Any:
        completed_successfully = False
        try:
            user_message = ctx.text
            user_sub = ctx.sender_id
            if _LOG_USER_TEXT:
                logger.info("Processing visualization request: '%s'", user_message)
            else:
                logger.info(
                    "Processing visualization request (text_len=%s)",
                    len(user_message or ""),
                )

            latest_meta = ctx.metadata

            latest_any = ctx.tracker_snapshot.get("latest_message")
            entities_list: List[Dict[str, Any]] = []
            if isinstance(latest_any, dict):
                latest_msg = cast(Dict[str, Any], latest_any)
                ents_any = latest_msg.get("entities", [])
                if isinstance(ents_any, list):
                    ents_list = cast(List[Any], ents_any)
                    for e_any in ents_list:
                        if isinstance(e_any, dict):
                            entities_list.append(cast(Dict[str, Any], e_any))
            extracted_entities: Dict[str, Any] = {ent["entity"]: ent["value"] for ent in entities_list if isinstance(ent.get("entity"), str) and "value" in ent}

            override_language: Any = None
            try:
                lang_meta = latest_meta.get("language")
                if isinstance(lang_meta, str) and lang_meta.strip():
                    override_language = lang_meta
                if override_language is None:
                    slot_lang = ctx.slots.get("language")
                    if isinstance(slot_lang, str) and slot_lang.strip():
                        override_language = slot_lang
            except Exception:
                pass
            if isinstance(override_language, str):
                override_language = override_language.split("-")[0].lower() or None

            def progress(msg: str) -> None:
                ctx.say(progress=msg)

            heuristic_plan = HeuristicVisualizationPlanner.try_plan(
                question=user_message,
                entities=extracted_entities,
                language=override_language,
            )

            if heuristic_plan is not None:
                progress("Using simple heuristic plan (no LLM needed)")
                plan_obj: lang_schema.AnalysisPlan = heuristic_plan
            else:
                progress("Calling planner LLM to build a plan")
                plan_obj = lang_pipeline.generate_analysis_plan(
                    question=user_message,
                    entities=extracted_entities,
                    language=override_language,
                    max_retries=2,
                    debug=False,
                    progress_cb=progress,
                )

            visualization = await plan_executor.execute_plan_async(
                plan_obj,
                user_sub=user_sub,
                max_concurrency=4,
                progress_cb=progress,
            )

            ctx.say(json_message=json.loads(visualization.model_dump_json()))
            completed_successfully = True
        except Exception as e:
            logger.exception("Error generating visualization")
            ctx.say(text=f"❌ {friendly_visualization_error(e)}")
            if _ECHO_INTERNAL_ERRORS:
                ctx.say(text=f"Error generating visualization: {str(e)}")
        finally:
            if completed_successfully:
                ctx.say(text="✅ Visualization generation complete.")
            ctx.done()
        return None


class ActionListHospitals(Action):
    """List hospitals/providers available for comparison."""

    def name(self) -> str:
        return "action_list_hospitals"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: rasa_types.DomainDict,
    ) -> List[Dict[Text, Any]]:
        try:
            user_sub = tracker.sender_id
            filters = self._extract_filters(tracker)
            client = get_analytics_center_client()

            raw_country = filters.get("country_code")
            if isinstance(raw_country, str) and raw_country.strip():
                resolved_country = client.resolve_country_code(user_sub=user_sub, country_input=raw_country, raise_on_error=True)
                if resolved_country:
                    filters["country_code"] = resolved_country
                else:
                    dispatcher.utter_message(text=f"I couldn't match country '{raw_country}'. Please try a 2-letter code like ES, MX, DE, or FR.")
                    return []

            provider_page = client.list_providers(
                user_sub=user_sub,
                limit=filters["limit"],
                offset=filters["offset"],
                country_code=filters.get("country_code"),
                sort=filters.get("sort"),
                user=filters.get("user_id"),
                group=filters.get("group_id"),
                raise_on_error=True,
            )
            if not provider_page:
                dispatcher.utter_message(text="I couldn't find any hospitals you can compare against.")
                return []

            providers = provider_page["results"]
            total_count = provider_page["count"]
            offset = provider_page["offset"]
            limit = provider_page["limit"]

            names: List[str] = []
            for provider in providers:
                name = provider.get("nameEnglish") or provider.get("nameNative") or provider.get("shortName")
                if isinstance(name, str) and name.strip():
                    names.append(name.strip())

            name_filter = filters.get("name_contains")
            if isinstance(name_filter, str) and name_filter.strip():
                needle = name_filter.strip().lower()
                names = [n for n in names if needle in n.lower()]

            if not names:
                no_match_msg = "I couldn't find any hospitals matching your filters."
                if isinstance(name_filter, str) and name_filter.strip():
                    no_match_msg += " Try a shorter or less specific name."
                dispatcher.utter_message(text=no_match_msg)
                return []

            preview = ", ".join(names[:10])
            more_count = max(len(names) - 10, 0)
            criteria: List[str] = []
            country_code = filters.get("country_code")
            sort = filters.get("sort")
            if isinstance(country_code, str) and country_code.strip():
                criteria.append(f"country={country_code.strip().upper()}")
            if isinstance(sort, str) and sort.strip():
                criteria.append(f"sort={sort.strip()}")

            criteria_text = f" ({', '.join(criteria)})" if criteria else ""

            if isinstance(name_filter, str) and name_filter.strip():
                prefix = f"I found {len(names)} matching hospitals on this page (search='{name_filter.strip()}', offset={offset}, limit={limit}); total providers before name filter: {total_count}."
            else:
                shown_start = offset + 1 if names else 0
                shown_end = offset + len(names)
                prefix = f"I found {total_count} hospitals{criteria_text}; showing {shown_start}-{shown_end}."

            text_message = prefix + f" {preview}." + (f" (+{more_count} more in this page)" if more_count else "")
            dispatcher.utter_message(text=text_message)
            return []
        except Exception as exc:
            logger.exception("Error listing hospitals")
            dispatcher.utter_message(text=f"❌ {friendly_hospital_error(exc)}")
            if _ECHO_INTERNAL_ERRORS:
                dispatcher.utter_message(text=f"Error listing hospitals: {str(exc)}")
            return []

    @staticmethod
    def _extract_filters(tracker: Tracker) -> Dict[str, Any]:
        tracker_any: Any = tracker
        latest_any: Any = tracker_any.latest_message or {}
        latest = cast(Dict[str, Any], latest_any) if isinstance(latest_any, dict) else {}

        metadata_any: Any = latest.get("metadata")
        metadata = cast(Dict[str, Any], metadata_any) if isinstance(metadata_any, dict) else {}

        entities_by_name: Dict[str, Any] = {}
        entities_any: Any = latest.get("entities")
        if isinstance(entities_any, list):
            entities_list = cast(List[Any], entities_any)
            for ent_any in entities_list:
                if not isinstance(ent_any, dict):
                    continue
                ent = cast(Dict[str, Any], ent_any)
                key = ent.get("entity")
                value = ent.get("value")
                if isinstance(key, str) and key.strip() and value is not None and key not in entities_by_name:
                    entities_by_name[key] = value

        def first_str(*candidates: Any) -> Optional[str]:
            for candidate in candidates:
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
            return None

        def first_int(*candidates: Any) -> Optional[int]:
            for candidate in candidates:
                if isinstance(candidate, int):
                    return candidate
                if isinstance(candidate, str):
                    txt = candidate.strip()
                    if txt.isdigit():
                        return int(txt)
            return None

        country_code = first_str(
            metadata.get("countryCode"),
            metadata.get("country_code"),
            entities_by_name.get("countryCode"),
            entities_by_name.get("country_code"),
            entities_by_name.get("country"),
            tracker_any.get_slot("countryCode"),
            tracker_any.get_slot("country_code"),
            tracker_any.get_slot("country"),
        )

        name_contains = first_str(
            metadata.get("hospitalName"),
            metadata.get("hospital_name"),
            metadata.get("name"),
            entities_by_name.get("hospital_name"),
            entities_by_name.get("hospital"),
            entities_by_name.get("provider"),
            entities_by_name.get("name"),
            tracker_any.get_slot("hospital_name"),
            tracker_any.get_slot("hospital"),
            tracker_any.get_slot("provider"),
            tracker_any.get_slot("name"),
        )

        sort = first_str(
            metadata.get("sort"),
            tracker_any.get_slot("sort"),
        )

        user_id = first_str(
            metadata.get("user"),
            metadata.get("userId"),
            tracker_any.get_slot("user"),
            tracker_any.get_slot("user_id"),
        )

        group_id = first_int(
            metadata.get("group"),
            metadata.get("groupId"),
            tracker_any.get_slot("group"),
            tracker_any.get_slot("group_id"),
        )

        limit_val = first_int(
            metadata.get("limit"),
            tracker_any.get_slot("limit"),
        )
        offset_val = first_int(
            metadata.get("offset"),
            tracker_any.get_slot("offset"),
        )

        limit = 50 if limit_val is None else max(1, min(limit_val, 200))
        offset = 0 if offset_val is None else max(0, offset_val)

        return {
            "country_code": country_code.upper() if isinstance(country_code, str) and len(country_code) == 2 else country_code,
            "name_contains": name_contains,
            "sort": sort,
            "user_id": user_id,
            "group_id": group_id,
            "limit": limit,
            "offset": offset,
        }


class ActionExplainMetric(Action):
    """Explain a metric/KPI based on MetricType.yml and language.

    The metric is resolved from the latest user message `kpi` entity (or the
    `kpi` slot as a fallback), normalized using the same rules as the SSOT
    lookup. The response uses the localized description from the SSOT
    `descriptions` block, falling back to English or the first available
    language.
    """

    def name(self) -> str:
        return "action_explain_metric"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: rasa_types.DomainDict,
    ) -> List[Dict[Text, Any]]:
        try:
            language = self._resolve_language(tracker)

            raw_kpi = self._extract_kpi(tracker)
            if not raw_kpi:
                dispatcher.utter_message(text="I couldn't find any metric in your request.")
                return []

            norm_key = ssot_loader.normalize_metric_text_key(raw_kpi)
            if not norm_key:
                dispatcher.utter_message(text="I couldn't understand the metric you asked about.")
                return []

            record = _METRIC_TEXT_LOOKUP.get(norm_key)
            if not record:
                suggestions = self._suggest_metrics(language, max_items=5)
                if suggestions:
                    dispatcher.utter_message(text=("I don't recognise that metric. " + "Here are a few metrics I can describe: " + ", ".join(suggestions) + "."))
                else:
                    dispatcher.utter_message(text="I don't recognise that metric.")
                return []

            canonical = cast(str, record.get("canonical") or "")
            descriptions = cast(Dict[str, str], record.get("descriptions") or {})
            data_type = cast(Optional[str], record.get("data_type"))
            unit = cast(Optional[str], record.get("unit"))

            description_text = self._pick_description(descriptions, language)
            if not description_text:
                # Fallback: use any available description if language-specific is missing.
                for txt in descriptions.values():
                    txt_val = txt.strip()
                    if txt_val:
                        description_text = txt_val
                        break

            if not canonical or not description_text:
                dispatcher.utter_message(text="This is a known metric, but its description is not configured yet.")
                return []

            # Human-friendly display name from SSOT synonyms.
            display_name = ssot_loader.get_metric_display_name(canonical)

            header: str
            if display_name and display_name != canonical:
                header = f"{canonical} – {display_name}."
            else:
                header = f"{canonical}."

            parts: List[str] = [header, description_text]
            if data_type:
                if unit:
                    parts.append(f"Data type: {data_type} ({unit}).")
                else:
                    parts.append(f"Data type: {data_type}.")

            dispatcher.utter_message(text=" ".join(parts))
            return []
        except Exception as e:
            logger.exception("Error explaining metric")
            dispatcher.utter_message(text=f"❌ {friendly_metric_error(e)}")
            if _ECHO_INTERNAL_ERRORS:
                dispatcher.utter_message(text=f"Error explaining metric: {str(e)}")
            return []

    @staticmethod
    def _resolve_language(tracker: Tracker) -> str:
        """Resolve language code from tracker metadata or slots.

        Prefers tracker.latest_message.metadata.language, then the `language`
        slot. Always returns a short language code like `en` or `es`.
        """

        lang: Optional[str] = None

        tracker_any: Any = tracker

        meta_any = tracker_any.latest_message.get("metadata")
        if isinstance(meta_any, dict):
            meta = cast(Dict[str, Any], meta_any)
            lang_val = meta.get("language")
            if isinstance(lang_val, str) and lang_val.strip():
                lang = lang_val.strip()

        if not lang:
            slot_lang_any: Any = tracker_any.get_slot("language")
            if isinstance(slot_lang_any, str) and slot_lang_any.strip():
                lang = slot_lang_any.strip()

        if not lang:
            return "en"

        # Normalize: take primary language subtag only (e.g., en-US -> en).
        primary = lang.split("-")[0].strip().lower()
        return primary or "en"

    @staticmethod
    def _extract_kpi(tracker: Tracker) -> Optional[str]:
        """Extract the KPI/metric text from the latest user message or slot.

        Primary source is the latest_message `kpi` entity. If not found,
        falls back to the `kpi` slot.
        """

        tracker_any: Any = tracker
        latest: Dict[str, Any] = tracker_any.latest_message or {}
        entities_any: Any = latest.get("entities")
        if isinstance(entities_any, list):
            entities_list: List[Any] = cast(List[Any], entities_any)
            for ent_any in entities_list:
                if not isinstance(ent_any, dict):
                    continue
                ent = cast(Dict[str, Any], ent_any)
                if ent.get("entity") == "kpi":
                    val = ent.get("value")
                    if isinstance(val, str) and val.strip():
                        return val.strip()

        slot_val_any: Any = tracker_any.get_slot("kpi")
        if isinstance(slot_val_any, str) and slot_val_any.strip():
            return slot_val_any.strip()

        return None

    @staticmethod
    def _pick_description(descriptions: Dict[str, str], language: str) -> str:
        """Pick the best description for the requested language.

        - Try exact language key (e.g. `en`, `es`).
        - Fallback to English (`en`) if available.
        - Otherwise return empty string and let caller decide.
        """

        if not descriptions:
            return ""

        # Exact match
        txt = descriptions.get(language)
        if isinstance(txt, str) and txt.strip():
            return txt.strip()

        # Fallback to English
        txt = descriptions.get("en")
        if isinstance(txt, str) and txt.strip():
            return txt.strip()

        return ""

    @staticmethod
    def _suggest_metrics(language: str, max_items: int = 5) -> List[str]:
        """Return a small list of known metrics for graceful fallbacks."""

        seen: Dict[str, str] = {}
        for record in _METRIC_TEXT_LOOKUP.values():
            canonical = record.get("canonical")
            if not isinstance(canonical, str) or not canonical:
                continue
            code = canonical.upper()
            if code in seen:
                continue
            label = ssot_loader.get_metric_display_name(code)
            seen[code] = label
            if len(seen) >= max_items:
                break

        return list(seen.values())
