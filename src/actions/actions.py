import json
import logging
from typing import Any, Dict, List, Optional, Text, cast

from rasa_sdk import Action, Tracker  # type: ignore
from rasa_sdk import types as rasa_types  # type: ignore
from rasa_sdk.executor import CollectingDispatcher  # type: ignore

from src.actions.long_action.long_action import LongAction
from src.actions.long_action.long_action_context import LongActionContext
from src.domain.langchain import schema as lang_schema
from src.executors import plan_executor
from src.executors.langchain import pipeline as lang_pipeline
from src.executors.simple_planner import HeuristicVisualizationPlanner
from src.shared import ssot_loader

logger = logging.getLogger(__name__)

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
        try:
            user_message = ctx.text
            session_token = ctx.sender_id
            logger.info("Processing visualization request: '%s'", user_message)

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
                session_token=session_token,
                max_concurrency=4,
                progress_cb=progress,
            )

            ctx.say(json_message=json.loads(visualization.model_dump_json()))
        except Exception as e:
            error_msg = f"Error generating visualization: {str(e)}"
            logger.error(error_msg)
            ctx.say(text="❌ Error generating visualization.")
            ctx.say(text=error_msg)
        finally:
            ctx.say(text="✅ Visualization generation complete.")
            ctx.done()
        return None


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
