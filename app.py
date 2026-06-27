# -*- coding: utf-8 -*-
"""Streamlit swarm prototype for Israeli vehicle-model mapping with Kimi K2.6."""

from __future__ import annotations

import json
import os
import re
import time
from threading import BoundedSemaphore
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from openai import OpenAI

MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"
KIMI_MODEL = "kimi-k2.6"
SEARCH_TEMPERATURE = 0.6
CONSOLIDATION_TEMPERATURE = 0.6
MAX_TOOL_ROUNDS = 15
MAX_PARALLEL_KIMI_CALLS = 2
KIMI_CONCURRENCY_SEMAPHORE = BoundedSemaphore(MAX_PARALLEL_KIMI_CALLS)
API_CONCURRENCY_RETRY_DELAY_SECONDS = 2
API_CONCURRENCY_MAX_RETRIES = 2
MAX_DISCOVERY_TOKENS = 1800
MAX_DISCOVERY_FALLBACK_TOKENS = 1200
MAX_TECHNICAL_AGENT_TOKENS = 2500
TECHNICAL_MODEL_CHUNK_SIZE = 4
MAX_VERIFIER_TOKENS = 3500
VERIFIER_MODEL_CHUNK_SIZE = 12
MAX_FINAL_BUILDER_TOKENS = 4500
MAX_SUMMARY_TOKENS = 1200
RAW_DEBUG_PREVIEW_CHARS = 2000
MAX_REASONABLE_OUTPUT_CHARS = 120_000
INPUT_COST_PER_1M = 0.95
OUTPUT_COST_PER_1M = 4.00
DEFAULT_MANUFACTURER = "Hyundai"
DEFAULT_MARKET = "Israel"
DEFAULT_PERIOD = "2010 to June 2026"

WEB_SEARCH_TOOL = [{"type": "builtin_function", "function": {"name": "$web_search"}}]
MANDATORY_WEB_SEARCH_INSTRUCTION = (
    "IMPORTANT: You MUST call the $web_search tool to find information. "
    "Do NOT answer from memory. Do NOT describe what you plan to search. "
    "Execute the search immediately.\n\n"
)

ARCHITECTURE_ASCII = """
Phase 1: Discovery (3 focused web agents)
          |
          v
Phase 2: Python merge + normalizer
          |
          v
Phase 3: Technical enrichment (4 focused web agents)
          |
          v
Phase 4: Verifier -> Final builder -> Hebrew summary
"""

ISRAEL_DISCOVERY_CONTEXT = """
Israeli market context:
- The Israeli market has unique model names — some models are sold under different names than global.
- Search Israeli automotive sources, including official local importer/manufacturer sources, Israeli car portals, and Israeli used-car marketplaces.
- Some models sold in Israel were never sold in the US/Europe and vice versa.
- Israeli model years sometimes lag global launch by 1-2 years.
"""

ISRAEL_ENRICHMENT_CONTEXT = """
Israeli market context:
- Specs must reflect ISRAELI-spec vehicles, not global/US/EU specs.
- Trim level names in Israel are frequently different from global naming.
- Prices should be in ILS (Israeli New Shekel) if found.
- Israeli vehicles are often imported by a single authorized importer — their website is a primary source.
- Fuel consumption figures should follow Israeli/European standards (l/100km), not US MPG.
- Search queries should include Hebrew terms alongside English to find local sources.
- Safety equipment may differ from European spec due to local regulations.
"""

CONSOLIDATION_CONTEXT = """
Israeli market consolidation rules:
- model_name_he field is mandatory for every model — if discovery did not find it, mark as null, never transliterate.
- Prices in ILS only, not converted from other currencies.
- If a model has different specs for Israeli market vs global, the Israeli spec wins.
- Use null for missing data. Never invent facts.
"""


@dataclass(frozen=True)
class AgentConfig:
    key: str
    name: str
    description: str
    responsibility: str


DISCOVERY_AGENTS: List[AgentConfig] = [
    AgentConfig("current_official_lineup_agent", "Current official lineup", "Current official/importer models", "Return only currently listed Hyundai model names from official Israel/importer sources."),
    AgentConfig("historical_used_market_agent", "Historical used market", "Historical used-car/model-list models", "Return only historical Hyundai model names that appear in Israeli used-market/model-listing sources."),
    AgentConfig("ev_hybrid_edge_cases_agent", "EV/hybrid edge cases", "EV/hybrid model names", "Return only EV/hybrid Hyundai model names sold in Israel."),
]

DISCOVERY_MAX_MODELS: Dict[str, int] = {
    "current_official_lineup_agent": 25,
    "historical_used_market_agent": 40,
    "ev_hybrid_edge_cases_agent": 25,
}

TECHNICAL_AGENTS: List[AgentConfig] = [
    AgentConfig("trims_years_agent", "Trims & years", "Israeli trims and years", "Collect Israeli trims / versions, approximate years sold, and generation labels when known."),
    AgentConfig("engines_fuel_power_agent", "Engines, fuel & power", "Powertrain facts", "Collect engine displacement, fuel type, hybrid/EV details, power hp, and torque when available."),
    AgentConfig("transmission_drivetrain_performance_agent", "Transmission & performance", "Transmission/drivetrain/performance", "Collect transmission type, drivetrain, 0-100 when available, and notable gearbox notes."),
    AgentConfig("dimensions_safety_equipment_agent", "Dimensions, safety & equipment", "Dimensions/safety/equipment", "Collect body type, seats, trunk volume, dimensions, safety rating/systems, and key common equipment."),
]


PLANNING_LOOP_LIMITS: Tuple[Tuple[str, int], ...] = (
    ("I'll search", 3),
    ("I will search", 3),
    ("Let me search", 3),
    ("Let me search again", 2),
    ("I need to search", 3),
    ("I need to search for more specific information", 2),
    ("I need to find", 5),
    ("I should also search", 5),
    ("I'll also search", 5),
    ("search again with different queries", 2),
    ("more targeted queries", 3),
)


def detect_planning_or_repetition_loop(text: str) -> tuple[bool, str]:
    """Detect repeated planning/search narration that indicates a model loop."""
    lowered = (text or "").lower()
    tripped = []
    for phrase, limit in PLANNING_LOOP_LIMITS:
        count = lowered.count(phrase.lower())
        if count > limit:
            tripped.append((phrase, count))
    if not tripped:
        return False, ""
    repeated_counts = [count for _, count in tripped]
    reason = "MODEL_REPETITION_LOOP" if max(repeated_counts) > 5 or len(tripped) > 1 else "MODEL_PLANNING_LOOP"
    return True, reason


def _parse_json_strict(content: str) -> Tuple[Optional[Any], Optional[str]]:
    """Parse model output as JSON. Invalid JSON is a hard failure."""
    text = (content or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text), None
    except Exception:
        return None, "INVALID_JSON"


def _looks_like_partial_json(content: str) -> bool:
    """Return True when truncated output appears to be an unfinished JSON value."""
    text = (content or "").lstrip()
    if not text.startswith(("{", "[")):
        return False
    parsed, parse_error = _parse_json_strict(text)
    return parse_error is not None and parsed is None


def _error_payload(error: str, finish_reason: Any, input_tokens: int, output_tokens: int, content: str, *, agent: str = "", phase: str = "") -> Dict[str, Any]:
    payload = {
        "_error": error,
        "agent": agent,
        "phase": phase,
        "finish_reason": finish_reason,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "raw_preview": (content or "")[:RAW_DEBUG_PREVIEW_CHARS],
    }
    if error == "MODEL_JSON_TRUNCATED":
        if (phase or "").startswith("technical"):
            payload["message"] = (
                "Technical agent produced valid-looking JSON but exceeded token budget. "
                "Reduce chunk size or compact the technical schema."
            )
        elif (phase or "").startswith("verification"):
            payload["message"] = (
                "Verifier produced valid-looking JSON but exceeded token budget. "
                "Reduce verifier chunk size or compact verifier schema."
            )
        else:
            payload["message"] = (
                "Discovery produced valid-looking JSON but exceeded token budget. "
                "Reduce schema or increase MAX_DISCOVERY_TOKENS."
            )
    return payload


def validate_model_response(
    result: Dict[str, Any],
    *,
    require_json: bool,
    validator: Optional[Any] = None,
    required_keys: Optional[List[str]] = None,
    non_empty_lists: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Apply common hard safety checks to a Kimi/Moonshot response."""
    content = result.get("content", "") or ""
    agent = result.get("agent", "")
    phase = result.get("phase", "")
    if result.get("finish_reason") == "length":
        error = "MODEL_JSON_TRUNCATED" if require_json and _looks_like_partial_json(content) else "MODEL_OUTPUT_TRUNCATED"
        return _error_payload(error, "length", result.get("input_tokens", 0), result.get("output_tokens", 0), content, agent=agent, phase=phase)
    looped, reason = detect_planning_or_repetition_loop(content)
    if looped:
        return _error_payload(reason, result.get("finish_reason"), result.get("input_tokens", 0), result.get("output_tokens", 0), content, agent=agent, phase=phase)
    if len(content) > MAX_REASONABLE_OUTPUT_CHARS:
        return _error_payload("MODEL_OUTPUT_TOO_LARGE", result.get("finish_reason"), result.get("input_tokens", 0), result.get("output_tokens", 0), content, agent=agent, phase=phase)
    if require_json:
        parsed, parse_error = _parse_json_strict(content)
        if parse_error or (isinstance(parsed, dict) and "_raw_text" in parsed):
            return _error_payload("INVALID_JSON", result.get("finish_reason"), result.get("input_tokens", 0), result.get("output_tokens", 0), content, agent=agent, phase=phase)
        if phase == "technical" and isinstance(parsed, dict):
            if GENERIC_AUTOMOTIVE_TOP_LEVEL_KEYS.intersection(parsed.keys()):
                return _error_payload("GENERIC_AUTOMOTIVE_OUTPUT", result.get("finish_reason"), result.get("input_tokens", 0), result.get("output_tokens", 0), content, agent=agent, phase=phase)
            combined_text = json.dumps(parsed, ensure_ascii=False).lower()
            if any(marker in combined_text for marker in MISSING_MAKE_MODEL_MARKERS):
                return _error_payload("AGENT_DID_NOT_RECEIVE_MODEL_CHUNK", result.get("finish_reason"), result.get("input_tokens", 0), result.get("output_tokens", 0), content, agent=agent, phase=phase)
        repaired_fields: List[str] = []
        if phase == "technical" and isinstance(parsed, dict):
            if "agent" not in parsed:
                parsed["agent"] = agent
                repaired_fields.append("agent")
            if "missing_data" not in parsed:
                parsed["missing_data"] = []
                repaired_fields.append("missing_data")
            if "extra_candidate_models" not in parsed:
                parsed["extra_candidate_models"] = []
                repaired_fields.append("extra_candidate_models")
        if required_keys and isinstance(parsed, dict):
            for key in required_keys:
                if key not in parsed:
                    return _error_payload(f"MISSING_REQUIRED_KEY:{key}", result.get("finish_reason"), result.get("input_tokens", 0), result.get("output_tokens", 0), content, agent=agent, phase=phase)
        if non_empty_lists and isinstance(parsed, dict):
            for key in non_empty_lists:
                if not isinstance(parsed.get(key), list) or not parsed.get(key):
                    return _error_payload(f"EMPTY_REQUIRED_LIST:{key}", result.get("finish_reason"), result.get("input_tokens", 0), result.get("output_tokens", 0), content, agent=agent, phase=phase)
        if validator:
            validation_error = validator(parsed)
            if validation_error:
                payload = _error_payload(validation_error, result.get("finish_reason"), result.get("input_tokens", 0), result.get("output_tokens", 0), content, agent=agent, phase=phase)
                payload["parsed_preview"] = parsed if isinstance(parsed, (dict, list)) else None
                return payload
        ok = dict(result)
        ok["parsed"] = parsed
        if repaired_fields:
            ok["repaired_fields"] = repaired_fields
        return ok
    return dict(result)


def phase_result(
    *,
    status: str,
    agent: str,
    parsed: Optional[Any] = None,
    error: Optional[str] = None,
    raw_preview: str = "",
    finish_reason: Any = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    used_fallback: bool = False,
    api_retry_count: int = 0,
    message: str = "",
) -> Dict[str, Any]:
    return {
        "status": status,
        "agent": agent,
        "parsed": parsed,
        "error": error,
        "raw_preview": (raw_preview or "")[:RAW_DEBUG_PREVIEW_CHARS],
        "finish_reason": finish_reason,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "used_fallback": used_fallback,
        "api_retry_count": api_retry_count,
        "message": message,
    }


TRIM_OR_PACKAGE_NAMES = {"n line", "premium", "prestige", "executive", "limited", "luxury", "comfort", "style", "ultimate", "gl", "gls", "lx", "ex"}


def normalize_model_name(name: str) -> str:
    cleaned = re.sub(r"\b(hyundai|kia|toyota|mazda|ford|nissan|mitsubishi|suzuki)\b", "", name or "", flags=re.IGNORECASE)
    cleaned = re.sub(r"[\u200e\u200f\"'`]", "", cleaned)
    cleaned = re.sub(r"[^0-9A-Za-zא-ת]+", " ", cleaned).strip().lower()
    return re.sub(r"\s+", " ", cleaned)


def _confidence_rank(value: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}.get((value or "").lower(), 0)


def merge_discovery_candidates(discovery_results: list[dict]) -> dict:
    merged: Dict[str, Dict[str, Any]] = {}
    rejected: List[Dict[str, str]] = []
    failed_agents: List[Dict[str, Any]] = []
    manufacturer = market = period = ""

    for result in discovery_results:
        if result.get("status") != "success":
            failed_agents.append({"agent": result.get("agent"), "error": result.get("error")})
            continue
        parsed = result.get("parsed") or {}
        manufacturer = manufacturer or parsed.get("manufacturer", "")
        market = market or parsed.get("market", "")
        period = period or parsed.get("period", "")
        agent = parsed.get("agent") or result.get("agent")
        for item in parsed.get("models", []):
            name = item.get("model_name_en") or item.get("model") or item.get("name") or ""
            norm = normalize_model_name(name)
            if not norm:
                continue
            if norm in TRIM_OR_PACKAGE_NAMES:
                rejected.append({"name": name, "reason": "trim_or_package_not_model"})
                continue
            entry = merged.setdefault(norm, {
                "canonical_model_name": name.strip(),
                "aliases": [],
                "found_by_agents": [],
                "currently_sold": None,
                "confidence": "low",
                "sources": [],
            })
            if agent and agent not in entry["found_by_agents"]:
                entry["found_by_agents"].append(agent)
            source_url = item.get("source_url")
            if source_url and source_url not in entry["sources"]:
                entry["sources"].append(source_url)

    return {
        "manufacturer": manufacturer,
        "market": market,
        "period": period,
        "candidate_models": sorted(merged.values(), key=lambda x: x["canonical_model_name"].lower()),
        "rejected_candidates": rejected,
        "failed_agents": failed_agents,
    }


DISCOVERY_MODEL_ALLOWED_KEYS = {"model_name_en", "source_url"}


def strip_or_reject_extra_discovery_fields(parsed: Any) -> Optional[str]:
    """Strip enriched discovery fields so Discovery stays ultra-thin."""
    if not isinstance(parsed, dict):
        return "INVALID_DISCOVERY_SCHEMA"
    models = parsed.get("models")
    if not isinstance(models, list) or not models:
        return "INVALID_DISCOVERY_SCHEMA"
    max_models = DISCOVERY_MAX_MODELS.get(str(parsed.get("agent", "")), max(DISCOVERY_MAX_MODELS.values()))
    cleaned = []
    for item in models[:max_models]:
        if not isinstance(item, dict):
            return "INVALID_DISCOVERY_SCHEMA"
        name = item.get("model_name_en")
        if not name or not isinstance(name, str):
            return "INVALID_DISCOVERY_SCHEMA"
        source_url = item.get("source_url")
        if source_url is None and isinstance(item.get("sources"), list) and item.get("sources"):
            source_url = item["sources"][0]
        if source_url is not None and not isinstance(source_url, str):
            source_url = str(source_url)
        cleaned.append({"model_name_en": name.strip(), "source_url": source_url})
    allowed_top_level = {"agent", "manufacturer", "market", "period", "models"}
    for key in list(parsed.keys()):
        if key not in allowed_top_level:
            parsed.pop(key, None)
    parsed["models"] = cleaned
    return None


def validate_discovery_schema(parsed: Any) -> Optional[str]:
    return strip_or_reject_extra_discovery_fields(parsed)


def build_model_list_text(discovery_output: Any) -> str:
    """Convert discovery output in list or dict form into a numbered prompt-safe model list."""
    data = discovery_output
    if isinstance(data, dict):
        for key in ("models", "data", "result", "vehicles"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        return json.dumps(data, ensure_ascii=False, indent=2)

    lines: List[str] = []
    for idx, model in enumerate(data, start=1):
        if isinstance(model, dict):
            name_en = model.get("model_name_en") or model.get("name") or model.get("model")
            source_url = model.get("source_url")
            lines.append(f"{idx}. model_name_en={name_en}; source_url={source_url}")
        else:
            lines.append(f"{idx}. {model}")
    return "\n".join(lines)


def format_debug_json(obj: Any) -> str:
    """Pretty-print debug objects with valid JSON punctuation and Hebrew preserved."""
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _usage_tokens(response: Any) -> Tuple[int, int]:
    usage = getattr(response, "usage", None)
    if not usage:
        return 0, 0
    return int(getattr(usage, "prompt_tokens", 0) or 0), int(getattr(usage, "completion_tokens", 0) or 0)


def _message_content(message: Any) -> str:
    content = getattr(message, "content", "") or ""
    if isinstance(content, list):
        return "".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
    return str(content)



def is_kimi_concurrency_error(exc: Any) -> bool:
    text = str(exc or "").lower()
    status_code = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    return (
        status_code == 429
        or "http 429" in text
        or "error code: 429" in text
        or "max organization concurrency" in text
        or "rate_limit_reached_error" in text
    )


def _api_concurrency_payload(exc: Any, attempts: int, *, agent: str = "", phase: str = "") -> Dict[str, Any]:
    message = str(exc)
    return {
        "_error": "API_CONCURRENCY_LIMIT",
        "retryable": True,
        "message": message,
        "attempts": attempts,
        "agent": agent,
        "phase": phase,
        "finish_reason": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "raw_preview": message[:RAW_DEBUG_PREVIEW_CHARS],
    }

def moonshot_chat(
    api_key: str,
    messages: List[Dict[str, Any]],
    *,
    temperature: float,
    use_web_search: bool,
    response_format: Optional[Dict[str, str]] = None,
    max_tokens: int,
    agent_name: str = "",
    phase_name: str = "",
) -> Dict[str, Any]:
    """Call Kimi and handle Moonshot's server-side builtin $web_search echo loop."""
    if not max_tokens:
        raise ValueError("moonshot_chat requires max_tokens for every model call")
    client = OpenAI(api_key=api_key, base_url=MOONSHOT_BASE_URL)
    history = list(messages)
    total_input = 0
    total_output = 0
    finish_reason = None
    content = ""
    rounds = 0

    while finish_reason not in ("stop", "length") and rounds < MAX_TOOL_ROUNDS:
        rounds += 1
        kwargs: Dict[str, Any] = {
            "model": KIMI_MODEL,
            "messages": history,
            "temperature": 0.6 if temperature < 0.6 else temperature,
            "max_tokens": max_tokens,
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        if use_web_search:
            kwargs["tools"] = WEB_SEARCH_TOOL
        if response_format:
            kwargs["response_format"] = response_format

        with KIMI_CONCURRENCY_SEMAPHORE:
            response = client.chat.completions.create(**kwargs)
        in_tokens, out_tokens = _usage_tokens(response)
        total_input += in_tokens
        total_output += out_tokens

        choice = response.choices[0]
        finish_reason = choice.finish_reason
        message = choice.message
        content = _message_content(message)

        if finish_reason == "tool_calls":
            history.append(message.model_dump(exclude_none=True))
            for tool_call in message.tool_calls or []:
                args = json.loads(tool_call.function.arguments or "{}")
                history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": json.dumps(args, ensure_ascii=False),
                })
            continue

        if finish_reason in {"stop", "length"}:
            break

    # Retry if agent talked about searching but never actually searched.
    if use_web_search and rounds <= 2 and finish_reason == "stop" and not any(
        m.get("role") == "tool" for m in history if isinstance(m, dict)
    ):
        history.append({
            "role": "user",
            "content": "You did not use the $web_search tool. You MUST search the web now. Do not describe what to search — call the tool directly.",
        })
        finish_reason = None
        while finish_reason not in ("stop", "length") and rounds < MAX_TOOL_ROUNDS:
            rounds += 1
            kwargs = {
                "model": KIMI_MODEL,
                "messages": history,
                "temperature": 0.6 if temperature < 0.6 else temperature,
                "max_tokens": max_tokens,
                "extra_body": {"thinking": {"type": "disabled"}},
                "tools": WEB_SEARCH_TOOL,
            }
            if response_format:
                kwargs["response_format"] = response_format

            with KIMI_CONCURRENCY_SEMAPHORE:
                response = client.chat.completions.create(**kwargs)
            in_tokens, out_tokens = _usage_tokens(response)
            total_input += in_tokens
            total_output += out_tokens

            choice = response.choices[0]
            finish_reason = choice.finish_reason
            message = choice.message
            content = _message_content(message)

            if finish_reason == "tool_calls":
                history.append(message.model_dump(exclude_none=True))
                for tool_call in message.tool_calls or []:
                    args = json.loads(tool_call.function.arguments or "{}")
                    history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": json.dumps(args, ensure_ascii=False),
                    })
                continue

            if finish_reason in {"stop", "length"}:
                break

    return {
        "content": content,
        "finish_reason": finish_reason,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "parsed": None,
        "agent": agent_name,
        "phase": phase_name,
    }


def discovery_prompt(agent: AgentConfig, manufacturer: str, market: str, period: str, retry: bool = False) -> List[Dict[str, str]]:
    max_models = DISCOVERY_MAX_MODELS.get(agent.key, 25)
    if retry:
        system = MANDATORY_WEB_SEARCH_INSTRUCTION + f"You are {agent.key}. JSON only. No prose. Extra fields are forbidden."
        user = f'''Return only this compact JSON object:
{{
  "agent": "{agent.key}",
  "models": [
    {{"model_name_en": "...", "source_url": "..."}}
  ]
}}
No other fields. Maximum {max_models} models. Scope: {agent.responsibility}'''
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    system = MANDATORY_WEB_SEARCH_INSTRUCTION + f"You are {agent.key}. Find Hyundai model names for {market}. JSON only; no planning text. Extra model fields are forbidden."
    user = f'''Manufacturer: {manufacturer}
Market: {market}
Period: {period}
Scope: {agent.responsibility}
Maximum models: {max_models}
Return only:
{{"agent":"{agent.key}","manufacturer":"{manufacturer}","market":"{market}","period":"{period}","models":[{{"model_name_en":"string","source_url":"string|null"}}]}}
No other model fields are allowed.'''
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalizer_prompt(merged: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are normalizer_deduper. JSON only. No web search. Clean candidate model names only: normalize names, merge aliases, reject trims, separate distinct models, and flag uncertainty. Do not add technical data. Hybrid/Electric/PHEV usually become variant notes under a base model unless marketed as standalone; keep Ioniq numbered families and Nexo separate; N Line is a trim/package; uncertain market names and global-only models go to needs_review."},
        {"role": "user", "content": "Clean this merged candidate JSON and return schema {\"agent\":\"normalizer_deduper\",\"canonical_models\":[{\"canonical_model_name\":\"string\",\"model_name_he\":\"string|null\",\"aliases\":[\"string\"],\"currently_sold\":true/false/null,\"confidence\":\"high|medium|low\",\"sources\":[\"url\"]}],\"rejected_items\":[{\"name\":\"string\",\"reason\":\"trim_or_package_not_model\"}],\"needs_review\":[{\"name\":\"string\",\"reason\":\"string\"}]}. Do not output model_he; use model_name_he.\n" + json.dumps(merged, ensure_ascii=False, indent=2)},
    ]


def technical_prompt(agent: AgentConfig, manufacturer: str, market: str, period: str, canonical_models: List[Dict[str, Any]], retry: bool = False) -> List[Dict[str, str]]:
    strict = "Your previous response was invalid. Return compact JSON only. Do not describe searching. Use missing_data if unsure.\n" if retry else ""
    schema_by_agent = {
        "trims_years_agent": '{"agent":"trims_years_agent","items":[{"model":"string","years_sold":"string|null","generation_or_series":"string|null","trims":["string"],"confidence":"high|medium|low","sources":["url"],"notes":"string|null"}],"missing_data":[{"model":"string","field":"string","reason":"not_found|conflicting_sources|not_applicable"}],"extra_candidate_models":[]}',
        "engines_fuel_power_agent": '{"agent":"engines_fuel_power_agent","items":[{"model":"string","years":"string|null","variant_or_generation":"string|null","engine":"string|null","fuel_type":"petrol|diesel|hybrid|plug_in_hybrid|electric|fuel_cell|unknown|null","power_hp":0,"torque_nm":0,"confidence":"high|medium|low","sources":["url"],"notes":"string|null"}],"missing_data":[],"extra_candidate_models":[]}',
        "transmission_drivetrain_performance_agent": '{"agent":"transmission_drivetrain_performance_agent","items":[{"model":"string","years":"string|null","variant_or_generation":"string|null","transmission":"string|null","drivetrain":"FWD|RWD|AWD|4WD|unknown|null","zero_to_100_kmh_sec":0,"confidence":"high|medium|low","sources":["url"],"notes":"string|null"}],"missing_data":[],"extra_candidate_models":[]}',
        "dimensions_safety_equipment_agent": '{"agent":"dimensions_safety_equipment_agent","items":[{"model":"string","years":"string|null","body_type":"string|null","seats":0,"trunk_liters":0,"length_mm":0,"width_mm":0,"height_mm":0,"safety":"string|null","equipment_notes":"string|null","confidence":"high|medium|low","sources":["url"],"notes":"string|null"}],"missing_data":[],"extra_candidate_models":[]}',
    }
    max_items = TECHNICAL_MAX_ITEMS_PER_MODEL[agent.key]
    compact_models = compact_technical_models(canonical_models)
    system = MANDATORY_WEB_SEARCH_INSTRUCTION + f"""You are {agent.name}, an Israeli-market automotive data enrichment researcher.
{ISRAEL_ENRICHMENT_CONTEXT}
{strict}ONLY research models in the provided canonical model list.
Do not add new models directly; put surprises in extra_candidate_models.
Do not invent. If global data is all you find, set confidence low/medium.
Return flat compact items only. Max {max_items} item(s) per model. Choose representative Israel-market variants; summarize omitted variants briefly.
Do not enumerate every year, every global trim, or every global engine. Max 2 sources per item. Notes max 160 characters. Trims max 6 strings.
Forbidden output keys: trims_by_year, engines_fuel_power, engine_code, displacement_cc, cylinders, aspiration, transmission_drivetrain_performance, dimensions, safety_equipment.
Agent name: {agent.key}
Output ONLY a valid JSON object that matches the exact JSON schema. No markdown, no explanations.
Never return generic automotive glossary keys: engine_types, transmission_types, drivetrain_configs, safety_systems, body_types."""
    user = f"""Manufacturer: {manufacturer}
Market: {market}
Period: {period}
Agent name: {agent.key}
Concrete canonical model chunk:
{json.dumps(compact_models, ensure_ascii=False, indent=2)}

Responsibility: {agent.responsibility}
Limits: max {max_items} item(s) per model; max 2 sources; notes <=160 chars; trims <=6. No nested objects.
Return exactly this JSON schema shape: {schema_by_agent[agent.key]}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def final_builder_prompt(normalized: Any, technical: Dict[str, Any], verifier: Any, failed_summaries: List[Dict[str, Any]], manufacturer: str, market: str, period: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": f"You are final_builder. JSON only. No web search. Build final JSON from clean structured inputs only. Never invent facts.\n{CONSOLIDATION_CONTEXT}"},
        {"role": "user", "content": f"Manufacturer: {manufacturer}\nMarket: {market}\nPeriod: {period}\nReturn final schema with manufacturer, market, period, status, models, needs_review, rejected, failed_agents, token_usage. Inputs:\nNormalized:\n{json.dumps(normalized, ensure_ascii=False, indent=2)}\nTechnical:\n{json.dumps(technical, ensure_ascii=False, indent=2)}\nVerifier:\n{json.dumps(verifier, ensure_ascii=False, indent=2)}\nFailed agent summaries:\n{json.dumps(failed_summaries, ensure_ascii=False, indent=2)}"},
    ]


def summary_prompt(consolidated: Any) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are Hebrew Summary Agent. Write a concise Hebrew user-facing summary. No web search. Include model count, verified count, needs-review count, technical completeness/partials, and failed agents. Do not change JSON data."},
        {"role": "user", "content": json.dumps(consolidated, ensure_ascii=False, indent=2)},
    ]


def init_state() -> None:
    defaults = {"results": {}, "consolidated": None, "summary": "", "discovery_data": None, "input_tokens": 0, "output_tokens": 0, "elapsed": 0.0}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def add_tokens(result: Dict[str, Any]) -> None:
    st.session_state.input_tokens += result.get("input_tokens", 0)
    st.session_state.output_tokens += result.get("output_tokens", 0)


def reset_state() -> None:
    for key in ("results", "consolidated", "summary", "discovery_data", "input_tokens", "output_tokens", "elapsed"):
        st.session_state.pop(key, None)
    init_state()


def count_models_trims(consolidated: Any) -> Tuple[int, int]:
    models = consolidated.get("models", []) if isinstance(consolidated, dict) else []
    trim_count = sum(len(m.get("trims") or []) for m in models if isinstance(m, dict))
    return len(models), trim_count


def render_sidebar() -> str:
    st.sidebar.title("Kimi Swarm Prototype")
    env_key = os.getenv("MOONSHOT_API_KEY", "")
    api_key = st.sidebar.text_input("Moonshot API key", value=env_key, type="password", help="Reads MOONSHOT_API_KEY by default.")
    st.sidebar.subheader("Agents")
    st.sidebar.markdown(
        "- Discovery:\n" + "\n".join(f"  - {a.key}: {a.description}" for a in DISCOVERY_AGENTS) +
        "\n- Normalizer / deduper\n- Technical enrichment:\n" + "\n".join(f"  - {a.key}: {a.description}" for a in TECHNICAL_AGENTS) +
        "\n- Source verifier\n- Final builder\n- Hebrew summary"
    )
    st.sidebar.subheader("Architecture")
    st.sidebar.code(ARCHITECTURE_ASCII)
    return api_key


def render_persistent_outputs() -> None:
    if not st.session_state.results and not st.session_state.consolidated and not st.session_state.summary:
        return
    st.divider()
    st.header("Persistent display")
    tab_raw, tab_consolidated, tab_summary = st.tabs(["Raw agent JSON", "Consolidated JSON", "Summary"])
    with tab_raw:
        st.code(format_debug_json(st.session_state.results), language="json")
    with tab_consolidated:
        st.json(st.session_state.consolidated)
        if st.session_state.consolidated is not None:
            st.download_button("Download consolidated JSON", json.dumps(st.session_state.consolidated, ensure_ascii=False, indent=2), "consolidated_vehicle_models.json", "application/json")
    with tab_summary:
        st.markdown(st.session_state.summary or "_No summary yet._")



RETRYABLE_ERRORS = {"MODEL_PLANNING_LOOP", "MODEL_REPETITION_LOOP", "INVALID_JSON", "MODEL_OUTPUT_TRUNCATED", "MODEL_JSON_TRUNCATED"}

GENERIC_AUTOMOTIVE_TOP_LEVEL_KEYS = {
    "engine_types",
    "transmission_types",
    "drivetrain_configs",
    "safety_systems",
    "body_types",
}

MISSING_MAKE_MODEL_MARKERS = (
    "missing make/model",
    "missing make and model",
    "provide make/model",
    "provide a make/model",
    "provide the make and model",
    "need the make and model",
    "need a make and model",
    "please specify the make",
    "please provide the make",
)

TECHNICAL_VERBOSE_KEYS = {
    "trims_by_year",
    "engines_fuel_power",
    "engine_code",
    "displacement_cc",
    "cylinders",
    "aspiration",
    "transmission_drivetrain_performance",
    "dimensions",
    "safety_equipment",
}

TECHNICAL_ALLOWED_ITEM_KEYS: Dict[str, set[str]] = {
    "trims_years_agent": {"model", "years_sold", "generation_or_series", "trims", "confidence", "sources", "notes"},
    "engines_fuel_power_agent": {"model", "years", "variant_or_generation", "engine", "fuel_type", "power_hp", "torque_nm", "confidence", "sources", "notes"},
    "transmission_drivetrain_performance_agent": {"model", "years", "variant_or_generation", "transmission", "drivetrain", "zero_to_100_kmh_sec", "confidence", "sources", "notes"},
    "dimensions_safety_equipment_agent": {"model", "years", "body_type", "seats", "trunk_liters", "length_mm", "width_mm", "height_mm", "safety", "equipment_notes", "confidence", "sources", "notes"},
}

TECHNICAL_MAX_ITEMS_PER_MODEL = {
    "trims_years_agent": 1,
    "engines_fuel_power_agent": 2,
    "transmission_drivetrain_performance_agent": 1,
    "dimensions_safety_equipment_agent": 1,
}

TECHNICAL_FALLBACK_TOKENS = {
    "trims_years_agent": 1000,
    "engines_fuel_power_agent": 1200,
    "transmission_drivetrain_performance_agent": 1000,
    "dimensions_safety_equipment_agent": 1000,
}


def validate_normalizer_schema(parsed: Any) -> Optional[str]:
    if not isinstance(parsed, dict) or not isinstance(parsed.get("canonical_models"), list):
        return "INVALID_NORMALIZER_SCHEMA"
    if not isinstance(parsed.get("rejected_items"), list) or not isinstance(parsed.get("needs_review"), list):
        return "INVALID_NORMALIZER_SCHEMA"
    for item in parsed["canonical_models"]:
        if not isinstance(item, dict) or not item.get("canonical_model_name"):
            return "INVALID_NORMALIZER_SCHEMA"
        if "model_he" in item and "model_name_he" not in item:
            item["model_name_he"] = item.pop("model_he")
        item.setdefault("model_name_he", None)
        if not isinstance(item.get("sources"), list):
            item["sources"] = []
    if len(parsed["canonical_models"]) > 45:
        parsed.setdefault("warnings", []).append("TOO_MANY_CANONICAL_MODELS_REVIEW_REQUIRED")
    return None


def validate_items_schema(parsed: Any) -> Optional[str]:
    if not isinstance(parsed, dict):
        return "INVALID_AGENT_SCHEMA"
    if TECHNICAL_VERBOSE_KEYS.intersection(parsed.keys()):
        return "TECHNICAL_OUTPUT_TOO_VERBOSE"
    for key in ("items", "missing_data", "extra_candidate_models"):
        if not isinstance(parsed.get(key), list):
            return f"INVALID_AGENT_SCHEMA:{key}"
    agent = str(parsed.get("agent", ""))
    allowed = TECHNICAL_ALLOWED_ITEM_KEYS.get(agent)
    per_model: Dict[str, int] = {}
    for item in parsed.get("items", []):
        if not isinstance(item, dict):
            return "INVALID_AGENT_SCHEMA:item_model"
        if TECHNICAL_VERBOSE_KEYS.intersection(item.keys()):
            return "TECHNICAL_OUTPUT_TOO_VERBOSE"
        if "canonical_model_name" in item and not item.get("model"):
            item["model"] = item.pop("canonical_model_name")
        if not item.get("model"):
            return "INVALID_AGENT_SCHEMA:item_model"
        if allowed:
            for key in list(item.keys()):
                if key not in allowed:
                    item.pop(key, None)
        if not isinstance(item.get("sources"), list):
            item["sources"] = []
        item["sources"] = item["sources"][:2]
        if "notes" in item and isinstance(item.get("notes"), str):
            item["notes"] = item["notes"][:160]
        if "trims" in item:
            item["trims"] = item["trims"][:6] if isinstance(item.get("trims"), list) else []
        model = str(item.get("model"))
        per_model[model] = per_model.get(model, 0) + 1
        if per_model[model] > TECHNICAL_MAX_ITEMS_PER_MODEL.get(agent, 1):
            return "TECHNICAL_OUTPUT_TOO_VERBOSE"
    return None


def validate_verifier_schema(parsed: Any) -> Optional[str]:
    if not isinstance(parsed, dict):
        return "INVALID_VERIFIER_SCHEMA"
    for key in ("verified_models", "rejected_data_points", "needs_review"):
        if not isinstance(parsed.get(key), list):
            return f"INVALID_VERIFIER_SCHEMA:{key}"
    global_markers = ("hyundaiksa.com", "hyundainews.com", "motortrend.com", "jdpower.com")
    for item in parsed.get("verified_models", []):
        if not isinstance(item, dict):
            return "INVALID_VERIFIER_SCHEMA:model"
        item.setdefault("source_region", "unknown")
        item.setdefault("source_strength", "unknown")
        issues = item.setdefault("issues", [])
        issue_text = " ".join(str(x).lower() for x in issues)
        if item.get("source_region") in {"global", "foreign_market"} or item.get("source_strength") in {"global_official", "foreign_market"} or any(m in issue_text for m in global_markers):
            item["status"] = "needs_review"
            if item.get("confidence") == "high":
                item["confidence"] = "medium"
    return None


def validate_final_schema(parsed: Any) -> Optional[str]:
    if not isinstance(parsed, dict):
        return "INVALID_FINAL_SCHEMA"
    for key in ("manufacturer", "market", "period", "status", "models", "needs_review", "rejected", "failed_agents", "token_usage"):
        if key not in parsed:
            return f"INVALID_FINAL_SCHEMA:{key}"
    if not isinstance(parsed.get("models"), list):
        return "INVALID_FINAL_SCHEMA:models"
    return None


def safe_agent_result(agent: str, phase: str, checked: Dict[str, Any], *, used_fallback: bool = False, api_retry_count: int = 0, allow_partial: bool = False) -> Dict[str, Any]:
    if checked.get("_error"):
        status = "partial" if allow_partial and checked.get("_error") in {"API_CONCURRENCY_LIMIT"} else "failed"
        return phase_result(status=status, agent=agent, parsed=None, error=checked["_error"], raw_preview=checked.get("raw_preview", ""), finish_reason=checked.get("finish_reason"), input_tokens=checked.get("input_tokens", 0), output_tokens=checked.get("output_tokens", 0), used_fallback=used_fallback, api_retry_count=api_retry_count, message=checked.get("message", ""))
    return phase_result(status="success", agent=agent, parsed=checked.get("parsed"), finish_reason=checked.get("finish_reason"), input_tokens=checked.get("input_tokens", 0), output_tokens=checked.get("output_tokens", 0), used_fallback=used_fallback, api_retry_count=api_retry_count)


def run_safe_agent(
    api_key: str,
    *,
    agent_name: str,
    phase_name: str,
    prompt: List[Dict[str, Any]],
    max_tokens: int,
    required_top_keys: List[str],
    fallback_max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, str]] = {"type": "json_object"},
    fallback_prompt: Optional[List[Dict[str, Any]]] = None,
    allow_partial: bool = False,
    use_web_search: bool = False,
    validator: Optional[Any] = None,
    non_empty_lists: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if not max_tokens:
        raise ValueError("run_safe_agent requires max_tokens")

    def attempt(messages: List[Dict[str, Any]], phase: str, token_limit: int) -> Dict[str, Any]:
        attempts = 0
        while True:
            try:
                return moonshot_chat(api_key, messages, temperature=0.6, use_web_search=use_web_search, response_format=response_format, max_tokens=token_limit, agent_name=agent_name, phase_name=phase)
            except Exception as exc:  # noqa: BLE001 - API errors must be classified for UI/tests.
                if is_kimi_concurrency_error(exc):
                    if attempts < API_CONCURRENCY_MAX_RETRIES:
                        attempts += 1
                        time.sleep(API_CONCURRENCY_RETRY_DELAY_SECONDS)
                        continue
                    payload = _api_concurrency_payload(exc, attempts + 1, agent=agent_name, phase=phase)
                    payload["api_retry_count"] = attempts
                    return payload
                raise

    raw = attempt(prompt, phase_name, max_tokens)
    if raw.get("_error") == "API_CONCURRENCY_LIMIT":
        return safe_agent_result(agent_name, phase_name, raw, api_retry_count=raw.get("api_retry_count", API_CONCURRENCY_MAX_RETRIES), allow_partial=allow_partial)
    checked = validate_model_response(raw, require_json=response_format is not None, validator=validator, required_keys=required_top_keys, non_empty_lists=non_empty_lists)
    if checked.get("_error") in RETRYABLE_ERRORS and fallback_prompt is not None:
        fallback_raw = attempt(fallback_prompt, f"{phase_name}_fallback", fallback_max_tokens or max_tokens)
        if fallback_raw.get("_error") == "API_CONCURRENCY_LIMIT":
            return safe_agent_result(agent_name, phase_name, fallback_raw, api_retry_count=fallback_raw.get("api_retry_count", API_CONCURRENCY_MAX_RETRIES), allow_partial=allow_partial)
        fallback_checked = validate_model_response(fallback_raw, require_json=response_format is not None, validator=validator, required_keys=required_top_keys, non_empty_lists=non_empty_lists)
        fallback_checked["input_tokens"] = fallback_checked.get("input_tokens", 0) + checked.get("input_tokens", 0)
        fallback_checked["output_tokens"] = fallback_checked.get("output_tokens", 0) + checked.get("output_tokens", 0)
        return safe_agent_result(agent_name, phase_name, fallback_checked, used_fallback=True, allow_partial=allow_partial)
    return safe_agent_result(agent_name, phase_name, checked, allow_partial=allow_partial)


def run_discovery_agent(api_key: str, agent: AgentConfig, manufacturer: str, market: str, period: str) -> Dict[str, Any]:
    return run_safe_agent(
        api_key, agent_name=agent.key, phase_name="discovery", prompt=discovery_prompt(agent, manufacturer, market, period),
        max_tokens=MAX_DISCOVERY_TOKENS, required_top_keys=["agent", "models"], fallback_prompt=discovery_prompt(agent, manufacturer, market, period, retry=True),
        use_web_search=True, validator=validate_discovery_schema, non_empty_lists=["models"]
    )


def run_discovery_phase(api_key: str, manufacturer: str, market: str, period: str) -> List[Dict[str, Any]]:
    return [run_discovery_agent(api_key, agent, manufacturer, market, period) for agent in DISCOVERY_AGENTS]


def run_normalizer_phase(api_key: str, merged: Dict[str, Any]) -> Dict[str, Any]:
    return run_safe_agent(api_key, agent_name="normalizer_deduper", phase_name="normalizer", prompt=normalizer_prompt(merged), max_tokens=2500, required_top_keys=["agent", "canonical_models", "rejected_items", "needs_review"], use_web_search=False, validator=validate_normalizer_schema, non_empty_lists=["canonical_models"])


def technical_max_tokens(agent_key: str) -> int:
    return 2500 if agent_key == "trims_years_agent" else 3000


def compact_technical_models(canonical_models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact = []
    for model in canonical_models:
        if not isinstance(model, dict):
            continue
        compact.append({
            "canonical_model_name": model.get("canonical_model_name"),
            "model_name_he": model.get("model_name_he"),
            "aliases": (model.get("aliases") or [])[:3] if isinstance(model.get("aliases"), list) else [],
            "sources": (model.get("sources") or [])[:2] if isinstance(model.get("sources"), list) else [],
        })
    return compact


def technical_fallback_prompt(agent: AgentConfig, manufacturer: str, market: str, period: str, canonical_models: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    empty_schema = {"agent": agent.key, "items": [], "missing_data": [], "extra_candidate_models": []}
    task = "trims and years" if agent.key == "trims_years_agent" else "your assigned automotive facts"
    compact_models = compact_technical_models(canonical_models)
    return [
        {"role": "system", "content": MANDATORY_WEB_SEARCH_INSTRUCTION + f"You are {agent.key}. Return compact JSON only. For each model, return at most one item. Do not list data by year. Do not list every trim. Do not list every engine. Do not output nested objects. Use null/0 when unknown. Keep notes short. No explanation. Do not return generic automotive glossary keys such as engine_types, transmission_types, drivetrain_configs, safety_systems, or body_types."},
        {"role": "user", "content": f"Manufacturer: {manufacturer}\nMarket: {market}\nPeriod: {period}\nAgent name: {agent.key}\nConcrete canonical model chunk:\n{json.dumps(compact_models, ensure_ascii=False, indent=2)}\nTask: Return compact JSON only for {task} for the provided canonical model chunk. One compact item per model. Max 2 sources per item. Notes max 160 characters. No nested objects or per-year arrays. Required exact JSON schema: " + json.dumps(empty_schema, ensure_ascii=False)},
    ]


def run_technical_agent(api_key: str, agent: AgentConfig, manufacturer: str, market: str, period: str, canonical_models: List[Dict[str, Any]]) -> Dict[str, Any]:
    fallback = technical_fallback_prompt(agent, manufacturer, market, period, canonical_models)
    return run_safe_agent(api_key, agent_name=agent.key, phase_name="technical", prompt=technical_prompt(agent, manufacturer, market, period, canonical_models), max_tokens=technical_max_tokens(agent.key), fallback_max_tokens=TECHNICAL_FALLBACK_TOKENS[agent.key], required_top_keys=["agent", "items", "missing_data", "extra_candidate_models"], fallback_prompt=fallback, allow_partial=True, use_web_search=True, validator=validate_items_schema)


def merge_chunk_results(agent_name: str, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    parsed = {"agent": agent_name, "items": [], "missing_data": [], "extra_candidate_models": []}
    failed_chunks = []
    input_tokens = output_tokens = 0
    for result in chunk_results:
        input_tokens += result.get("input_tokens", 0)
        output_tokens += result.get("output_tokens", 0)
        if result.get("status") not in {"success", "partial"}:
            failed_chunks.append({"agent": result.get("agent", agent_name), "error": result.get("error")})
            continue
        data = result.get("parsed") or {}
        parsed["agent"] = data.get("agent", agent_name) if isinstance(data, dict) else agent_name
        if isinstance(data, dict):
            for key in ("items", "missing_data", "extra_candidate_models"):
                if isinstance(data.get(key), list):
                    parsed[key].extend(data[key])
    status = "success" if not failed_chunks else ("partial" if parsed["items"] or parsed["missing_data"] or parsed["extra_candidate_models"] else "failed")
    return {
        "status": status,
        "agent": agent_name,
        "phase": "technical_enrichment",
        "parsed": parsed,
        "error": None if status != "failed" else "TECHNICAL_CHUNKS_FAILED",
        "failed_chunks": failed_chunks,
        "token_usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def run_technical_enrichment_phase(api_key: str, manufacturer: str, market: str, period: str, canonical_models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Stable policy: run Phase 3 technical agents sequentially to avoid Kimi org concurrency=3 failures.
    results = []
    for agent in TECHNICAL_AGENTS:
        chunk_results = []
        for i in range(0, len(canonical_models), TECHNICAL_MODEL_CHUNK_SIZE):
            chunk = canonical_models[i:i + TECHNICAL_MODEL_CHUNK_SIZE]
            chunk_results.append(run_technical_agent(api_key, agent, manufacturer, market, period, chunk))
        results.append(merge_chunk_results(agent.key, chunk_results))
    return results


def compact_verifier_input(normalized: Any, technical: Dict[str, Any], failed_summaries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    canonical_models = []
    for model in (normalized or {}).get("canonical_models", []) if isinstance(normalized, dict) else []:
        if isinstance(model, dict):
            canonical_models.append({
                "canonical_model_name": model.get("canonical_model_name"),
                "model_name_he": model.get("model_name_he"),
                "sources": model.get("sources", []),
            })
    technical_summaries: Dict[str, Any] = {}
    for agent, parsed in (technical or {}).items():
        if not isinstance(parsed, dict):
            continue
        items = []
        for item in parsed.get("items", []):
            if isinstance(item, dict):
                items.append({"model": item.get("model"), "confidence": item.get("confidence"), "sources": item.get("sources", []), "fields": sorted(k for k, v in item.items() if k not in {"sources", "notes"} and v not in (None, "", [], {}))})
        technical_summaries[agent] = {"agent": parsed.get("agent", agent), "items": items, "missing_data_count": len(parsed.get("missing_data", [])), "extra_candidate_models_count": len(parsed.get("extra_candidate_models", []))}
    return {"canonical_models": canonical_models, "technical_summaries": technical_summaries, "failed_summaries": compact_failed_summaries(failed_summaries or [])}


def verifier_prompt(normalized: Any, technical: Dict[str, Any], failed_summaries: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, str]]:
    compact_input = compact_verifier_input(normalized, technical, failed_summaries)
    return [
        {"role": "system", "content": "You are source_verifier. JSON only. No broad new research. Review existing compact structured JSON only; do not invent missing data. If sources are only global or foreign-market, mark needs_review rather than verified."},
        {"role": "user", "content": "Verify Israel-market relevance, model-vs-trim status, contradictions, unsupported data, source_region, and source_strength. Return schema {\"agent\":\"source_verifier\",\"verified_models\":[],\"rejected_data_points\":[],\"needs_review\":[]} with compact objects in those lists. Compact input:\n" + json.dumps(compact_input, ensure_ascii=False, indent=2)},
    ]


def _chunk_models(normalized: Any, chunk_size: int = VERIFIER_MODEL_CHUNK_SIZE) -> List[Dict[str, Any]]:
    models = (normalized or {}).get("canonical_models", []) if isinstance(normalized, dict) else []
    if not models:
        return [normalized]
    chunks = []
    for i in range(0, len(models), chunk_size):
        chunk = dict(normalized)
        chunk["canonical_models"] = models[i:i + chunk_size]
        chunks.append(chunk)
    return chunks


def merge_verifier_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    parsed = {"agent": "source_verifier", "verified_models": [], "rejected_data_points": [], "needs_review": []}
    failed = []
    input_tokens = output_tokens = 0
    for result in results:
        input_tokens += result.get("input_tokens", 0)
        output_tokens += result.get("output_tokens", 0)
        if result.get("status") not in {"success", "partial"}:
            failed.append({"agent": result.get("agent"), "error": result.get("error")})
            continue
        data = result.get("parsed") or {}
        for key in ("verified_models", "rejected_data_points", "needs_review"):
            if isinstance(data.get(key), list):
                parsed[key].extend(data[key])
    status = "success" if not failed else ("partial" if any(r.get("status") in {"success", "partial"} for r in results) else "failed")
    return {"status": status, "agent": "source_verifier", "phase": "verification", "parsed": parsed if status != "failed" else None, "error": "VERIFIER_CHUNK_FAILED" if status == "partial" else ("VERIFIER_FAILED" if status == "failed" else None), "failed_chunks": failed, "input_tokens": input_tokens, "output_tokens": output_tokens}


def run_verification_phase(api_key: str, normalized: Any, technical: Dict[str, Any], failed_summaries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    chunks = _chunk_models(normalized)
    results = [run_safe_agent(api_key, agent_name="source_verifier", phase_name="verification", prompt=verifier_prompt(chunk, technical, failed_summaries), max_tokens=MAX_VERIFIER_TOKENS, required_top_keys=["agent", "verified_models", "rejected_data_points", "needs_review"], use_web_search=False, validator=validate_verifier_schema) for chunk in chunks]
    return merge_verifier_results(results)


def compact_failed_summaries(failed_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"agent": item.get("agent"), "error": item.get("error"), "message": item.get("message", "")} for item in failed_summaries]


def run_final_builder_phase(api_key: str, normalized: Any, technical: Dict[str, Any], verifier: Any, failed_summaries: List[Dict[str, Any]], manufacturer: str, market: str, period: str) -> Dict[str, Any]:
    safe_failed = compact_failed_summaries(failed_summaries)
    result = run_safe_agent(api_key, agent_name="final_builder", phase_name="final_builder", prompt=final_builder_prompt(normalized, technical, verifier, safe_failed, manufacturer, market, period), max_tokens=MAX_FINAL_BUILDER_TOKENS, required_top_keys=["manufacturer", "market", "period", "status", "models", "needs_review", "rejected", "failed_agents", "token_usage"], use_web_search=False, validator=validate_final_schema)
    if result.get("status") == "success" and safe_failed and result.get("parsed", {}).get("status") == "complete":
        result["parsed"]["status"] = "partial"
    return result


def run_hebrew_summary_phase(api_key: str, final_json: Any) -> Dict[str, Any]:
    try:
        result = moonshot_chat(api_key, summary_prompt(final_json), temperature=0.6, use_web_search=False, max_tokens=MAX_SUMMARY_TOKENS, agent_name="hebrew_summary", phase_name="summary")
        checked = validate_model_response(result, require_json=False)
    except Exception as exc:  # noqa: BLE001
        if is_kimi_concurrency_error(exc):
            checked = _api_concurrency_payload(exc, 1, agent="hebrew_summary", phase="summary")
        else:
            raise
    return safe_agent_result("hebrew_summary", "summary", checked) if checked.get("_error") else phase_result(status="success", agent="hebrew_summary", parsed={"summary": checked.get("content", "")}, finish_reason=checked.get("finish_reason"), input_tokens=checked.get("input_tokens", 0), output_tokens=checked.get("output_tokens", 0))

def run_pipeline(api_key: str, manufacturer: str, market: str, period: str) -> None:
    start = time.perf_counter()
    st.subheader("Phase 1 — focused discovery")
    discovery_results = run_discovery_phase(api_key, manufacturer, market, period)
    for r in discovery_results:
        add_tokens(r)
        st.write(f"{r['agent']}: {r['status']}" + (f" — {r['error']}" if r.get("error") else ""))
        if r.get("raw_preview"):
            st.code(format_debug_json(r), language="json")
    st.session_state.results["discovery_phase"] = discovery_results
    successful_discovery = [r for r in discovery_results if r["status"] == "success"]
    if not successful_discovery:
        reason = discovery_results[0].get("error") if discovery_results else "NO_DISCOVERY_RESULTS"
        message = discovery_results[0].get("message") if discovery_results else ""
        st.error(f"Discovery failed: {reason}" + (f"\n\n{message}" if message else ""))
        return

    merged = merge_discovery_candidates(discovery_results)
    st.session_state.results["python_discovery_merge"] = merged
    st.success(f"Discovery merge complete: {len(merged['candidate_models'])} candidate models.")

    st.subheader("Phase 2 — normalizer / deduper")
    normalizer = run_normalizer_phase(api_key, merged)
    add_tokens(normalizer)
    st.session_state.results["normalizer_deduper"] = normalizer
    if normalizer["status"] != "success":
        st.error(f"Normalizer failed: {normalizer['error']}")
        return
    canonical_models = normalizer["parsed"].get("canonical_models", [])
    st.session_state.discovery_data = normalizer["parsed"]

    st.subheader("Phase 3 — technical enrichment")
    technical_results = run_technical_enrichment_phase(api_key, manufacturer, market, period, canonical_models)
    technical_clean: Dict[str, Any] = {}
    failed_summaries: List[Dict[str, Any]] = []
    for r in technical_results:
        add_tokens(r)
        st.write(f"{r['agent']}: {r['status']}" + (f" — {r['error']}" if r.get("error") else ""))
        if r["status"] in {"success", "partial"} and isinstance(r.get("parsed"), dict):
            technical_clean[r["agent"]] = r["parsed"]
        else:
            failed_summaries.append({"agent": r["agent"], "error": r["error"], "message": r.get("message", "")})
    st.session_state.results["technical_enrichment_phase"] = technical_results

    if not technical_clean:
        st.error("Pipeline failed: TECHNICAL_ENRICHMENT_FAILED\nReason: all technical enrichment agents failed.")
        return

    st.subheader("Phase 4 — verifier")
    verifier = run_verification_phase(api_key, normalizer["parsed"], technical_clean, failed_summaries)
    add_tokens(verifier)
    st.session_state.results["source_verifier"] = verifier
    if verifier["status"] == "success":
        verifier_data = verifier["parsed"]
    elif verifier["status"] == "partial" and isinstance(verifier.get("parsed"), dict):
        failed_summaries.append({"agent": "source_verifier", "error": verifier["error"], "message": verifier.get("message", "")})
        verifier_data = verifier["parsed"]
        st.warning(f"Verifier partially failed: {verifier['error']}; continuing partial.")
    else:
        failed_summaries.append({"agent": "source_verifier", "error": verifier["error"], "message": verifier.get("message", "")})
        verifier_data = {"agent": "source_verifier", "verified_models": [], "rejected_data_points": [], "needs_review": [{"model": "*", "reason": verifier["error"]}]}
        st.warning(f"Verifier failed: {verifier['error']}; continuing partial.")

    st.subheader("Phase 5 — final builder")
    final = run_final_builder_phase(api_key, normalizer["parsed"], technical_clean, verifier_data, failed_summaries, manufacturer, market, period)
    add_tokens(final)
    st.session_state.results["final_builder"] = final
    if final["status"] != "success":
        st.error(f"Final builder failed: {final['error']}")
        return
    st.session_state.consolidated = final["parsed"]
    st.success(f"Final JSON complete with status: {final['parsed'].get('status')}")

    st.subheader("Phase 6 — Hebrew summary")
    summary = run_hebrew_summary_phase(api_key, final["parsed"])
    add_tokens(summary)
    st.session_state.results["hebrew_summary"] = summary
    if summary["status"] == "success":
        st.session_state.summary = summary["parsed"]["summary"]
        st.markdown(st.session_state.summary)
    else:
        st.warning(f"Summary failed: {summary['error']}")

    st.session_state.elapsed = time.perf_counter() - start
    estimated_cost = (st.session_state.input_tokens / 1_000_000 * INPUT_COST_PER_1M) + (st.session_state.output_tokens / 1_000_000 * OUTPUT_COST_PER_1M)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Elapsed time", f"{st.session_state.elapsed:.1f}s")
    c2.metric("Input tokens", f"{st.session_state.input_tokens:,}")
    c3.metric("Output tokens", f"{st.session_state.output_tokens:,}")
    c4.metric("Estimated cost", f"${estimated_cost:.4f}")


def main() -> None:
    st.set_page_config(page_title="Kimi Vehicle Swarm", page_icon="🚗", layout="wide")
    init_state()
    api_key = render_sidebar()

    st.title("🚗 Streamlit Swarm Agent Prototype")
    st.caption("Zero hardcoded vehicle data: all model and specification data is discovered at runtime through Kimi K2.6 web search.")

    col_a, col_b, col_c = st.columns(3)
    manufacturer = col_a.text_input("Manufacturer", value=DEFAULT_MANUFACTURER)
    market = col_b.text_input("Market", value=DEFAULT_MARKET)
    period = col_c.text_input("Period", value=DEFAULT_PERIOD)

    run_col, clear_col = st.columns([1, 1])
    run_clicked = run_col.button("Run", type="primary", use_container_width=True)
    clear_clicked = clear_col.button("Clear", use_container_width=True)

    if clear_clicked:
        reset_state()
        st.rerun()

    if run_clicked:
        if not api_key:
            st.error("Missing MOONSHOT_API_KEY. Enter an API key in the sidebar or set the environment variable.")
        else:
            try:
                reset_state()
                run_pipeline(api_key, manufacturer, market, period)
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")

    render_persistent_outputs()


if __name__ == "__main__":
    main()
