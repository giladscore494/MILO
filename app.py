# -*- coding: utf-8 -*-
"""Streamlit swarm prototype for Israeli vehicle-model mapping with Kimi K2.6."""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from openai import OpenAI

MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"
KIMI_MODEL = "kimi-k2.6"
SEARCH_TEMPERATURE = 0.6
CONSOLIDATION_TEMPERATURE = 0.6
MAX_TOOL_ROUNDS = 15
MAX_DISCOVERY_TOKENS = 1600
MAX_DISCOVERY_FALLBACK_TOKENS = 1200
MAX_TECHNICAL_AGENT_TOKENS = 2500
MAX_VERIFIER_TOKENS = 3500
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
    AgentConfig("current_official_lineup_agent", "Current official lineup", "Current official/importer models", "Find currently sold models from official importer / official Israel sources."),
    AgentConfig("historical_used_market_agent", "Historical used market", "Historical used-car/model-list models", "Find historical models sold in Israel using Israeli used-car/model-listing/price-list sources."),
    AgentConfig("ev_hybrid_edge_cases_agent", "EV/hybrid edge cases", "EV, hybrid, and special models", "Find EV, hybrid, and special models that generic searches may miss."),
]

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


def _error_payload(error: str, finish_reason: Any, input_tokens: int, output_tokens: int, content: str, *, agent: str = "", phase: str = "") -> Dict[str, Any]:
    return {
        "_error": error,
        "agent": agent,
        "phase": phase,
        "finish_reason": finish_reason,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "raw_preview": (content or "")[:RAW_DEBUG_PREVIEW_CHARS],
    }


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
        return _error_payload("MODEL_OUTPUT_TRUNCATED", "length", result.get("input_tokens", 0), result.get("output_tokens", 0), content, agent=agent, phase=phase)
    looped, reason = detect_planning_or_repetition_loop(content)
    if looped:
        return _error_payload(reason, result.get("finish_reason"), result.get("input_tokens", 0), result.get("output_tokens", 0), content, agent=agent, phase=phase)
    if len(content) > MAX_REASONABLE_OUTPUT_CHARS:
        return _error_payload("MODEL_OUTPUT_TOO_LARGE", result.get("finish_reason"), result.get("input_tokens", 0), result.get("output_tokens", 0), content, agent=agent, phase=phase)
    if require_json:
        parsed, parse_error = _parse_json_strict(content)
        if parse_error or (isinstance(parsed, dict) and "_raw_text" in parsed):
            return _error_payload("INVALID_JSON", result.get("finish_reason"), result.get("input_tokens", 0), result.get("output_tokens", 0), content, agent=agent, phase=phase)
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
                "confidence": "low",
                "sources": [],
                "notes": [],
            })
            for alias in (item.get("model_name_he"), *(item.get("aliases") or [])):
                if alias and alias not in entry["aliases"] and alias != entry["canonical_model_name"]:
                    entry["aliases"].append(alias)
            if agent and agent not in entry["found_by_agents"]:
                entry["found_by_agents"].append(agent)
            if _confidence_rank(item.get("confidence")) > _confidence_rank(entry["confidence"]):
                entry["confidence"] = item.get("confidence")
            for src in item.get("sources") or []:
                if src and src not in entry["sources"]:
                    entry["sources"].append(src)
            if item.get("notes") and item.get("notes") not in entry["notes"]:
                entry["notes"].append(item["notes"])

    return {
        "manufacturer": manufacturer,
        "market": market,
        "period": period,
        "candidate_models": sorted(merged.values(), key=lambda x: x["canonical_model_name"].lower()),
        "rejected_candidates": rejected,
        "failed_agents": failed_agents,
    }


def validate_discovery_schema(parsed: Any) -> Optional[str]:
    if not isinstance(parsed, dict):
        return "INVALID_DISCOVERY_SCHEMA"
    models = parsed.get("models")
    if not isinstance(models, list) or not models:
        return "INVALID_DISCOVERY_SCHEMA"
    for item in models:
        if not isinstance(item, dict):
            return "INVALID_DISCOVERY_SCHEMA"
        if not item.get("model_name_en") or not item.get("confidence") or not isinstance(item.get("sources"), list):
            return "INVALID_DISCOVERY_SCHEMA"
    return None


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
            name_he = model.get("model_name_he")
            body = model.get("body_type")
            years = model.get("years_sold")
            current = model.get("currently_sold")
            generations = model.get("generations")
            lines.append(
                f"{idx}. model_name_en={name_en}; model_name_he={name_he}; body_type={body}; "
                f"years_sold={years}; currently_sold={current}; generations={generations}"
            )
        else:
            lines.append(f"{idx}. {model}")
    return "\n".join(lines)


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
    retry_prefix = ""
    if retry:
        retry_prefix = (
            "Your previous response was invalid.\nReturn only compact JSON.\nDo not describe searching.\n"
            "Do not write planning text.\nReturn model names only.\nMaximum 25 models.\n"
            "If unsure, mark confidence=\"low\".\nJSON object only.\n\n"
        )
    system = MANDATORY_WEB_SEARCH_INSTRUCTION + retry_prefix + f"""You are {agent.key}, a strict focused vehicle model discovery agent.

Task:
{agent.responsibility}

You must use web search.
Return only the final JSON object.
Do not describe your research process.
Do not write planning text.
Never write:
"I need to search"
"Let me search"
"Let me search again"
"I will search"
"I'll search"
"I should search"

Output must be exactly this JSON object:
{{
  "agent": "{agent.key}",
  "manufacturer": "...",
  "market": "...",
  "period": "...",
  "models": [...]
}}

Rules:
- Return JSON only.
- No markdown.
- No explanation outside JSON.
- Maximum 40 models.
- Do not include trims, engines, or variants.
- Only include model-level names.
- If unsure, include the model with confidence="low".
- Prefer Israeli official/importer sources and Israeli automotive sources.
- Each model should include at least one source URL when possible.
- If source coverage is incomplete, say so only inside notes.
- Stop immediately after the JSON object.
{ISRAEL_DISCOVERY_CONTEXT}"""
    user = f"""Manufacturer: {manufacturer}
Market: {market}
Period: {period}
Return only this JSON object schema: {{"agent": "{agent.key}", "manufacturer": "{manufacturer}", "market": "{market}", "period": "{period}", "models": [{{"model_name_en": "string", "model_name_he": "string|null", "body_type": "string|null", "years_sold": "string|null", "currently_sold": true, "confidence": "high|medium|low", "sources": ["url"], "notes": "string|null"}}]}}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalizer_prompt(merged: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are normalizer_deduper. JSON only. No web search. Clean candidate model names only: normalize names, merge aliases, reject trims, separate distinct models, and flag uncertainty. Do not add technical data."},
        {"role": "user", "content": "Clean this merged candidate JSON and return schema {\"agent\":\"normalizer_deduper\",\"canonical_models\":[{\"canonical_model_name\":\"string\",\"model_name_he\":\"string|null\",\"aliases\":[\"string\"],\"body_type\":\"sedan|null\",\"currently_sold\":true,\"confidence\":\"high|medium|low\",\"sources\":[\"url\"],\"notes\":\"string|null\"}],\"rejected_items\":[{\"name\":\"string\",\"reason\":\"trim_or_package_not_model\"}],\"needs_review\":[]}.\n" + json.dumps(merged, ensure_ascii=False, indent=2)},
    ]


def technical_prompt(agent: AgentConfig, manufacturer: str, market: str, period: str, canonical_models: List[Dict[str, Any]], retry: bool = False) -> List[Dict[str, str]]:
    strict = "Your previous response was invalid. Return compact JSON only. Do not describe searching. Use missing_data if unsure.\n" if retry else ""
    system = MANDATORY_WEB_SEARCH_INSTRUCTION + f"""You are {agent.name}, an Israeli-market automotive data enrichment researcher.
{ISRAEL_ENRICHMENT_CONTEXT}
{strict}ONLY research models in the provided canonical model list.
Do not add new models directly; put model-level surprises in extra_candidate_models.
Output ONLY a valid JSON object. No markdown, no explanations."""
    user = f"""Manufacturer: {manufacturer}
Market: {market}
Period: {period}
Canonical models:
{json.dumps(canonical_models, ensure_ascii=False, indent=2)}

Responsibility: {agent.responsibility}
Return schema: {{"agent":"{agent.key}","manufacturer":"{manufacturer}","market":"{market}","items":[{{"model":"string","years":"string|null","variant_or_generation":"string|null","data":{{}},"confidence":"high|medium|low","sources":["url"],"notes":"string|null"}}],"missing_data":[{{"model":"string","field":"string","reason":"not_found|conflicting_sources|not_applicable"}}],"extra_candidate_models":[]}}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def verifier_prompt(normalized: Any, technical: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are source_verifier. JSON only. No large new research. Review existing structured JSON only; do not invent missing data."},
        {"role": "user", "content": "Verify Israel-market relevance, model-vs-trim status, contradictions, and unsupported data. Return schema {\"agent\":\"source_verifier\",\"verified_models\":[{\"model\":\"string\",\"status\":\"verified|partial|needs_review\",\"issues\":[],\"confidence\":\"high|medium|low\"}],\"rejected_data_points\":[{\"model\":\"string\",\"field\":\"string\",\"value\":\"string\",\"reason\":\"source_not_israel|conflict|trim_not_model|unsupported\"}],\"needs_review\":[{\"model\":\"string\",\"reason\":\"string\"}]}.\nNormalized:\n" + json.dumps(normalized, ensure_ascii=False, indent=2) + "\nTechnical:\n" + json.dumps(technical, ensure_ascii=False, indent=2)},
    ]


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
        st.json(st.session_state.results)
    with tab_consolidated:
        st.json(st.session_state.consolidated)
        if st.session_state.consolidated is not None:
            st.download_button("Download consolidated JSON", json.dumps(st.session_state.consolidated, ensure_ascii=False, indent=2), "consolidated_vehicle_models.json", "application/json")
    with tab_summary:
        st.markdown(st.session_state.summary or "_No summary yet._")



RETRYABLE_ERRORS = {"MODEL_PLANNING_LOOP", "MODEL_REPETITION_LOOP", "INVALID_JSON", "MODEL_OUTPUT_TRUNCATED"}


def _checked_agent_call(
    api_key: str,
    messages: List[Dict[str, Any]],
    *,
    agent_name: str,
    phase_name: str,
    use_web_search: bool,
    max_tokens: int,
    validator: Optional[Any] = None,
    required_keys: Optional[List[str]] = None,
    non_empty_lists: Optional[List[str]] = None,
) -> Dict[str, Any]:
    result = moonshot_chat(
        api_key,
        messages,
        temperature=SEARCH_TEMPERATURE if use_web_search else CONSOLIDATION_TEMPERATURE,
        use_web_search=use_web_search,
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
        agent_name=agent_name,
        phase_name=phase_name,
    )
    return validate_model_response(result, require_json=True, validator=validator, required_keys=required_keys, non_empty_lists=non_empty_lists)


def _to_phase_result(agent: str, checked: Dict[str, Any], used_fallback: bool = False) -> Dict[str, Any]:
    if checked.get("_error"):
        return phase_result(status="failed", agent=agent, error=checked["_error"], raw_preview=checked.get("raw_preview", ""), finish_reason=checked.get("finish_reason"), input_tokens=checked.get("input_tokens", 0), output_tokens=checked.get("output_tokens", 0), used_fallback=used_fallback)
    return phase_result(status="success", agent=agent, parsed=checked.get("parsed"), finish_reason=checked.get("finish_reason"), input_tokens=checked.get("input_tokens", 0), output_tokens=checked.get("output_tokens", 0), used_fallback=used_fallback)


def run_discovery_agent(api_key: str, agent: AgentConfig, manufacturer: str, market: str, period: str) -> Dict[str, Any]:
    checked = _checked_agent_call(
        api_key, discovery_prompt(agent, manufacturer, market, period), agent_name=agent.key, phase_name="discovery",
        use_web_search=True, max_tokens=MAX_DISCOVERY_TOKENS, validator=validate_discovery_schema, required_keys=["agent", "manufacturer", "market", "period", "models"], non_empty_lists=["models"]
    )
    if checked.get("_error") not in RETRYABLE_ERRORS:
        return _to_phase_result(agent.key, checked)
    retry_checked = _checked_agent_call(
        api_key, discovery_prompt(agent, manufacturer, market, period, retry=True), agent_name=agent.key, phase_name="discovery_fallback",
        use_web_search=True, max_tokens=MAX_DISCOVERY_FALLBACK_TOKENS, validator=validate_discovery_schema, required_keys=["agent", "manufacturer", "market", "period", "models"], non_empty_lists=["models"]
    )
    retry_checked["input_tokens"] = retry_checked.get("input_tokens", 0) + checked.get("input_tokens", 0)
    retry_checked["output_tokens"] = retry_checked.get("output_tokens", 0) + checked.get("output_tokens", 0)
    return _to_phase_result(agent.key, retry_checked, used_fallback=True)


def run_discovery_phase(api_key: str, manufacturer: str, market: str, period: str) -> List[Dict[str, Any]]:
    return [run_discovery_agent(api_key, agent, manufacturer, market, period) for agent in DISCOVERY_AGENTS]


def run_normalizer_phase(api_key: str, merged: Dict[str, Any]) -> Dict[str, Any]:
    checked = _checked_agent_call(api_key, normalizer_prompt(merged), agent_name="normalizer_deduper", phase_name="normalizer", use_web_search=False, max_tokens=MAX_TECHNICAL_AGENT_TOKENS, required_keys=["agent", "canonical_models"], non_empty_lists=["canonical_models"])
    return _to_phase_result("normalizer_deduper", checked)


def run_technical_agent(api_key: str, agent: AgentConfig, manufacturer: str, market: str, period: str, canonical_models: List[Dict[str, Any]]) -> Dict[str, Any]:
    checked = _checked_agent_call(api_key, technical_prompt(agent, manufacturer, market, period, canonical_models), agent_name=agent.key, phase_name="technical", use_web_search=True, max_tokens=MAX_TECHNICAL_AGENT_TOKENS, required_keys=["agent", "manufacturer", "market", "items", "missing_data"])
    used_fallback = False
    if checked.get("_error") in RETRYABLE_ERRORS:
        checked = _checked_agent_call(api_key, technical_prompt(agent, manufacturer, market, period, canonical_models, retry=True), agent_name=agent.key, phase_name="technical_fallback", use_web_search=True, max_tokens=MAX_TECHNICAL_AGENT_TOKENS, required_keys=["agent", "manufacturer", "market", "items", "missing_data"])
        used_fallback = True
    return _to_phase_result(agent.key, checked, used_fallback)


def run_technical_enrichment_phase(api_key: str, manufacturer: str, market: str, period: str, canonical_models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_technical_agent, api_key, a, manufacturer, market, period, canonical_models) for a in TECHNICAL_AGENTS]
        return [f.result() for f in as_completed(futures)]


def run_verification_phase(api_key: str, normalized: Any, technical: Dict[str, Any]) -> Dict[str, Any]:
    checked = _checked_agent_call(api_key, verifier_prompt(normalized, technical), agent_name="source_verifier", phase_name="verification", use_web_search=False, max_tokens=MAX_VERIFIER_TOKENS, required_keys=["agent", "verified_models", "rejected_data_points", "needs_review"])
    return _to_phase_result("source_verifier", checked)


def run_final_builder_phase(api_key: str, normalized: Any, technical: Dict[str, Any], verifier: Any, failed_summaries: List[Dict[str, Any]], manufacturer: str, market: str, period: str) -> Dict[str, Any]:
    checked = _checked_agent_call(api_key, final_builder_prompt(normalized, technical, verifier, failed_summaries, manufacturer, market, period), agent_name="final_builder", phase_name="final_builder", use_web_search=False, max_tokens=MAX_FINAL_BUILDER_TOKENS, required_keys=["manufacturer", "market", "period", "status", "models", "failed_agents", "token_usage"])
    return _to_phase_result("final_builder", checked)


def run_hebrew_summary_phase(api_key: str, final_json: Any) -> Dict[str, Any]:
    result = moonshot_chat(api_key, summary_prompt(final_json), temperature=CONSOLIDATION_TEMPERATURE, use_web_search=False, max_tokens=MAX_SUMMARY_TOKENS, agent_name="hebrew_summary", phase_name="summary")
    checked = validate_model_response(result, require_json=False)
    return _to_phase_result("hebrew_summary", checked) if checked.get("_error") else phase_result(status="success", agent="hebrew_summary", parsed={"summary": checked.get("content", "")}, finish_reason=checked.get("finish_reason"), input_tokens=checked.get("input_tokens", 0), output_tokens=checked.get("output_tokens", 0))

def run_pipeline(api_key: str, manufacturer: str, market: str, period: str) -> None:
    start = time.perf_counter()
    st.subheader("Phase 1 — focused discovery")
    discovery_results = run_discovery_phase(api_key, manufacturer, market, period)
    for r in discovery_results:
        add_tokens(r)
        st.write(f"{r['agent']}: {r['status']}" + (f" — {r['error']}" if r.get("error") else ""))
        if r.get("raw_preview"):
            st.text_area(f"{r['agent']} raw preview", r["raw_preview"], height=120)
    st.session_state.results["discovery_phase"] = discovery_results
    successful_discovery = [r for r in discovery_results if r["status"] == "success"]
    if not successful_discovery:
        reason = discovery_results[0].get("error") if discovery_results else "NO_DISCOVERY_RESULTS"
        st.error(f"Discovery failed: {reason}")
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
        if r["status"] == "success":
            technical_clean[r["agent"]] = r["parsed"]
        else:
            failed_summaries.append({"agent": r["agent"], "error": r["error"], "raw_preview": r.get("raw_preview", "")})
    st.session_state.results["technical_enrichment_phase"] = technical_results

    st.subheader("Phase 4 — verifier")
    verifier = run_verification_phase(api_key, normalizer["parsed"], technical_clean)
    add_tokens(verifier)
    st.session_state.results["source_verifier"] = verifier
    if verifier["status"] != "success":
        failed_summaries.append({"agent": "source_verifier", "error": verifier["error"], "raw_preview": verifier.get("raw_preview", "")})
        verifier_data = {"agent": "source_verifier", "verified_models": [], "rejected_data_points": [], "needs_review": [{"model": "*", "reason": verifier["error"]}]}
        st.warning(f"Verifier failed: {verifier['error']}; continuing partial.")
    else:
        verifier_data = verifier["parsed"]

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
