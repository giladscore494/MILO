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
MAX_DISCOVERY_TOKENS = 1800
MAX_ENRICHMENT_TOKENS = 3000
MAX_CONSOLIDATION_TOKENS = 5000
MAX_SUMMARY_TOKENS = 1200
DISCOVERY_RETRY_TOKENS = 1200
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
Phase 1A: Discovery (serial, web)
          |
          v
Phase 1B: Enrichment (5 parallel web agents)
  +-------+-------+-------+-------+-------+
  |Trim   |Engine |Trans. |Dim/Saf|Equip. |
  +-------+-------+-------+-------+-------+
          |
          v
Phase 2: Consolidation (serial, no web)
          |
          v
Phase 3: Hebrew Summary (serial, no web)
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


ENRICHMENT_AGENTS: List[AgentConfig] = [
    AgentConfig("agent_2_trims", "Agent 2", "Trim levels per model", "For each provided model, research Israeli-market trim levels: trim name, years available, and base price in ILS if found."),
    AgentConfig("agent_3_powertrain", "Agent 3", "Engine & drivetrain", "For each provided model, research Israeli-market powertrain specs: displacement, cylinders, HP, torque, fuel type, drivetrain, battery capacity and range for EVs."),
    AgentConfig("agent_4_performance", "Agent 4", "Transmission & performance", "For each provided model, research Israeli-market transmission and performance: gearbox type/speeds, 0-100 km/h, top speed, city/highway/combined fuel consumption in l/100km."),
    AgentConfig("agent_5_dimensions_safety", "Agent 5", "Dimensions & safety", "For each provided model, research Israeli-market dimensions and safety: L/W/H/wheelbase/trunk/weight, Euro NCAP, airbags, and ADAS feature list."),
    AgentConfig("agent_6_equipment", "Agent 6", "Equipment", "For each provided model, research Israeli-market equipment: infotainment, connectivity, comfort features, exterior features, and interior material."),
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


def _error_payload(error: str, finish_reason: Any, input_tokens: int, output_tokens: int, content: str) -> Dict[str, Any]:
    return {
        "_error": error,
        "finish_reason": finish_reason,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "raw_preview": (content or "")[:RAW_DEBUG_PREVIEW_CHARS],
    }


def validate_model_response(result: Dict[str, Any], *, require_json: bool, validator: Optional[Any] = None) -> Dict[str, Any]:
    """Apply common hard safety checks to a Kimi/Moonshot response."""
    content = result.get("content", "") or ""
    if result.get("finish_reason") == "length":
        return _error_payload("MODEL_OUTPUT_TRUNCATED", "length", result.get("input_tokens", 0), result.get("output_tokens", 0), content)
    looped, reason = detect_planning_or_repetition_loop(content)
    if looped:
        return _error_payload(reason, result.get("finish_reason"), result.get("input_tokens", 0), result.get("output_tokens", 0), content)
    if len(content) > MAX_REASONABLE_OUTPUT_CHARS:
        return _error_payload("MODEL_OUTPUT_TOO_LARGE", result.get("finish_reason"), result.get("input_tokens", 0), result.get("output_tokens", 0), content)
    if require_json:
        parsed, parse_error = _parse_json_strict(content)
        if parse_error or (isinstance(parsed, dict) and "_raw_text" in parsed):
            return _error_payload("INVALID_JSON", result.get("finish_reason"), result.get("input_tokens", 0), result.get("output_tokens", 0), content)
        if validator:
            validation_error = validator(parsed)
            if validation_error:
                payload = _error_payload(validation_error, result.get("finish_reason"), result.get("input_tokens", 0), result.get("output_tokens", 0), content)
                payload["parsed_preview"] = parsed if isinstance(parsed, (dict, list)) else None
                return payload
        ok = dict(result)
        ok["parsed"] = parsed
        return ok
    return dict(result)


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
) -> Dict[str, Any]:
    """Call Kimi and handle Moonshot's server-side builtin $web_search echo loop."""
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
            "temperature": max(0.6, temperature),
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
                "temperature": max(0.6, temperature),
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
    }


def discovery_prompt(manufacturer: str, market: str, period: str, retry: bool = False) -> List[Dict[str, str]]:
    retry_prefix = ""
    if retry:
        retry_prefix = (
            "Your previous response was invalid because it contained planning text.\n"
            "Return only the final JSON object.\n"
            "Do not say what you will search.\n"
            "Do not write \"I need to search\" or \"Let me search\".\n"
            "Return compact JSON only.\n\n"
        )
    system = MANDATORY_WEB_SEARCH_INSTRUCTION + retry_prefix + f"""You are a strict vehicle model discovery agent.

Task:
Find passenger vehicle models by the requested manufacturer that were sold in the requested market and period.

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
Return only this JSON object schema: {{"manufacturer": "{manufacturer}", "market": "{market}", "period": "{period}", "models": [{{"model_name_en": "string", "model_name_he": "string|null", "body_type": "string|null", "years_sold": "string|null", "currently_sold": true, "generations": ["string"], "confidence": "high|medium|low", "sources": ["url"], "notes": "string|null"}}]}}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def enrichment_prompt(agent: AgentConfig, manufacturer: str, market: str, period: str, model_list: str) -> List[Dict[str, str]]:
    system = MANDATORY_WEB_SEARCH_INSTRUCTION + f"""You are {agent.name}, an Israeli-market automotive data enrichment researcher.
{ISRAEL_ENRICHMENT_CONTEXT}
ONLY research models in the provided list — do not add models.
Output ONLY a valid JSON array. No markdown, no explanations."""
    user = f"""Manufacturer: {manufacturer}
Market: {market}
Period: {period}
Provided model list:
{{model_list}}

Responsibility: {agent.responsibility}
Return ONLY a JSON array keyed by model_name_en/model_name_he where possible.""".format(model_list=model_list)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def consolidation_prompt(discovery: Any, enrichments: Dict[str, Any]) -> List[Dict[str, str]]:
    schema = {
        "manufacturer": "string", "market": "string", "period": "string", "models": [{
            "model_name_en": "string|null", "model_name_he": "string|null", "body_type": "string|null",
            "years_sold": "string|array|null", "currently_sold": "boolean|null", "generations": "array|null",
            "trims": "array", "powertrains": "array", "transmission_performance": "array",
            "dimensions_safety": "object|null", "equipment": "object|null", "sources_notes": "array"
        }]
    }
    return [
        {"role": "system", "content": f"You are Agent 7, a strict JSON consolidation agent. No web search is available.\n{CONSOLIDATION_CONTEXT}"},
        {"role": "user", "content": "Merge these six agent outputs into one unified JSON object. Fuzzy-match model names across fragments. Use this schema shape:\n" + json.dumps(schema, ensure_ascii=False, indent=2) + "\n\nDiscovery output:\n" + json.dumps(discovery, ensure_ascii=False, indent=2) + "\n\nEnrichment outputs:\n" + json.dumps(enrichments, ensure_ascii=False, indent=2)},
    ]


def summary_prompt(consolidated: Any) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are Agent 8. Write a concise Hebrew market summary. No web search is available. Include specific numbers: total models, body-type breakdown, notable models, price ranges if found, and trends over the period."},
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
    st.sidebar.markdown("- Agent 1A: Discovery\n" + "\n".join(f"- {a.name}: {a.description}" for a in ENRICHMENT_AGENTS) + "\n- Agent 7: Consolidation\n- Agent 8: Hebrew summary")
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



def run_discovery_agent(api_key: str, manufacturer: str, market: str, period: str) -> Dict[str, Any]:
    """Run discovery with one bounded retry for planning/repetition loops."""
    result = moonshot_chat(
        api_key,
        discovery_prompt(manufacturer, market, period),
        temperature=SEARCH_TEMPERATURE,
        use_web_search=True,
        response_format={"type": "json_object"},
        max_tokens=MAX_DISCOVERY_TOKENS,
    )
    checked = validate_model_response(result, require_json=True, validator=validate_discovery_schema)
    if checked.get("_error") not in {"MODEL_PLANNING_LOOP", "MODEL_REPETITION_LOOP"}:
        return checked

    retry = moonshot_chat(
        api_key,
        discovery_prompt(manufacturer, market, period, retry=True),
        temperature=SEARCH_TEMPERATURE,
        use_web_search=True,
        response_format={"type": "json_object"},
        max_tokens=DISCOVERY_RETRY_TOKENS,
    )
    retry_checked = validate_model_response(retry, require_json=True, validator=validate_discovery_schema)
    retry_checked["retry_after_error"] = checked.get("_error")
    retry_checked["input_tokens"] = retry_checked.get("input_tokens", 0) + checked.get("input_tokens", 0)
    retry_checked["output_tokens"] = retry_checked.get("output_tokens", 0) + checked.get("output_tokens", 0)
    return retry_checked

def run_pipeline(api_key: str, manufacturer: str, market: str, period: str) -> None:
    start = time.perf_counter()

    discovery_box = st.empty()
    discovery_box.info("Phase 1A discovery running...")
    discovery = run_discovery_agent(api_key, manufacturer, market, period)
    add_tokens(discovery)
    if discovery.get("_error"):
        discovery_box.error(f"Discovery failed: {discovery['_error']}")
        st.caption(f"Input tokens: {discovery.get('input_tokens', 0):,} | Output tokens: {discovery.get('output_tokens', 0):,}")
        if discovery.get("raw_preview"):
            st.text_area("Discovery raw preview", discovery["raw_preview"], height=240)
        st.session_state.results["agent_1_discovery"] = discovery
        return
    discovery_data = discovery["parsed"]
    model_candidates = discovery_data["models"]

    st.session_state.discovery_data = discovery_data
    st.session_state.results["agent_1_discovery"] = discovery_data
    discovery_box.success(f"Phase 1A complete: discovered {len(model_candidates)} models.")
    with st.expander("Discovered model list", expanded=True):
        st.json(discovery_data)

    st.subheader("Phase 1B — parallel enrichment")
    progress = st.progress(0)
    status_cols = st.columns(3)
    boxes = {agent.key: status_cols[i % 3].empty() for i, agent in enumerate(ENRICHMENT_AGENTS)}
    for agent in ENRICHMENT_AGENTS:
        boxes[agent.key].info(f"{agent.name}: running")

    model_list_text = build_model_list_text(discovery_data)
    enrichments: Dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {
            executor.submit(moonshot_chat, api_key, enrichment_prompt(agent, manufacturer, market, period, model_list_text), temperature=SEARCH_TEMPERATURE, use_web_search=True, max_tokens=MAX_ENRICHMENT_TOKENS): agent
            for agent in ENRICHMENT_AGENTS
        }
        completed = 0
        for future in as_completed(future_map):
            agent = future_map[future]
            try:
                result = validate_model_response(future.result(), require_json=True)
                add_tokens(result)
                if result.get("_error"):
                    enrichments[agent.key] = result
                    boxes[agent.key].error(f"{agent.name}: failed — {result['_error']}")
                else:
                    enrichments[agent.key] = result["parsed"]
                    boxes[agent.key].success(f"{agent.name}: complete")
            except Exception as exc:
                enrichments[agent.key] = {"_error": str(exc)}
                boxes[agent.key].error(f"{agent.name}: failed")
            completed += 1
            progress.progress(completed / len(ENRICHMENT_AGENTS))

    st.session_state.results.update(enrichments)
    failed_enrichments = {key: value for key, value in enrichments.items() if isinstance(value, dict) and value.get("_error")}
    if failed_enrichments:
        st.error("Enrichment failed; pipeline aborted before consolidation.")
        st.json(failed_enrichments)
        return

    st.subheader("Phase 2 — consolidation")
    with st.spinner("Consolidating without web search..."):
        consolidated_result = validate_model_response(
            moonshot_chat(api_key, consolidation_prompt(discovery_data, enrichments), temperature=CONSOLIDATION_TEMPERATURE, use_web_search=False, response_format={"type": "json_object"}, max_tokens=MAX_CONSOLIDATION_TOKENS),
            require_json=True,
        )
        add_tokens(consolidated_result)
    if consolidated_result.get("_error"):
        st.error(f"Consolidation failed: {consolidated_result['_error']}")
        st.session_state.results["agent_7_consolidation"] = consolidated_result
        return
    st.session_state.consolidated = consolidated_result["parsed"]
    model_count, trim_count = count_models_trims(st.session_state.consolidated)
    st.success(f"Consolidation complete: {model_count} models, {trim_count} trims.")

    st.subheader("Phase 3 — Hebrew summary")
    with st.spinner("Generating Hebrew summary without web search..."):
        summary_result = validate_model_response(moonshot_chat(api_key, summary_prompt(st.session_state.consolidated), temperature=CONSOLIDATION_TEMPERATURE, use_web_search=False, max_tokens=MAX_SUMMARY_TOKENS), require_json=False)
        add_tokens(summary_result)
    if summary_result.get("_error"):
        st.error(f"Summary failed: {summary_result['_error']}")
        st.session_state.results["agent_8_summary"] = summary_result
        return
    st.session_state.summary = summary_result["content"]
    st.markdown(st.session_state.summary)

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
