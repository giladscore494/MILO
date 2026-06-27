import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import app
from app import AgentConfig


AGENT_KEYS = [
    "current_official_lineup_agent",
    "historical_used_market_agent",
    "ev_hybrid_edge_cases_agent",
    "normalizer_deduper",
    "trims_years_agent",
    "engines_fuel_power_agent",
    "transmission_drivetrain_performance_agent",
    "dimensions_safety_equipment_agent",
    "source_verifier",
    "final_builder",
]


def ok_payload(agent="x"):
    return {"agent": agent, "items": [], "missing_data": [], "extra_candidate_models": []}


def fake_result(content, finish_reason="stop", agent="x", phase="p"):
    return {"content": content, "finish_reason": finish_reason, "input_tokens": 1, "output_tokens": 2, "agent": agent, "phase": phase}


def test_discovery_still_accepts_ultra_thin_schema():
    parsed = {"agent": "current_official_lineup_agent", "models": [{"model_name_en": "i10", "source_url": None}]}
    assert app.validate_discovery_schema(parsed) is None
    assert parsed["models"] == [{"model_name_en": "i10", "source_url": None}]


def test_normalizer_requires_canonical_models_list():
    assert app.validate_normalizer_schema({"agent": "normalizer_deduper", "canonical_models": {}, "rejected_items": [], "needs_review": []}) == "INVALID_NORMALIZER_SCHEMA"


def test_agents_3_to_8_use_shared_safe_wrapper():
    calls = []

    def fake_safe(*args, **kwargs):
        calls.append(kwargs["agent_name"])
        return app.phase_result(status="success", agent=kwargs["agent_name"], parsed={"ok": True})

    models = [{"canonical_model_name": "i10", "sources": []}]
    with patch("app.run_safe_agent", side_effect=fake_safe):
        for agent in app.TECHNICAL_AGENTS:
            app.run_technical_agent("key", agent, "Hyundai", "Israel", "2010-2026", models)
        app.run_verification_phase("key", {}, {})
        app.run_final_builder_phase("key", {}, {}, {}, [], "Hyundai", "Israel", "2010-2026")

    assert calls == [
        "trims_years_agent",
        "engines_fuel_power_agent",
        "transmission_drivetrain_performance_agent",
        "dimensions_safety_equipment_agent",
        "source_verifier",
        "final_builder",
    ]


def test_finish_reason_length_fails_for_every_agent():
    for agent in AGENT_KEYS:
        checked = app.validate_model_response(fake_result('{"agent":"x"}', finish_reason="length", agent=agent), require_json=True)
        assert checked["_error"] == "MODEL_OUTPUT_TRUNCATED"


def test_invalid_json_fails_for_every_agent():
    for agent in AGENT_KEYS:
        checked = app.validate_model_response(fake_result("not json", agent=agent), require_json=True)
        assert checked["_error"] == "INVALID_JSON"


def test_planning_loop_fails_for_every_agent():
    for agent in AGENT_KEYS:
        checked = app.validate_model_response(fake_result("I need to search. " * 4, agent=agent), require_json=True)
        assert checked["_error"] in {"MODEL_PLANNING_LOOP", "MODEL_REPETITION_LOOP"}


def test_429_concurrency_error_triggers_backoff_retry_and_classification(monkeypatch):
    calls = {"n": 0, "sleep": 0}

    def fake_chat(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("Error code: 429 request reached max organization concurrency: 3")

    monkeypatch.setattr(app, "moonshot_chat", fake_chat)
    monkeypatch.setattr(app.time, "sleep", lambda seconds: calls.__setitem__("sleep", calls["sleep"] + 1))
    result = app.run_safe_agent(
        "key", agent_name="engines_fuel_power_agent", phase_name="technical", prompt=[{"role": "user", "content": "x"}],
        max_tokens=3000, required_top_keys=["agent"], use_web_search=True,
    )
    assert result["status"] == "failed"
    assert result["error"] == "API_CONCURRENCY_LIMIT"
    assert result["api_retry_count"] == 2
    assert calls == {"n": 3, "sleep": 2}


def test_429_does_not_use_fallback_prompt(monkeypatch):
    phases = []

    def fake_chat(*args, **kwargs):
        phases.append(kwargs["phase_name"])
        raise RuntimeError("rate_limit_reached_error HTTP 429 max organization concurrency")

    monkeypatch.setattr(app, "moonshot_chat", fake_chat)
    monkeypatch.setattr(app.time, "sleep", lambda seconds: None)
    result = app.run_safe_agent(
        "key", agent_name="trims_years_agent", phase_name="technical", prompt=[{"role": "user", "content": "x"}],
        fallback_prompt=[{"role": "user", "content": "fallback"}], max_tokens=2500, required_top_keys=["agent"], use_web_search=True,
    )
    assert result["error"] == "API_CONCURRENCY_LIMIT"
    assert set(phases) == {"technical"}


def test_technical_phase_runs_sequentially_without_executor():
    order = []

    def fake_agent(api_key, agent, manufacturer, market, period, canonical_models):
        order.append(agent.key)
        return app.phase_result(status="success", agent=agent.key, parsed={})

    with patch("app.run_technical_agent", side_effect=fake_agent):
        app.run_technical_enrichment_phase("key", "Hyundai", "Israel", "2010-2026", [])
    assert order == [a.key for a in app.TECHNICAL_AGENTS]


def test_global_kimi_semaphore_limits_parallel_calls(monkeypatch):
    active = 0
    max_active = 0
    lock = threading.Lock()

    class FakeMessage:
        content = '{"ok": true}'
        tool_calls = None
        def model_dump(self, exclude_none=True):
            return {"role": "assistant", "content": self.content}

    class FakeResponse:
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
        choices = [SimpleNamespace(finish_reason="stop", message=FakeMessage())]

    class FakeCompletions:
        def create(self, **kwargs):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return FakeResponse()

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(app, "OpenAI", FakeClient)
    threads = [threading.Thread(target=app.moonshot_chat, args=("key", [{"role": "user", "content": "hi"}]), kwargs={"temperature": 0.6, "use_web_search": False, "max_tokens": 10}) for _ in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert max_active <= app.MAX_PARALLEL_KIMI_CALLS == 2


def test_raw_failed_output_is_capped_to_constant():
    checked = app.validate_model_response(fake_result("x" * 5000, finish_reason="length"), require_json=True)
    assert len(checked["raw_preview"]) == app.RAW_DEBUG_PREVIEW_CHARS


def test_final_builder_never_receives_raw_failed_text():
    captured = {}

    def fake_safe(*args, **kwargs):
        captured["prompt"] = kwargs["prompt"]
        return app.phase_result(status="success", agent="final_builder", parsed={"status": "partial"})

    with patch("app.run_safe_agent", side_effect=fake_safe):
        app.run_final_builder_phase("key", {}, {}, {}, [{"agent": "a", "error": "INVALID_JSON", "raw_preview": "SECRET_RAW"}], "Hyundai", "Israel", "2010-2026")
    prompt_text = json.dumps(captured["prompt"], ensure_ascii=False)
    assert "SECRET_RAW" not in prompt_text
    assert "INVALID_JSON" in prompt_text


def test_global_only_sources_are_marked_needs_review_by_verifier_validator():
    parsed = {"agent": "source_verifier", "verified_models": [{"model": "Ioniq", "status": "verified", "confidence": "high", "issues": ["hyundainews.com"], "source_region": "global", "source_strength": "global_official"}], "rejected_data_points": [], "needs_review": []}
    assert app.validate_verifier_schema(parsed) is None
    assert parsed["verified_models"][0]["status"] == "needs_review"
    assert parsed["verified_models"][0]["confidence"] == "medium"


def test_if_enrichment_agent_fails_final_status_becomes_partial():
    def fake_safe(*args, **kwargs):
        return app.phase_result(status="success", agent="final_builder", parsed={"manufacturer": "Hyundai", "market": "Israel", "period": "2010-2026", "status": "complete", "models": [], "needs_review": [], "rejected": [], "failed_agents": [], "token_usage": {}})

    with patch("app.run_safe_agent", side_effect=fake_safe):
        result = app.run_final_builder_phase("key", {}, {}, {}, [{"agent": "engines", "error": "INVALID_JSON"}], "Hyundai", "Israel", "2010-2026")
    assert result["parsed"]["status"] == "partial"
