import json
from types import SimpleNamespace
from unittest.mock import patch

from app import (
    RAW_DEBUG_PREVIEW_CHARS,
    AgentConfig,
    detect_planning_or_repetition_loop,
    merge_discovery_candidates,
    moonshot_chat,
    run_discovery_agent,
    validate_discovery_schema,
    validate_model_response,
)


def result(content, finish_reason="stop"):
    return {"content": content, "finish_reason": finish_reason, "input_tokens": 11, "output_tokens": 22, "parsed": None}


def valid_discovery_text():
    return json.dumps({
        "manufacturer": "Hyundai",
        "market": "Israel",
        "period": "2010 to June 2026",
        "models": [{
            "model_name_en": "i10",
            "model_name_he": None,
            "body_type": "hatchback",
            "years_sold": "2010-2026",
            "currently_sold": True,
            "generations": ["IA", "AC3"],
            "confidence": "high",
            "sources": ["https://example.com"],
            "notes": None,
        }],
    })


def test_repeated_i_need_to_search_rejected():
    looped, reason = detect_planning_or_repetition_loop("I need to search. " * 4)
    assert looped
    assert reason in {"MODEL_PLANNING_LOOP", "MODEL_REPETITION_LOOP"}


def test_repeated_let_me_search_again_rejected():
    checked = validate_model_response(result("Let me search again. " * 3), require_json=True)
    assert checked["_error"] in {"MODEL_PLANNING_LOOP", "MODEL_REPETITION_LOOP"}


def test_finish_reason_length_rejected():
    checked = validate_model_response(result('{"ok": true}', finish_reason="length"), require_json=True)
    assert checked["_error"] == "MODEL_OUTPUT_TRUNCATED"
    assert checked["finish_reason"] == "length"


def test_invalid_json_rejected():
    checked = validate_model_response(result("not json"), require_json=True)
    assert checked["_error"] == "INVALID_JSON"


def test_valid_discovery_json_object_passes():
    checked = validate_model_response(result(valid_discovery_text()), require_json=True, validator=validate_discovery_schema)
    assert "_error" not in checked
    assert checked["parsed"]["models"][0]["model_name_en"] == "i10"


def test_failed_raw_preview_is_capped_to_2000_chars():
    checked = validate_model_response(result("x" * 5000, finish_reason="length"), require_json=True)
    assert len(checked["raw_preview"]) == RAW_DEBUG_PREVIEW_CHARS == 2000


class FakeMessage:
    content = '{"ok": true}'
    tool_calls = None

    def model_dump(self, exclude_none=True):
        return {"role": "assistant", "content": self.content}


class FakeResponse:
    def __init__(self):
        self.usage = SimpleNamespace(prompt_tokens=1, completion_tokens=2)
        self.choices = [SimpleNamespace(finish_reason="stop", message=FakeMessage())]


class FakeCompletions:
    def __init__(self):
        self.kwargs = []

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        return FakeResponse()


class FakeClient:
    completions = FakeCompletions()

    def __init__(self, *args, **kwargs):
        self.chat = SimpleNamespace(completions=self.completions)


def test_moonshot_chat_passes_max_tokens_normal_and_retry_paths():
    FakeClient.completions = FakeCompletions()
    with patch("app.OpenAI", FakeClient):
        moonshot_chat("key", [{"role": "user", "content": "hi"}], temperature=0.6, use_web_search=False, max_tokens=123)
        assert FakeClient.completions.kwargs[-1]["max_tokens"] == 123

    FakeClient.completions = FakeCompletions()
    with patch("app.OpenAI", FakeClient):
        moonshot_chat("key", [{"role": "user", "content": "hi"}], temperature=0.6, use_web_search=True, max_tokens=456)
        assert len(FakeClient.completions.kwargs) == 2
        assert all(call["max_tokens"] == 456 for call in FakeClient.completions.kwargs)


def test_python_merge_deduplicates_model_names():
    merged = merge_discovery_candidates([
        {"status": "success", "agent": "a1", "parsed": {"manufacturer": "Hyundai", "market": "Israel", "period": "2010-2026", "models": [
            {"model_name_en": "Hyundai Tucson", "model_name_he": "טוסון", "confidence": "medium", "sources": ["https://a"], "notes": None}
        ]}},
        {"status": "success", "agent": "a2", "parsed": {"manufacturer": "Hyundai", "market": "Israel", "period": "2010-2026", "models": [
            {"model_name_en": "Tucson", "model_name_he": None, "confidence": "high", "sources": ["https://b"], "notes": "seen"}
        ]}},
    ])
    assert len(merged["candidate_models"]) == 1
    assert merged["candidate_models"][0]["confidence"] == "high"
    assert set(merged["candidate_models"][0]["sources"]) == {"https://a", "https://b"}


def test_python_merge_rejects_obvious_trim_package_names():
    merged = merge_discovery_candidates([
        {"status": "success", "agent": "a1", "parsed": {"manufacturer": "Hyundai", "market": "Israel", "period": "2010-2026", "models": [
            {"model_name_en": "N Line", "model_name_he": None, "confidence": "low", "sources": ["https://a"], "notes": None},
            {"model_name_en": "i20", "model_name_he": None, "confidence": "high", "sources": ["https://b"], "notes": None},
        ]}},
    ])
    assert [x["canonical_model_name"] for x in merged["candidate_models"]] == ["i20"]
    assert merged["rejected_candidates"][0]["reason"] == "trim_or_package_not_model"


def test_discovery_fallback_attempted_at_most_once():
    agent = AgentConfig("current_official_lineup_agent", "Current", "Current", "Find current models")
    calls = []

    def fake_call(*args, **kwargs):
        calls.append(kwargs["phase_name"])
        return {"content": "I need to search. " * 4, "finish_reason": "stop", "input_tokens": 1, "output_tokens": 2, "agent": kwargs["agent_name"], "phase": kwargs["phase_name"]}

    with patch("app.moonshot_chat", side_effect=fake_call):
        result = run_discovery_agent("key", agent, "Hyundai", "Israel", "2010 to June 2026")

    assert result["status"] == "failed"
    assert result["used_fallback"] is True
    assert calls == ["discovery", "discovery_fallback"]
