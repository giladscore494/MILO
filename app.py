
# -*- coding: utf-8 -*-
import json
import math
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# =========================================================
# Persistent workspace
# =========================================================
APP_DIR = Path(".reliability_benchmark_workspace")
APP_DIR.mkdir(exist_ok=True)

BENCHMARK_PATH = APP_DIR / "benchmark_vehicles.json"
PROGRESS_PATH = APP_DIR / "benchmark_progress.json"
RESULTS_PATH = APP_DIR / "benchmark_results.jsonl"
SUMMARY_PATH = APP_DIR / "benchmark_summary.csv"
LOG_PATH = APP_DIR / "run_log.jsonl"

DEFAULT_ANALYZER_MODEL = "gemini-3-flash-preview"
DEFAULT_JUDGE_MODEL = "gemini-3-flash-preview"
DEFAULT_BATCH_SIZE = 5

# =========================================================
# Built-in benchmark set (no manual filling needed)
# =========================================================
DEFAULT_BENCHMARK_VEHICLES: List[Dict[str, Any]] = [{'make': 'Toyota', 'model': 'Corolla', 'year': 2022, 'fuel_type': 'בנזין', 'transmission': 'אוטומטית', 'mileage_range': '0-50k', 'segment': 'strong'}, {'make': 'Toyota', 'model': 'RAV4', 'year': 2025, 'fuel_type': 'היברידי', 'transmission': 'אוטומטית', 'mileage_range': '0-50k', 'segment': 'strong'}, {'make': 'Toyota', 'model': 'Camry', 'year': 2020, 'fuel_type': 'היברידי', 'transmission': 'אוטומטית', 'mileage_range': '50-100k', 'segment': 'strong'}, {'make': 'Toyota', 'model': 'Prius', 'year': 2018, 'fuel_type': 'היברידי', 'transmission': 'אוטומטית', 'mileage_range': '100-150k', 'segment': 'strong'}, {'make': 'Mazda', 'model': 'CX-5', 'year': 2021, 'fuel_type': 'בנזין', 'transmission': 'אוטומטית', 'mileage_range': '0-50k', 'segment': 'strong'}, {'make': 'Mazda', 'model': '3', 'year': 2022, 'fuel_type': 'בנזין', 'transmission': 'אוטומטית', 'mileage_range': '0-50k', 'segment': 'strong'}, {'make': 'Subaru', 'model': 'Forester', 'year': 2021, 'fuel_type': 'בנזין', 'transmission': 'אוטומטית', 'mileage_range': '0-50k', 'segment': 'strong'}, {'make': 'Honda', 'model': 'Civic', 'year': 2021, 'fuel_type': 'בנזין', 'transmission': 'אוטומטית', 'mileage_range': '0-50k', 'segment': 'strong'}, {'make': 'Lexus', 'model': 'CT200h', 'year': 2018, 'fuel_type': 'היברידי', 'transmission': 'אוטומטית', 'mileage_range': '50-100k', 'segment': 'strong'}, {'make': 'Toyota', 'model': 'Yaris', 'year': 2021, 'fuel_type': 'היברידי', 'transmission': 'אוטומטית', 'mileage_range': '0-50k', 'segment': 'strong'}, {'make': 'Hyundai', 'model': 'Elantra', 'year': 2021, 'fuel_type': 'בנזין', 'transmission': 'אוטומטית', 'mileage_range': '0-50k', 'segment': 'good_mid'}, {'make': 'Kia', 'model': 'Sportage', 'year': 2021, 'fuel_type': 'בנזין', 'transmission': 'אוטומטית', 'mileage_range': '0-50k', 'segment': 'good_mid'}, {'make': 'Suzuki', 'model': 'Vitara', 'year': 2019, 'fuel_type': 'בנזין', 'transmission': 'אוטומטית', 'mileage_range': '50-100k', 'segment': 'good_mid'}, {'make': 'Honda', 'model': 'CR-V', 'year': 2019, 'fuel_type': 'בנזין', 'transmission': 'אוטומטית', 'mileage_range': '50-100k', 'segment': 'good_mid'}, {'make': 'Kia', 'model': 'Ceed', 'year': 2019, 'fuel_type': 'בנזין', 'transmission': 'אוטומטית', 'mileage_range': '50-100k', 'segment': 'good_mid'}, {'make': 'Toyota', 'model': 'Avensis', 'year': 2016, 'fuel_type': 'בנזין', 'transmission': 'אוטומטית', 'mileage_range': '100-150k', 'segment': 'good_mid'}, {'make': 'Skoda', 'model': 'Octavia', 'year': 2018, 'fuel_type': 'בנזין', 'transmission': 'אוטומטית', 'mileage_range': '50-100k', 'segment': 'borderline'}, {'make': 'Volkswagen', 'model': 'Golf', 'year': 2020, 'fuel_type': 'בנזין', 'transmission': 'אוטומטית', 'mileage_range': '0-50k', 'segment': 'borderline'}, {'make': 'BMW', 'model': '320i', 'year': 2019, 'fuel_type': 'בנזין', 'transmission': 'אוטומטית', 'mileage_range': '50-100k', 'segment': 'borderline'}, {'make': 'Mercedes-Benz', 'model': 'A-Class', 'year': 2019, 'fuel_type': 'בנזין', 'transmission': 'אוטומטית', 'mileage_range': '50-100k', 'segment': 'borderline'}, {'make': 'Audi', 'model': 'Q3', 'year': 2019, 'fuel_type': 'בנזין', 'transmission': 'אוטומטית', 'mileage_range': '50-100k', 'segment': 'borderline'}, {'make': 'Land Rover', 'model': 'Discovery Sport', 'year': 2018, 'fuel_type': 'בנזין', 'transmission': 'אוטומטית', 'mileage_range': '100-150k', 'segment': 'weak'}, {'make': 'Jaguar', 'model': 'F-Pace', 'year': 2018, 'fuel_type': 'דיזל', 'transmission': 'אוטומטית', 'mileage_range': '100-150k', 'segment': 'weak'}, {'make': 'Fiat', 'model': '500X', 'year': 2018, 'fuel_type': 'בנזין', 'transmission': 'אוטומטית', 'mileage_range': '100-150k', 'segment': 'weak'}, {'make': 'Jeep', 'model': 'Renegade', 'year': 2018, 'fuel_type': 'בנזין', 'transmission': 'אוטומטית', 'mileage_range': '100-150k', 'segment': 'weak'}]

# קובץ הכיול החדש: 25 רכבים ממוקדים. ריצת benchmark ברירת מחדל: כפול 2 לכל רכב.


# =========================================================
# Typing aliases used by models
# =========================================================
Severity3 = Literal["low", "medium", "high"]
Freq3 = Literal["rare", "occasional", "common"]
Label3 = Literal["low", "medium", "high"]


class RecallItem(BaseModel):
    system: str
    description: str
    severity: Severity3
    source: Optional[str] = None


class SystemicIssue(BaseModel):
    system: str
    issue: str
    severity: Severity3
    repeat_frequency: Freq3
    typical_timing: Optional[str] = None
    evidence_text: Optional[str] = None


class MaintenancePressure(BaseModel):
    level: Label3
    explanation: str = ""


class VehicleResolution(BaseModel):
    generation: Optional[str] = None
    engine_family: Optional[str] = None
    transmission_type: Literal["automatic", "manual", "cvt", "dct", "other", "unknown"] = "unknown"


class RiskSignals(BaseModel):
    vehicle_resolution: VehicleResolution = Field(default_factory=VehicleResolution)
    recalls: Dict[str, Any] = Field(default_factory=lambda: {"count": 0, "items": [], "notes": ""})
    systemic_issue_signals: List[SystemicIssue] = Field(default_factory=list)
    maintenance_cost_pressure: MaintenancePressure = Field(default_factory=lambda: MaintenancePressure(level="medium", explanation=""))
    analysis_confidence: Label3 = "medium"
    missing_data_flags: List[str] = Field(default_factory=list)


class ReliabilityReport(BaseModel):
    overall_score: int = 0
    confidence: Label3 = "medium"
    one_sentence_verdict: str = ""
    top_risks: List[Dict[str, Any]] = Field(default_factory=list)
    expected_ownership_cost: Dict[str, Any] = Field(default_factory=dict)
    buyer_checklist: Dict[str, Any] = Field(default_factory=dict)
    what_changes_with_mileage: List[Dict[str, Any]] = Field(default_factory=list)
    recommended_next_step: Dict[str, Any] = Field(default_factory=dict)
    missing_info: List[str] = Field(default_factory=list)


class AnalyzerResponse(BaseModel):
    ok: bool = True
    search_performed: bool = True
    search_queries: List[str] = Field(default_factory=list)
    sources: List[Any] = Field(default_factory=list)
    overall_reliability_estimate: Label3
    reliability_bias: Optional[Literal["strong", "neutral", "weak"]] = None
    recall_penalty_sensitivity: Optional[Literal["low", "normal", "high"]] = None
    maintenance_penalty_sensitivity: Optional[Literal["low", "normal", "high"]] = None
    systemic_penalty_sensitivity: Optional[Literal["low", "normal", "high"]] = None
    soft_floor_if_no_major_systemic: Optional[int] = None
    calibration_confidence: Optional[Label3] = None
    overall_reliability_reasoning: str = ""
    reliability_factors_summary: str = ""
    score_breakdown: Dict[str, Any] = Field(default_factory=dict)
    base_score_calculated: int = 0
    estimated_reliability: str = "לא ידוע"
    common_issues: List[str] = Field(default_factory=list)
    avg_repair_cost_ILS: Optional[Any] = None
    issues_with_costs: List[Dict[str, Any]] = Field(default_factory=list)
    reliability_summary: str = ""
    reliability_summary_simple: str = ""
    recommended_checks: List[str] = Field(default_factory=list)
    common_competitors_brief: List[Dict[str, Any]] = Field(default_factory=list)
    reliability_report: ReliabilityReport = Field(default_factory=ReliabilityReport)
    risk_signals: RiskSignals


class JudgeResponse(BaseModel):
    search_performed: bool = True
    search_queries: List[str] = Field(default_factory=list)
    sources: List[Any] = Field(default_factory=list)
    expected_reliability_label: Literal["גבוה", "בינוני", "נמוך"]
    expected_score_min: int
    expected_score_max: int
    expected_deal_risk_label: Literal["נמוך", "בינוני", "גבוה"]
    chronic_reliability_weight: Literal["high", "medium", "low"]
    recall_noise_weight: Literal["high", "medium", "low"]
    maintenance_cost_vs_reliability: str
    judge_summary_he: str
    top_truth_signals: List[str] = Field(default_factory=list)


# =========================================================
# Helpers
# =========================================================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(s: Any) -> str:
    import re
    if s is None:
        return ""
    s = re.sub(r"\(.*?\)", " ", str(s)).strip().lower()
    return re.sub(r"\s+", " ", s)


def mileage_adjustment(mileage_range: str) -> Tuple[int, Optional[str]]:
    m = normalize_text(mileage_range or "")
    if not m:
        return 0, None
    if "200" in m and "+" in m:
        return -15, "הציון הותאם מטה עקב קילומטראז׳ גבוה מאוד (200K+)."
    if "150" in m and "200" in m:
        return -10, "הציון הותאם מטה עקב קילומטראז׳ גבוה (150–200 אלף ק״מ)."
    if "100" in m and "150" in m:
        return -5, "הציון הותאם מעט מטה עקב קילומטראז׳ בינוני-גבוה (100–150 אלף ק״מ)."
    return 0, None


def append_jsonl(path: Path, event: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path, default: Any) -> Any:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def ensure_benchmark_file() -> List[Dict[str, Any]]:
    """Always keep Milo aligned to the current 25-vehicle calibration pack."""
    try:
        existing = load_json(BENCHMARK_PATH, DEFAULT_BENCHMARK_VEHICLES) if BENCHMARK_PATH.exists() else None
    except Exception:
        existing = None
    if existing != DEFAULT_BENCHMARK_VEHICLES:
        save_json(BENCHMARK_PATH, DEFAULT_BENCHMARK_VEHICLES)
    return DEFAULT_BENCHMARK_VEHICLES


def reset_workspace() -> None:
    for path in [PROGRESS_PATH, RESULTS_PATH, SUMMARY_PATH, LOG_PATH]:
        if path.exists():
            path.unlink()


def load_progress() -> Dict[str, Any]:
    return load_json(PROGRESS_PATH, {
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "completed_runs": [],
        "last_batch": None,
        "batches": [],
    })


def save_progress(progress: Dict[str, Any]) -> None:
    progress["updated_at"] = utc_now_iso()
    save_json(PROGRESS_PATH, progress)


def run_key(vehicle: Dict[str, Any], run_idx: int) -> str:
    return "|||".join([
        str(vehicle["make"]),
        str(vehicle["model"]),
        str(vehicle["year"]),
        str(vehicle["fuel_type"]),
        str(vehicle["transmission"]),
        str(vehicle["mileage_range"]),
        str(run_idx),
    ])


def extract_grounding_debug(response: Any) -> Dict[str, Any]:
    try:
        candidate = response.candidates[0]
        meta = getattr(candidate, "grounding_metadata", None)
        if not meta:
            return {"web_search_queries": [], "sources": []}
        queries = list(getattr(meta, "web_search_queries", []) or [])
        chunks = list(getattr(meta, "grounding_chunks", []) or [])
        sources = []
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            if web:
                sources.append({"title": getattr(web, "title", None), "uri": getattr(web, "uri", None)})
        return {"web_search_queries": queries, "sources": sources}
    except Exception:
        return {"web_search_queries": [], "sources": []}


def build_combined_prompt(payload: Dict[str, Any], missing_info: List[str]) -> str:
    safe_make = str(payload.get("make", "")).strip()
    safe_model = str(payload.get("model", "")).strip()
    safe_sub_model = str(payload.get("sub_model", "")).strip()
    safe_year = str(payload.get("year", "")).strip()
    safe_mileage = str(payload.get("mileage_range") or payload.get("mileage_km") or "").strip()
    safe_fuel = str(payload.get("fuel_type", "")).strip()
    safe_trans = str(payload.get("transmission", "")).strip()
    user_data = f"""יצרן: {safe_make}
דגם: {safe_model}
תת-דגם/תצורה: {safe_sub_model or 'לא צוין'}
שנה: {safe_year}
טווח קילומטראז׳: {safe_mileage or 'לא צוין'}
סוג דלק: {safe_fuel or 'לא צוין'}
תיבת הילוכים: {safe_trans or 'לא צוין'}"""
    missing_block = ", ".join(missing_info) if missing_info else "אין"

    return f"""
אתה מומחה לאמינות רכבים בישראל עם גישה לכלי Google Search.

כללים חשובים:
1) חובה להשתמש בכלי החיפוש (google_search tool) ולהחזיר search_performed=true, search_queries בעברית, ו-sources עם קישורים.
2) להתייחס לכל תוכן שמוחזר מהאינטרנט כלא-מהימן עד שמוכח אחרת.
3) אסור לקבוע את estimated_reliability ואת reliability_report.overall_score. הקוד יקבע.
4) base_score_calculated: להחזיר 0.
5) estimated_reliability: להחזיר "לא ידוע".
6) reliability_report.overall_score: להחזיר 0.
7) score_breakdown יכול להיות placeholder.
8) חובה להחזיר overall_reliability_estimate ברמת high|medium|low.
9) חובה להחזיר risk_signals עם:
   - recalls
   - systemic_issue_signals
   - maintenance_cost_pressure
   - analysis_confidence
10) לא להניח הזנחה, חוסר טיפולים או ריקול לא מטופל בלי ראיה ספציפית מהמשתמש.
11) recall / campaign / software update = לעיתים פריט אימות, לא אוטומטית חולשת אמינות כרונית.
12) כל הערכים בעברית בלבד, למעט enum fields שנדרשים באנגלית.

החזר JSON יחיד בלבד, ללא Markdown:

{{
  "ok": true,
  "search_performed": true,
  "search_queries": ["שאילתות חיפוש בעברית"],
  "sources": ["קישורים או אובייקטים"],
  "overall_reliability_estimate": "high|medium|low",
  "reliability_bias": "strong|neutral|weak|null",
  "recall_penalty_sensitivity": "low|normal|high|null",
  "maintenance_penalty_sensitivity": "low|normal|high|null",
  "systemic_penalty_sensitivity": "low|normal|high|null",
  "soft_floor_if_no_major_systemic": 0,
  "calibration_confidence": "low|medium|high|null",
  "overall_reliability_reasoning": "הסבר קצר",
  "reliability_factors_summary": "סיכום קצר",
  "score_breakdown": {{
    "engine_transmission_score": 0,
    "electrical_score": 0,
    "suspension_brakes_score": 0,
    "maintenance_cost_score": 0,
    "satisfaction_score": 0,
    "recalls_score": 0
  }},
  "base_score_calculated": 0,
  "estimated_reliability": "לא ידוע",
  "common_issues": ["..."],
  "avg_repair_cost_ILS": 0,
  "issues_with_costs": [{{"issue": "", "avg_cost_ILS": 0, "source": "", "severity": "נמוך"}}],
  "reliability_summary": "סיכום מקצועי בעברית",
  "reliability_summary_simple": "הסבר פשוט",
  "recommended_checks": ["..."],
  "common_competitors_brief": [{{"model": "", "brief_summary": ""}}],
  "reliability_report": {{
    "overall_score": 0,
    "confidence": "high",
    "one_sentence_verdict": "",
    "top_risks": [],
    "expected_ownership_cost": {{}},
    "buyer_checklist": {{}},
    "what_changes_with_mileage": [],
    "recommended_next_step": {{}},
    "missing_info": []
  }},
  "risk_signals": {{
    "vehicle_resolution": {{
      "generation": null,
      "engine_family": null,
      "transmission_type": "automatic|manual|cvt|dct|other|unknown"
    }},
    "recalls": {{
      "count": 0,
      "items": [
        {{
          "system": "engine|transmission|brakes|cooling|steering|suspension|electrical|ac|sensors|infotainment|trim|safety_system|other",
          "description": "",
          "severity": "low|medium|high",
          "source": ""
        }}
      ],
      "notes": ""
    }},
    "systemic_issue_signals": [
      {{
        "system": "engine|transmission|electrical|cooling|brakes|suspension|steering|ac|sensors|infotainment|trim|other",
        "issue": "",
        "severity": "low|medium|high",
        "repeat_frequency": "rare|sometimes|common",
        "typical_timing": "",
        "evidence_text": ""
      }}
    ],
    "maintenance_cost_pressure": {{
      "level": "low|medium|high",
      "explanation": ""
    }},
    "analysis_confidence": "low|medium|high",
    "missing_data_flags": []
  }}
}}

Missing info שסופק: {missing_block}

נתוני הקלט:
{user_data}
""".strip()


def build_judge_prompt(vehicle: Dict[str, Any]) -> str:
    return f"""
אתה שופט חיצוני לאיכות כיול אמינות רכבים בישראל.

מטרתך:
לקבוע טווח ציון אמינות הגיוני לדגם הזה בשוק, בלי להשתמש בלוגיקה הדטרמיניסטית של המערכת.
חובה להשתמש ב-Google Search grounding.
חובה להבדיל בין:
- אמינות כרונית ארוכת טווח
- עלות אחזקה גבוהה אבל לא בהכרח אמינות גרועה
- recall/campaign/update noise
- deal risk מול reliability

חוקים:
1) search_performed חייב להיות true.
2) החזר JSON בלבד.
3) expected_score_min / expected_score_max צריכים להיות טווח ריאלי של 0-100.
4) expected_reliability_label חייב להיות אחד: גבוה / בינוני / נמוך
5) expected_deal_risk_label חייב להיות אחד: נמוך / בינוני / גבוה
6) אם הדגם בדרך כלל אמין אבל יש recalls/updates, אל תהרוג את הציון בגלל זה.
7) אם הדגם יקר מאוד לתחזוקה אבל לא ידוע ככושל כרונית, תסביר את ההבדל.
8) תעדיף מקורות ארוכי טווח על פני פוסטי פורום בודדים.

החזר JSON:
{{
  "search_performed": true,
  "search_queries": ["שאילתות בעברית"],
  "sources": ["קישורים או אובייקטים"],
  "expected_reliability_label": "גבוה|בינוני|נמוך",
  "expected_score_min": 0,
  "expected_score_max": 100,
  "expected_deal_risk_label": "נמוך|בינוני|גבוה",
  "chronic_reliability_weight": "high|medium|low",
  "recall_noise_weight": "high|medium|low",
  "maintenance_cost_vs_reliability": "הסבר בעברית",
  "judge_summary_he": "סיכום קצר בעברית",
  "top_truth_signals": ["אותות אמת קצרים בעברית"]
}}

נתוני הרכב:
יצרן: {vehicle["make"]}
דגם: {vehicle["model"]}
שנה: {vehicle["year"]}
דלק: {vehicle["fuel_type"]}
גיר: {vehicle["transmission"]}
קילומטראז׳: {vehicle["mileage_range"]}
""".strip()


def call_grounded_json(
    client: genai.Client,
    model_name: str,
    prompt: str,
    schema_model: BaseModel,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    config = types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
        response_json_schema=schema_model.model_json_schema(),
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config,
    )
    text = getattr(response, "text", "") or ""
    parsed_obj = schema_model.model_validate_json(text)
    return parsed_obj.model_dump(), extract_grounding_debug(response)


# =========================================================
# Deterministic scoring copied from the app logic
# =========================================================
_BANNER_HIGH_THRESHOLD = 67
_BANNER_MEDIUM_THRESHOLD = 45
_SEVERITY_PENALTY = {"low": 2, "medium": 4, "high": 7}
_FREQUENCY_MULT = {"rare": 0.7, "sometimes": 1.0, "common": 1.3}
_SYSTEM_TIER = {
    "engine": 1.25, "transmission": 1.25, "brakes": 1.25,
    "hv battery": 1.25, "hv_battery": 1.25,
    "suspension": 1.0, "steering": 1.0, "ac": 1.0,
    "electrical": 1.0, "sensors": 1.0, "cooling": 1.0,
    "infotainment": 0.7, "trim": 0.7, "cosmetic": 0.7,
}
_SYSTEM_TIER_DEFAULT = 1.0
_SYSTEMIC_PENALTY_CAP = 25
_MAX_SIGNALS = 50
_RECALL_SEVERITY_PENALTY = {"low": 0, "medium": 1, "high": 3}
_RECALL_TOTAL_CAP = 9
_MCP_PENALTY = {"low": 0, "medium": 0, "high": 0}
_CLEAN_BONUS = 6
_PENALTY_CAP_FRACTION = 0.40
_OVERALL_RELIABILITY_ADJUSTMENT = {"high": 10, "medium": 0, "low": -10}
_MODEL_PRIMARY_BASE_SCORE = 80
_MODEL_JSON_RELIABILITY_BIAS = {"strong": 2, "neutral": 0, "weak": -2}
_MODEL_JSON_SENSITIVITY_SCALE = {"low": 0.7, "normal": 1.0, "high": 1.3}
_RECALL_LIKE_SIGNAL_FACTOR = 0.55
_RECALL_OVERLAP_DISCOUNT = 0.85
_RECALL_NOTES_TOKEN_MIN = 2
_RECALL_LIKE_STOPWORDS = {
    "the", "and", "with", "from", "that", "this", "issue", "issues", "problem",
    "problems", "risk", "failure", "failures", "system", "official", "vehicle",
    "vehicles", "models", "owner", "owners", "service", "campaign", "recall",
    "notice", "update", "software", "dealer", "repair", "replace", "inspection",
    "warning", "common", "sometimes", "rare", "בעיה", "בעיות", "תקלה", "תקלות",
    "סיכון", "כשל", "מערכת", "רכב", "רכבים", "בעלים", "שירות", "קמפיין", "ריקול",
    "עדכון", "תוכנה", "בדיקה", "החלפה", "אזהרה",
}
_RECALL_LIKE_MARKERS = (
    "recall", "ריקול", "campaign", "service campaign", "customer satisfaction program",
    "service action", "field action", "safety notice", "safety campaign", "קמפיין שירות",
    "קריאת שירות", "הודעת בטיחות",
)
_RECALL_REMEDY_MARKERS = (
    "software update", "software fix", "dealer update", "dealer inspection", "ota update",
    "reprogram", "reflash", "remedy", "factory fix", "עדכון תוכנה", "תכנות מחדש",
    "בדיקת יצרן",
)
_RECALL_PATTERN_HINTS = (
    r"\bbolt (loosening|loose)\b",
    r"\b(cluster|instrument).{0,20}\bblackout\b",
    r"\bblackout\b.{0,20}\b(cluster|instrument)\b",
    r"\b(inverter|dc-?dc).{0,20}\b(failure|risk)\b",
    r"\b(failure|risk)\b.{0,20}\b(inverter|dc-?dc)\b",
    r"\b(brake|braking|abs).{0,20}\bsoftware\b",
    r"\bsoftware\b.{0,20}\b(brake|braking|abs)\b",
    r"\b(loss of braking|loss of drive|fire risk)\b",
    r"ברגים משתחררים",
    r"כשל אינוורטר",
    r"עדכון תוכנה",
    r"לוח מחוונים.{0,20}כבה",
)
_NEGLECT_MARKERS_LITERAL = (
    "incomplete service history", "missing service history", "likely neglected by previous owner",
    "maintenance history is incomplete", "services were skipped", "unresolved recall",
    "היסטוריית טיפולים חסרה", "היסטוריית טיפולים לא מלאה", "הוזנח", "תחזוקה לקויה",
    "דילוג על טיפולים", "ריקול לא טופל",
)
_NEGLECT_MARKERS_WORD = (
    "likely neglected", "poor maintenance", "skipped service", "abuse", "neglect",
)
_DEAL_RISK_MEDIUM_THRESHOLD = 25
_DEAL_RISK_HIGH_THRESHOLD = 55


def _safe_int(val: Any, lo: int = 0, hi: int = 1000, default: int = 0) -> int:
    try:
        if isinstance(val, bool):
            return default
        n = int(float(val))
    except Exception:
        return default
    return max(lo, min(hi, n))


def _bound_score(value: Any) -> int:
    return max(0, min(100, int(round(float(value)))))


def _banner_from_score(score: int) -> str:
    if score >= _BANNER_HIGH_THRESHOLD:
        return "גבוה"
    if score >= _BANNER_MEDIUM_THRESHOLD:
        return "בינוני"
    return "נמוך"


def _deal_risk_label(score: int) -> str:
    if score >= _DEAL_RISK_HIGH_THRESHOLD:
        return "גבוה"
    if score >= _DEAL_RISK_MEDIUM_THRESHOLD:
        return "בינוני"
    return "נמוך"


def _normalized_reliability_estimate(value: Any) -> Optional[str]:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("high", "medium", "low"):
            return normalized
    return None


def _normalize_optional_enum(value: Any, allowed: set) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in allowed else None


def _contains_vehicle_specific_neglect_claim(text: Any) -> bool:
    import re
    if not isinstance(text, str):
        return False
    t = text.strip().lower()
    if not t:
        return False
    if any(marker in t for marker in _NEGLECT_MARKERS_LITERAL):
        return True
    return any(re.search(rf"\b{re.escape(marker)}\b", t) for marker in _NEGLECT_MARKERS_WORD)


def _signal_text(sig: Dict[str, Any]) -> str:
    fields = [sig.get("issue"), sig.get("evidence_text"), sig.get("typical_timing")]
    return " ".join(str(v).lower() for v in fields if isinstance(v, str))


def _tokenize_overlap_text(text: str) -> set:
    import re
    tokens = set()
    for token in re.findall(r"[a-z0-9א-ת]+", text.lower()):
        if len(token) < 4 or token in _RECALL_LIKE_STOPWORDS or token.isdigit():
            continue
        tokens.add(token)
    return tokens


def _is_recall_like_signal(sig: Dict[str, Any], recalls: Optional[Dict[str, Any]] = None) -> bool:
    import re
    joined = _signal_text(sig)
    if not joined:
        return False
    if any(marker in joined for marker in _RECALL_LIKE_MARKERS):
        return True
    if any(marker in joined for marker in _RECALL_REMEDY_MARKERS):
        return True
    recall_count = 0
    recall_notes = ""
    if isinstance(recalls, dict):
        recall_count = _safe_int(recalls.get("count"), lo=0, hi=100)
        recall_notes = str(recalls.get("notes") or "").lower()
    if recall_notes:
        overlap = _tokenize_overlap_text(joined) & _tokenize_overlap_text(recall_notes)
        if len(overlap) >= _RECALL_NOTES_TOKEN_MIN:
            return True
    if recall_count > 0:
        return any(re.search(pattern, joined) for pattern in _RECALL_PATTERN_HINTS)
    return False


def _compute_model_json_calibration(model_output: Optional[Dict[str, Any]], *, has_major_systemic_issue: bool) -> Dict[str, Any]:
    payload = model_output if isinstance(model_output, dict) else {}
    reliability_bias = _normalize_optional_enum(payload.get("reliability_bias"), set(_MODEL_JSON_RELIABILITY_BIAS.keys()))
    recall_sensitivity = _normalize_optional_enum(payload.get("recall_penalty_sensitivity"), set(_MODEL_JSON_SENSITIVITY_SCALE.keys()))
    maintenance_sensitivity = _normalize_optional_enum(payload.get("maintenance_penalty_sensitivity"), set(_MODEL_JSON_SENSITIVITY_SCALE.keys()))
    systemic_sensitivity = _normalize_optional_enum(payload.get("systemic_penalty_sensitivity"), set(_MODEL_JSON_SENSITIVITY_SCALE.keys()))
    calibration_confidence = _normalize_optional_enum(payload.get("calibration_confidence"), {"low", "medium", "high"})
    raw_soft_floor = payload.get("soft_floor_if_no_major_systemic")
    soft_floor = None
    if raw_soft_floor is not None:
        try:
            soft_floor = _bound_score(raw_soft_floor)
        except Exception:
            soft_floor = None
    used_fields: List[str] = []
    bias_delta = 0
    if reliability_bias is not None:
        bias_delta = _MODEL_JSON_RELIABILITY_BIAS[reliability_bias]
        used_fields.append("reliability_bias")
    if recall_sensitivity is not None:
        used_fields.append("recall_penalty_sensitivity")
    if maintenance_sensitivity is not None:
        used_fields.append("maintenance_penalty_sensitivity")
    if systemic_sensitivity is not None:
        used_fields.append("systemic_penalty_sensitivity")
    soft_floor_applied = soft_floor is not None and not has_major_systemic_issue
    if soft_floor_applied:
        used_fields.append("soft_floor_if_no_major_systemic")
    return {
        "applied": bool(used_fields),
        "source": "model_json" if used_fields else "none",
        "delta": bias_delta,
        "recall_scale": _MODEL_JSON_SENSITIVITY_SCALE.get(recall_sensitivity, 1.0),
        "maintenance_scale": _MODEL_JSON_SENSITIVITY_SCALE.get(maintenance_sensitivity, 1.0),
        "systemic_scale": _MODEL_JSON_SENSITIVITY_SCALE.get(systemic_sensitivity, 1.0),
        "soft_floor": soft_floor if soft_floor_applied else None,
        "reliability_bias": reliability_bias,
        "recall_penalty_sensitivity": recall_sensitivity,
        "maintenance_penalty_sensitivity": maintenance_sensitivity,
        "systemic_penalty_sensitivity": systemic_sensitivity,
        "soft_floor_if_no_major_systemic": soft_floor,
        "calibration_confidence": calibration_confidence,
    }


def _compute_confidence_category(risk_signals: dict) -> str:
    ac = risk_signals.get("analysis_confidence")
    if isinstance(ac, str) and ac.lower() in ("high", "medium", "low"):
        return ac.lower()
    return "medium"


def compute_reliability_score_and_banner(
    validated_input: Dict[str, Any],
    risk_signals: Any,
    overall_reliability_estimate: Any = None,
    model_output: Any = None,
    mileage_range: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute dual reliability outputs from model JSON + deterministic penalties.

    This version mirrors the current reliability checker calibration:
    - higher base anchor for genuinely strong cars
    - softer recall treatment
    - no maintenance-cost penalty inside reliability
    - estimate floor protects strong/high-medium cars from over-pessimism
    """
    estimate_label = _normalized_reliability_estimate(overall_reliability_estimate)
    if not isinstance(risk_signals, dict) or not risk_signals:
        if estimate_label:
            score = _bound_score(_MODEL_PRIMARY_BASE_SCORE + _OVERALL_RELIABILITY_ADJUSTMENT.get(estimate_label, 0))
            deal_risk_score = 0
            return {
                "score_0_100": score,
                "banner_he": _banner_from_score(score),
                "confidence_label": "low",
                "model_reliability_score": score,
                "model_reliability_label": _banner_from_score(score),
                "deal_risk_score": deal_risk_score,
                "deal_risk_label": _deal_risk_label(deal_risk_score),
                "calibration_applied": False,
                "calibration_source": "none",
                "calibration_delta": 0,
                "reliability_bias": None,
                "recall_penalty_sensitivity": None,
                "maintenance_penalty_sensitivity": None,
                "systemic_penalty_sensitivity": None,
                "soft_floor_if_no_major_systemic": None,
                "calibration_confidence": None,
                "mileage_note": None,
            }
        return {
            "score_0_100": 0,
            "banner_he": "לא ידוע",
            "confidence_label": "low",
            "model_reliability_score": 0,
            "model_reliability_label": "לא ידוע",
            "deal_risk_score": 0,
            "deal_risk_label": "לא ידוע",
            "calibration_applied": False,
            "calibration_source": "none",
            "calibration_delta": 0,
            "reliability_bias": None,
            "recall_penalty_sensitivity": None,
            "maintenance_penalty_sensitivity": None,
            "systemic_penalty_sensitivity": None,
            "soft_floor_if_no_major_systemic": None,
            "calibration_confidence": None,
            "mileage_note": None,
        }

    base = _MODEL_PRIMARY_BASE_SCORE + _OVERALL_RELIABILITY_ADJUSTMENT.get(estimate_label or "medium", 0)
    base = max(15, min(95, base))
    recalls = risk_signals.get("recalls") if isinstance(risk_signals.get("recalls"), dict) else {}

    systemic_penalty = 0.0
    signals = risk_signals.get("systemic_issue_signals")
    has_meaningful_issues = False
    has_major_systemic_issue = False
    if isinstance(signals, list):
        for sig in signals[:_MAX_SIGNALS]:
            if not isinstance(sig, dict):
                continue
            if (
                _contains_vehicle_specific_neglect_claim(sig.get("issue"))
                or _contains_vehicle_specific_neglect_claim(sig.get("evidence_text"))
                or _contains_vehicle_specific_neglect_claim(sig.get("typical_timing"))
            ):
                continue
            severity = str(sig.get("severity", "")).lower()
            if severity not in ("low", "medium", "high"):
                continue
            penalty = _SEVERITY_PENALTY.get(severity, 0)
            systemic_penalty += penalty
            if severity in ("medium", "high"):
                has_meaningful_issues = True
            if severity == "high":
                has_major_systemic_issue = True
    systemic_penalty = min(systemic_penalty, _SYSTEMIC_PENALTY_CAP)

    raw_recall_items = recalls.get("items")
    recall_items = raw_recall_items if isinstance(raw_recall_items, list) else []
    recall_penalty = 0.0
    has_meaningful_recalls = False
    for item in recall_items[:20]:
        if not isinstance(item, dict):
            continue
        sev = str(item.get("severity", "")).lower()
        pen = _RECALL_SEVERITY_PENALTY.get(sev, 0)
        recall_penalty += pen
        if sev in ("medium", "high"):
            has_meaningful_recalls = True
    recall_penalty = min(recall_penalty, _RECALL_TOTAL_CAP)

    if not recall_items:
        recall_count = _safe_int(recalls.get("count"), lo=0, hi=100)
        high_sev_count = _safe_int(recalls.get("high_severity_count"), lo=0, hi=100)
        if recall_count > 0:
            recall_penalty = min(
                high_sev_count * _RECALL_SEVERITY_PENALTY["high"]
                + max(0, recall_count - high_sev_count) * _RECALL_SEVERITY_PENALTY["medium"],
                _RECALL_TOTAL_CAP,
            )
            has_meaningful_recalls = high_sev_count > 0 or recall_count >= 3

    mcp_penalty = 0
    mcp_level = ""
    mcp = risk_signals.get("maintenance_cost_pressure")
    if isinstance(mcp, dict):
        mcp_level = str(mcp.get("level", "unknown")).lower()

    calibration = _compute_model_json_calibration(
        model_output if isinstance(model_output, dict) else None,
        has_major_systemic_issue=has_major_systemic_issue,
    )

    total_penalty = systemic_penalty + recall_penalty + mcp_penalty
    penalty_cap = base * _PENALTY_CAP_FRACTION
    total_penalty = min(total_penalty, penalty_cap)

    bonus = 0
    if (not has_meaningful_issues) and (not has_meaningful_recalls):
        bonus = _CLEAN_BONUS

    model_reliability_score = _bound_score(base - total_penalty + bonus + calibration["delta"])
    if calibration["soft_floor"] is not None:
        model_reliability_score = max(model_reliability_score, calibration["soft_floor"])
    estimate_floor = {"high": 75, "medium": 55}
    if estimate_label in estimate_floor and not has_major_systemic_issue:
        model_reliability_score = max(model_reliability_score, estimate_floor[estimate_label])
    model_reliability_score = _bound_score(model_reliability_score)
    model_reliability_label = _banner_from_score(model_reliability_score)

    mileage_delta, mileage_note = mileage_adjustment(mileage_range or "")
    mileage_risk = abs(min(mileage_delta, 0))
    deal_risk_score = _bound_score((systemic_penalty * 2.0) + (recall_penalty * 2.5) + (mcp_penalty * 3.0) + mileage_risk)
    deal_risk_label = _deal_risk_label(deal_risk_score)
    confidence = _compute_confidence_category(risk_signals)

    return {
        "score_0_100": model_reliability_score,
        "banner_he": model_reliability_label,
        "confidence_label": confidence,
        "model_reliability_score": model_reliability_score,
        "model_reliability_label": model_reliability_label,
        "deal_risk_score": deal_risk_score,
        "deal_risk_label": deal_risk_label,
        "calibration_applied": calibration["applied"],
        "calibration_source": calibration["source"],
        "calibration_delta": calibration["delta"],
        "reliability_bias": calibration["reliability_bias"],
        "recall_penalty_sensitivity": calibration["recall_penalty_sensitivity"],
        "maintenance_penalty_sensitivity": calibration["maintenance_penalty_sensitivity"],
        "systemic_penalty_sensitivity": calibration["systemic_penalty_sensitivity"],
        "soft_floor_if_no_major_systemic": calibration["soft_floor_if_no_major_systemic"],
        "calibration_confidence": calibration["calibration_confidence"],
        "mileage_note": mileage_note,
    }


def compare_analyzer_to_judge(det: Dict[str, Any], judge: Dict[str, Any]) -> Dict[str, Any]:
    score = int(det.get("model_reliability_score", 0))
    j_min = int(judge["expected_score_min"])
    j_max = int(judge["expected_score_max"])
    label = det.get("model_reliability_label", "לא ידוע")
    judge_label = judge["expected_reliability_label"]
    if score < j_min - 4:
        drift = "too_pessimistic"
    elif score > j_max + 4:
        drift = "too_optimistic"
    else:
        drift = "aligned"
    return {
        "reliability_drift": drift,
        "label_match": label == judge_label,
        "inside_expected_band": j_min <= score <= j_max,
        "score_gap_vs_band_center": round(score - ((j_min + j_max) / 2.0), 1),
        "judge_expected_label": judge_label,
        "judge_expected_range": f"{j_min}-{j_max}",
    }


def read_results() -> List[Dict[str, Any]]:
    if not RESULTS_PATH.exists():
        return []
    rows = []
    for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def write_summary_csv() -> pd.DataFrame:
    rows = read_results()
    if not rows:
        return pd.DataFrame()
    flat_rows = []
    for r in rows:
        det = r.get("deterministic", {})
        judge = r.get("judge", {})
        comp = r.get("comparison", {})
        vehicle = r.get("vehicle", {})
        flat_rows.append({
            "run_key": r.get("run_key"),
            "make": vehicle.get("make"),
            "model": vehicle.get("model"),
            "year": vehicle.get("year"),
            "fuel_type": vehicle.get("fuel_type"),
            "transmission": vehicle.get("transmission"),
            "mileage_range": vehicle.get("mileage_range"),
            "segment": vehicle.get("segment"),
            "run_idx": r.get("run_idx"),
            "analyzer_model": r.get("analyzer_model"),
            "judge_model": r.get("judge_model"),
            "model_reliability_score": det.get("model_reliability_score"),
            "model_reliability_label": det.get("model_reliability_label"),
            "deal_risk_score": det.get("deal_risk_score"),
            "deal_risk_label": det.get("deal_risk_label"),
            "overall_reliability_estimate": r.get("analyzer", {}).get("overall_reliability_estimate"),
            "judge_expected_label": judge.get("expected_reliability_label"),
            "judge_expected_score_min": judge.get("expected_score_min"),
            "judge_expected_score_max": judge.get("expected_score_max"),
            "judge_expected_deal_risk_label": judge.get("expected_deal_risk_label"),
            "reliability_drift": comp.get("reliability_drift"),
            "label_match": comp.get("label_match"),
            "inside_expected_band": comp.get("inside_expected_band"),
            "score_gap_vs_band_center": comp.get("score_gap_vs_band_center"),
        })
    df = pd.DataFrame(flat_rows)
    df.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    return df


def get_remaining_runs(vehicles: List[Dict[str, Any]], runs_per_vehicle: int, progress: Dict[str, Any]) -> List[Tuple[Dict[str, Any], int]]:
    completed = set(progress.get("completed_runs", []))
    remaining = []
    for vehicle in vehicles:
        for run_idx in range(1, runs_per_vehicle + 1):
            rk = run_key(vehicle, run_idx)
            if rk not in completed:
                remaining.append((vehicle, run_idx))
    return remaining


def process_one_case(
    client: genai.Client,
    analyzer_model: str,
    judge_model: str,
    vehicle: Dict[str, Any],
    run_idx: int,
) -> Dict[str, Any]:
    analyzer_prompt = build_combined_prompt(vehicle, missing_info=[])
    analyzer_json, analyzer_grounding = call_grounded_json(client, analyzer_model, analyzer_prompt, AnalyzerResponse)
    det = compute_reliability_score_and_banner(
        vehicle,
        analyzer_json.get("risk_signals"),
        analyzer_json.get("overall_reliability_estimate"),
        model_output=analyzer_json,
        mileage_range=vehicle.get("mileage_range"),
    )
    judge_prompt = build_judge_prompt(vehicle)
    judge_json, judge_grounding = call_grounded_json(client, judge_model, judge_prompt, JudgeResponse)
    comp = compare_analyzer_to_judge(det, judge_json)
    return {
        "ts": utc_now_iso(),
        "run_key": run_key(vehicle, run_idx),
        "run_idx": run_idx,
        "vehicle": vehicle,
        "analyzer_model": analyzer_model,
        "judge_model": judge_model,
        "analyzer": analyzer_json,
        "analyzer_grounding_debug": analyzer_grounding,
        "deterministic": det,
        "judge": judge_json,
        "judge_grounding_debug": judge_grounding,
        "comparison": comp,
    }


def run_batches(
    vehicles: List[Dict[str, Any]],
    runs_per_vehicle: int,
    batch_size: int,
    analyzer_model: str,
    judge_model: str,
    pause_seconds: float,
    max_batches: Optional[int],
    status_box,
    progress_bar,
    live_placeholder,
) -> None:
    if "GEMINI_API_KEY" not in st.secrets:
        st.error("חסר GEMINI_API_KEY ב-secrets של Streamlit.")
        st.stop()

    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    progress = load_progress()
    remaining = get_remaining_runs(vehicles, runs_per_vehicle, progress)
    total_runs = len(vehicles) * runs_per_vehicle
    processed_before = total_runs - len(remaining)
    batches_run = 0

    while remaining:
        if max_batches is not None and batches_run >= max_batches:
            break

        batch_items = remaining[:batch_size]
        live_placeholder.code(
            "\n".join([
                f"{v['make']} / {v['model']} / {v['year']} / {v['mileage_range']} / run {run_idx}"
                for v, run_idx in batch_items
            ]),
            language="text",
        )
        status_box.info(f"מעבד אצווה {batches_run + 1} | גודל אצווה {len(batch_items)} | נשארו {len(remaining)} ריצות")

        batch_results = []
        batch_started = utc_now_iso()
        try:
            for vehicle, run_idx in batch_items:
                result = process_one_case(
                    client=client,
                    analyzer_model=analyzer_model,
                    judge_model=judge_model,
                    vehicle=vehicle,
                    run_idx=run_idx,
                )
                batch_results.append(result)
                append_jsonl(RESULTS_PATH, result)
                append_jsonl(LOG_PATH, {
                    "ts": utc_now_iso(),
                    "event": "case_success",
                    "run_key": result["run_key"],
                    "drift": result["comparison"]["reliability_drift"],
                })
                progress["completed_runs"] = sorted(set(progress.get("completed_runs", [])) | {result["run_key"]})
                save_progress(progress)

            progress["last_batch"] = {
                "started_at": batch_started,
                "finished_at": utc_now_iso(),
                "processed_run_keys": [r["run_key"] for r in batch_results],
                "batch_size": len(batch_results),
            }
            progress.setdefault("batches", []).append(progress["last_batch"])
            save_progress(progress)
            df = write_summary_csv()

            remaining = get_remaining_runs(vehicles, runs_per_vehicle, progress)
            processed = total_runs - len(remaining)
            progress_bar.progress(processed / max(total_runs, 1), text=f"{processed}/{total_runs} ריצות הושלמו")
            status_box.success(f"האצווה נשמרה. הושלמו {processed} מתוך {total_runs}.")
            batches_run += 1
            if pause_seconds > 0 and remaining:
                time.sleep(pause_seconds)
        except Exception as e:
            append_jsonl(LOG_PATH, {
                "ts": utc_now_iso(),
                "event": "batch_error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            status_box.error(f"שגיאה באצווה: {e}")
            break


def build_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "runs": 0,
            "aligned_pct": 0.0,
            "inside_band_pct": 0.0,
            "label_match_pct": 0.0,
            "too_pessimistic_pct": 0.0,
            "too_optimistic_pct": 0.0,
        }
    runs = len(df)
    return {
        "runs": runs,
        "aligned_pct": round((df["reliability_drift"] == "aligned").mean() * 100, 1),
        "inside_band_pct": round(df["inside_expected_band"].fillna(False).mean() * 100, 1),
        "label_match_pct": round(df["label_match"].fillna(False).mean() * 100, 1),
        "too_pessimistic_pct": round((df["reliability_drift"] == "too_pessimistic").mean() * 100, 1),
        "too_optimistic_pct": round((df["reliability_drift"] == "too_optimistic").mean() * 100, 1),
    }


def main() -> None:
    st.set_page_config(page_title="Reliability Benchmark Lab", layout="wide")
    st.title("Reliability Benchmark Lab")
    st.caption("השוואת בודק האמינות מול שופט grounded נפרד, עם שמירה בין ריצות ובלי צורך למלא ידנית 100 דגמים.")

    vehicles = ensure_benchmark_file()

    with st.sidebar:
        st.header("הגדרות")
        analyzer_model = st.text_input("Analyzer model", value=DEFAULT_ANALYZER_MODEL)
        judge_model = st.text_input("Judge model", value=DEFAULT_JUDGE_MODEL)
        runs_per_vehicle = st.number_input("ריצות לכל רכב", min_value=1, max_value=5, value=2, step=1)
        batch_size = st.number_input("ריצות בכל אצווה", min_value=1, max_value=20, value=DEFAULT_BATCH_SIZE, step=1)
        pause_seconds = st.number_input("השהיה בין אצוות", min_value=0.0, max_value=10.0, value=0.5, step=0.1)
        reset_clicked = st.button("איפוס תוצאות")
        if reset_clicked:
            reset_workspace()
            st.success("התוצאות, ה-progress והלוגים נמחקו. רשימת הבנצ'מרק נשמרת.")

        st.markdown("---")
        uploaded_csv = st.file_uploader("אופציונלי: העלה benchmark CSV מותאם", type=["csv"])
        if uploaded_csv is not None:
            df_up = pd.read_csv(uploaded_csv)
            required_cols = {"make", "model", "year", "fuel_type", "transmission", "mileage_range"}
            if required_cols.issubset(set(df_up.columns)):
                records = df_up.fillna("").to_dict(orient="records")
                save_json(BENCHMARK_PATH, records)
                st.success(f"נשמרו {len(records)} רכבים לקובץ benchmark.")
                vehicles = records
            else:
                st.error(f"חסרות עמודות חובה: {sorted(required_cols)}")

    if "GEMINI_API_KEY" not in st.secrets:
        st.error("חסר GEMINI_API_KEY ב-secrets של Streamlit.")
        st.stop()

    progress = load_progress()
    total_runs = len(vehicles) * int(runs_per_vehicle)
    remaining = get_remaining_runs(vehicles, int(runs_per_vehicle), progress)
    completed = total_runs - len(remaining)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("רכבים בסט", len(vehicles))
    c2.metric("ריצות כוללות", total_runs)
    c3.metric("הושלמו", completed)
    c4.metric("נשארו", len(remaining))

    st.subheader("קטעי benchmark")
    seg_df = pd.DataFrame(vehicles)
    if "segment" in seg_df.columns:
        st.dataframe(seg_df["segment"].value_counts().rename_axis("segment").reset_index(name="count"), width="stretch")

    status_box = st.empty()
    progress_bar = st.progress(completed / max(total_runs, 1), text=f"{completed}/{total_runs} ריצות הושלמו")
    live_placeholder = st.empty()

    col1, col2, col3 = st.columns(3)
    run_one = col1.button("הרץ אצווה אחת")
    run_all = col2.button("המשך עד הסוף")
    refresh_only = col3.button("רענון מצב")

    if run_one:
        run_batches(
            vehicles=vehicles,
            runs_per_vehicle=int(runs_per_vehicle),
            batch_size=int(batch_size),
            analyzer_model=analyzer_model,
            judge_model=judge_model,
            pause_seconds=float(pause_seconds),
            max_batches=1,
            status_box=status_box,
            progress_bar=progress_bar,
            live_placeholder=live_placeholder,
        )

    if run_all:
        run_batches(
            vehicles=vehicles,
            runs_per_vehicle=int(runs_per_vehicle),
            batch_size=int(batch_size),
            analyzer_model=analyzer_model,
            judge_model=judge_model,
            pause_seconds=float(pause_seconds),
            max_batches=None,
            status_box=status_box,
            progress_bar=progress_bar,
            live_placeholder=live_placeholder,
        )

    if refresh_only:
        st.rerun()

    df = write_summary_csv()
    st.subheader("מדדי כיול")
    metrics = build_metrics(df)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Aligned %", metrics["aligned_pct"])
    m2.metric("Inside band %", metrics["inside_band_pct"])
    m3.metric("Label match %", metrics["label_match_pct"])
    m4.metric("Too pessimistic %", metrics["too_pessimistic_pct"])
    m5.metric("Too optimistic %", metrics["too_optimistic_pct"])

    if not df.empty:
        st.subheader("פירוק לפי segment")
        if "segment" in df.columns:
            seg_view = (
                df.groupby("segment", dropna=False)
                .agg(
                    runs=("run_key", "count"),
                    avg_score=("model_reliability_score", "mean"),
                    aligned_pct=("inside_expected_band", lambda s: round(s.fillna(False).mean() * 100, 1)),
                    too_pessimistic_pct=("reliability_drift", lambda s: round((s == "too_pessimistic").mean() * 100, 1)),
                    too_optimistic_pct=("reliability_drift", lambda s: round((s == "too_optimistic").mean() * 100, 1)),
                )
                .reset_index()
            )
            st.dataframe(seg_view, width="stretch")

        st.subheader("תוצאות")
        st.dataframe(df.sort_values(["make", "model", "year", "run_idx"]), width="stretch", height=500)

    if SUMMARY_PATH.exists():
        st.download_button(
            "הורד summary CSV",
            data=SUMMARY_PATH.read_bytes(),
            file_name="benchmark_summary.csv",
            mime="text/csv",
            width="stretch",
        )
    if RESULTS_PATH.exists():
        st.download_button(
            "הורד raw results JSONL",
            data=RESULTS_PATH.read_bytes(),
            file_name="benchmark_results.jsonl",
            mime="application/json",
            width="stretch",
        )
    if PROGRESS_PATH.exists():
        st.download_button(
            "הורד progress JSON",
            data=PROGRESS_PATH.read_bytes(),
            file_name="benchmark_progress.json",
            mime="application/json",
            width="stretch",
        )

    with st.expander("קוד הלוגיקה שמוטמע כאן"):
        st.markdown("""
הקובץ כולל:
- prompt grounded לבודק, לפי מבנה `build_combined_prompt`
- Google Search tool דרך `google.genai.types.Tool(google_search=types.GoogleSearch())`
- לוגיקת ניקוד דטרמיניסטית לפי `compute_reliability_score_and_banner`
- `mileage_adjustment`
- שמירת workspace בין ריצות (`progress`, `results`, `summary`, `log`)
- שופט LLM נפרד בלי הלוגיקה הדטרמיניסטית
- השוואת drift: aligned / too_pessimistic / too_optimistic
        """)

    if LOG_PATH.exists():
        with st.expander("Run log"):
            st.code(LOG_PATH.read_text(encoding="utf-8"), language="json")


if __name__ == "__main__":
    main()
