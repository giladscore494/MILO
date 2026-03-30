import ast
import json
import math
import os
import re
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

APP_DIR = Path(".calibration_workspace")
APP_DIR.mkdir(exist_ok=True)
SOURCE_PATH = APP_DIR / "source_scoring_baseline.py"
PROGRESS_PATH = APP_DIR / "calibration_progress.json"
OUTPUT_PATH = APP_DIR / "scoring_baseline_calibrated.py"
LOG_PATH = APP_DIR / "run_log.jsonl"

REQUIRED_FIELDS = [
    "reliability_bias",
    "recall_penalty_sensitivity",
    "maintenance_penalty_sensitivity",
    "systemic_penalty_sensitivity",
    "soft_floor_if_no_major_systemic",
]

DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_BATCH_SIZE = 10


class VehicleCalibration(BaseModel):
    make: str = Field(description="Make key exactly as provided in the input list.")
    model: str = Field(description="Model key exactly as provided in the input list.")
    reliability_bias: str = Field(description="One of: strong, neutral, weak.")
    recall_penalty_sensitivity: str = Field(description="One of: low, normal, high.")
    maintenance_penalty_sensitivity: str = Field(description="One of: low, normal, high.")
    systemic_penalty_sensitivity: str = Field(description="One of: low, normal, high.")
    soft_floor_if_no_major_systemic: int = Field(description="Integer floor from 0 to 100. Use 0 only if no useful floor should exist.")
    rationale_he: str = Field(description="Short Hebrew rationale, max 30 words.")


class BatchCalibrationResponse(BaseModel):
    vehicles: List[VehicleCalibration]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_make_key(make: str) -> str:
    return str(make).strip().lower()


def normalize_model_key(model: str) -> str:
    return str(model).strip().lower()


def append_log(event: Dict[str, Any]) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_assignment(source: str, variable_name: str) -> Any:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == variable_name:
                    return ast.literal_eval(node.value)
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == variable_name:
                return ast.literal_eval(node.value)
    raise ValueError(f"Could not find assignment for {variable_name}")


def load_scoring_baseline(source_text: str) -> Dict[str, Any]:
    return {
        "MAKE_PROFILES": extract_assignment(source_text, "MAKE_PROFILES"),
        "MAKE_DEFAULT": extract_assignment(source_text, "MAKE_DEFAULT"),
        "MODEL_OVERRIDES": extract_assignment(source_text, "MODEL_OVERRIDES"),
    }


def dump_scoring_baseline_module(data: Dict[str, Any]) -> str:
    make_profiles = pformat(data["MAKE_PROFILES"], width=140, sort_dicts=False)
    make_default = pformat(data["MAKE_DEFAULT"], width=140, sort_dicts=False)
    model_overrides = pformat(data["MODEL_OVERRIDES"], width=140, sort_dicts=False)

    return textwrap.dedent(
        f'''\
        # -*- coding: utf-8 -*-
        """
        Auto-generated calibration version of scoring_baseline.py.
        This file was rewritten by the Streamlit calibration utility.
        """

        from typing import Any, Dict, Optional, Tuple

        MAKE_PROFILES: Dict[str, Dict[str, Any]] = {make_profiles}

        MAKE_DEFAULT: Dict[str, Any] = {make_default}

        MODEL_OVERRIDES: Dict[str, Dict[str, Dict[str, Any]]] = {model_overrides}


        def normalize_make(make: str) -> str:
            return str(make or "").strip().lower()


        def normalize_model(model: str) -> str:
            return str(model or "").strip().lower()


        def get_make_profile(make: str) -> Dict[str, Any]:
            return MAKE_PROFILES.get(normalize_make(make), MAKE_DEFAULT)


        def get_model_override(make: str, model: str) -> Optional[Dict[str, Any]]:
            make_key = normalize_make(make)
            model_key = normalize_model(model)
            return MODEL_OVERRIDES.get(make_key, {{}}).get(model_key)


        def get_combined_score_modifier(make: str, model: str) -> Tuple[int, float, Optional[str]]:
            make_profile = get_make_profile(make)
            model_override = get_model_override(make, model) or {{}}
            total_modifier = int(make_profile.get("base_modifier", 0)) + int(model_override.get("model_modifier", 0))
            confidence_boost = float(model_override.get("confidence_boost", 0.0))
            transmission_default = model_override.get("transmission_default")
            return total_modifier, confidence_boost, transmission_default
        '''
    )


def get_vehicle_rows(model_overrides: Dict[str, Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for make_key, models in model_overrides.items():
        for model_key, payload in models.items():
            rows.append(
                {
                    "make": make_key,
                    "model": model_key,
                    "payload": payload,
                }
            )
    rows.sort(key=lambda x: (x["make"], x["model"]))
    return rows


def has_required_fields(payload: Dict[str, Any]) -> bool:
    return all(field in payload for field in REQUIRED_FIELDS)


def load_progress() -> Dict[str, Any]:
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    return {
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "source_file": str(SOURCE_PATH),
        "completed": [],
        "failed": [],
        "last_batch": None,
        "batch_history": [],
    }


def save_progress(progress: Dict[str, Any]) -> None:
    progress["updated_at"] = utc_now_iso()
    PROGRESS_PATH.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def completed_key(make: str, model: str) -> str:
    return f"{make}|||{model}"


def get_remaining_rows(model_overrides: Dict[str, Dict[str, Dict[str, Any]]], progress: Dict[str, Any]) -> List[Dict[str, Any]]:
    completed = set(progress.get("completed", []))
    rows = []
    for row in get_vehicle_rows(model_overrides):
        key = completed_key(row["make"], row["model"])
        if has_required_fields(row["payload"]):
            continue
        if key in completed:
            continue
        rows.append(row)
    return rows


def build_prompt(batch_rows: List[Dict[str, Any]]) -> str:
    items = []
    for idx, row in enumerate(batch_rows, start=1):
        payload = row["payload"]
        compact_payload = {
            "existing_model_modifier": payload.get("model_modifier"),
            "known_issues": payload.get("known_issues", []),
            "transmission_default": payload.get("transmission_default"),
        }
        items.append(
            f"{idx}. make={row['make']} | model={row['model']} | existing={json.dumps(compact_payload, ensure_ascii=False)}"
        )

    return textwrap.dedent(
        f"""
        You are enriching a deterministic vehicle reliability calibration dictionary for the Israeli used-car market.

        Mandatory behavior:
        1. You MUST use Google Search grounding for every vehicle in this batch.
        2. Search the public web before deciding any field.
        3. Prefer broad long-term reliability evidence over isolated anecdotes.
        4. Distinguish chronic reliability weakness from campaign/recall verification noise.
        5. Return JSON only, matching the provided schema exactly.
        6. Keep make and model exactly as provided.
        7. Use conservative calibration values; do not overreact.

        Field meanings:
        - reliability_bias: strong | neutral | weak
        - recall_penalty_sensitivity: low | normal | high
        - maintenance_penalty_sensitivity: low | normal | high
        - systemic_penalty_sensitivity: low | normal | high
        - soft_floor_if_no_major_systemic: integer 0-100. Use a gentle floor only for models that are usually strong when there is no major chronic systemic signal. Otherwise use 0.
        - rationale_he: short Hebrew explanation up to 30 words.

        Important scoring intent:
        - low recall_penalty_sensitivity means recall/campaign noise should not drag the model too much.
        - high systemic_penalty_sensitivity means true chronic mechanical/electrical patterns should matter more.
        - soft_floor_if_no_major_systemic must stay gentle and realistic.

        Vehicles to enrich:
        {chr(10).join(items)}
        """
    ).strip()


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
                sources.append(
                    {
                        "title": getattr(web, "title", None),
                        "uri": getattr(web, "uri", None),
                    }
                )
        return {"web_search_queries": queries, "sources": sources}
    except Exception:
        return {"web_search_queries": [], "sources": []}


def call_gemini_batch(client: genai.Client, model_name: str, batch_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    prompt = build_prompt(batch_rows)
    config = types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
        response_json_schema=BatchCalibrationResponse.model_json_schema(),
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config,
    )
    parsed = BatchCalibrationResponse.model_validate_json(response.text)
    debug = extract_grounding_debug(response)
    return [item.model_dump() for item in parsed.vehicles], debug, prompt


def merge_batch_into_model_overrides(
    model_overrides: Dict[str, Dict[str, Dict[str, Any]]],
    batch_results: List[Dict[str, Any]],
) -> List[str]:
    merged_keys: List[str] = []
    for item in batch_results:
        make_key = normalize_make_key(item["make"])
        model_key = normalize_model_key(item["model"])
        if make_key not in model_overrides or model_key not in model_overrides[make_key]:
            continue
        target = model_overrides[make_key][model_key]
        for field in REQUIRED_FIELDS:
            target[field] = item[field]
        merged_keys.append(completed_key(make_key, model_key))
    return merged_keys


def prepare_source_text(uploaded_file) -> Optional[str]:
    if uploaded_file is not None:
        source_text = uploaded_file.getvalue().decode("utf-8")
        save_text(SOURCE_PATH, source_text)
        return source_text
    if SOURCE_PATH.exists():
        return load_text(SOURCE_PATH)
    return None


def reset_workspace() -> None:
    for path in [SOURCE_PATH, PROGRESS_PATH, OUTPUT_PATH, LOG_PATH]:
        if path.exists():
            path.unlink()


def run_batches(
    source_text: str,
    model_name: str,
    batch_size: int,
    max_batches: Optional[int],
    request_pause_seconds: float,
    status_box,
    progress_bar,
    live_placeholder,
) -> None:
    baseline = load_scoring_baseline(source_text)
    progress = load_progress()
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

    remaining = get_remaining_rows(baseline["MODEL_OVERRIDES"], progress)
    total_target = len(get_vehicle_rows(baseline["MODEL_OVERRIDES"]))
    processed_before = total_target - len(remaining)
    batches_run = 0

    while remaining:
        if max_batches is not None and batches_run >= max_batches:
            break

        batch_rows = remaining[:batch_size]
        status_box.info(
            f"מעבד אצווה {batches_run + 1} | רכבים באצווה: {len(batch_rows)} | נשארו {len(remaining)} רכבים בלי כיול מלא"
        )
        live_placeholder.code(
            "\n".join([f"{r['make']} / {r['model']}" for r in batch_rows]), language="text")

        batch_started = utc_now_iso()
        try:
            batch_results, grounding_debug, prompt = call_gemini_batch(client, model_name, batch_rows)
            merged_keys = merge_batch_into_model_overrides(baseline["MODEL_OVERRIDES"], batch_results)

            progress["completed"] = sorted(set(progress.get("completed", [])).union(merged_keys))
            progress["last_batch"] = {
                "started_at": batch_started,
                "finished_at": utc_now_iso(),
                "requested": [completed_key(r["make"], r["model"]) for r in batch_rows],
                "merged": merged_keys,
                "grounding": grounding_debug,
                "result_count": len(batch_results),
            }
            progress.setdefault("batch_history", []).append(progress["last_batch"])
            save_progress(progress)

            output_text = dump_scoring_baseline_module(baseline)
            save_text(OUTPUT_PATH, output_text)

            append_log(
                {
                    "ts": utc_now_iso(),
                    "event": "batch_success",
                    "requested": [completed_key(r["make"], r["model"]) for r in batch_rows],
                    "merged": merged_keys,
                    "queries": grounding_debug.get("web_search_queries", []),
                }
            )

        except Exception as e:
            fail_record = {
                "ts": utc_now_iso(),
                "requested": [completed_key(r["make"], r["model"]) for r in batch_rows],
                "error": str(e),
            }
            progress.setdefault("failed", []).append(fail_record)
            save_progress(progress)
            append_log({"event": "batch_error", **fail_record})
            status_box.error(f"שגיאה באצווה הנוכחית: {e}")
            break

        batches_run += 1
        remaining = get_remaining_rows(baseline["MODEL_OVERRIDES"], progress)
        processed_now = total_target - len(remaining)
        ratio = processed_now / max(total_target, 1)
        progress_bar.progress(min(max(ratio, 0.0), 1.0), text=f"{processed_now}/{total_target} רכבים עם כיול מלא")
        status_box.success(f"האצווה נשמרה. הושלמו עד עכשיו {processed_now} מתוך {total_target}.")

        if request_pause_seconds > 0 and remaining:
            time.sleep(request_pause_seconds)

    if not remaining:
        status_box.success("העיבוד הושלם לכל הרכבים שדורשים כיול.")


def render_summary(source_text: str) -> None:
    baseline = load_scoring_baseline(source_text)
    progress = load_progress()
    rows = get_vehicle_rows(baseline["MODEL_OVERRIDES"])
    completed_in_file = sum(1 for row in rows if has_required_fields(row["payload"]))
    completed_marked = len(set(progress.get("completed", [])))
    total = len(rows)
    remaining = len(get_remaining_rows(baseline["MODEL_OVERRIDES"], progress))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("סה״כ דגמים", total)
    c2.metric("מלאים בקובץ", completed_in_file)
    c3.metric("מסומנים כהושלמו", completed_marked)
    c4.metric("נשארו לעיבוד", remaining)

    last_batch = progress.get("last_batch")
    if last_batch:
        st.subheader("האצווה האחרונה")
        st.json(last_batch, expanded=False)


def main() -> None:
    st.set_page_config(page_title="Gemini Dictionary Calibrator", layout="wide")
    st.title("Gemini Grounded Calibration Builder")
    st.caption("מעדכן את MODEL_OVERRIDES בצורה אוטומטית, עם grounding, checkpoint, resume והורדה של הקובץ המלא.")

    with st.sidebar:
        st.header("הגדרות")
        model_name = st.text_input("Model", value=DEFAULT_MODEL)
        batch_size = st.number_input("רכבים בכל בקשה", min_value=1, max_value=20, value=DEFAULT_BATCH_SIZE, step=1)
        pause_seconds = st.number_input("השהיה בין בקשות (שניות)", min_value=0.0, max_value=10.0, value=0.6, step=0.1)
        reset_clicked = st.button("איפוס workspace")
        if reset_clicked:
            reset_workspace()
            st.success("נמחקו source, checkpoint ו-output.")

    if "GEMINI_API_KEY" not in st.secrets:
        st.error("חסר GEMINI_API_KEY ב-secrets של Streamlit.")
        st.stop()

    uploaded_file = st.file_uploader("העלה את scoring_baseline.py הקיים", type=["py"])
    source_text = prepare_source_text(uploaded_file)
    if not source_text:
        st.info("צריך להעלות את scoring_baseline.py לפחות פעם אחת. אחר כך אפשר להמשיך גם בלי להעלות שוב באותה סביבת עבודה.")
        st.stop()

    try:
        render_summary(source_text)
    except Exception as e:
        st.error(f"לא הצלחתי לטעון את הקובץ: {e}")
        st.stop()

    status_box = st.empty()
    progress_bar = st.progress(0.0, text="מוכן להתחלה")
    live_placeholder = st.empty()

    progress = load_progress()
    baseline = load_scoring_baseline(source_text)
    total = len(get_vehicle_rows(baseline["MODEL_OVERRIDES"]))
    remaining = len(get_remaining_rows(baseline["MODEL_OVERRIDES"], progress))
    progress_bar.progress((total - remaining) / max(total, 1), text=f"{total - remaining}/{total} רכבים עם כיול מלא")

    col1, col2, col3 = st.columns(3)
    run_one = col1.button("הרץ אצווה אחת")
    run_all = col2.button("המשך עד הסוף")
    refresh_only = col3.button("רענון מצב")

    if run_one:
        run_batches(
            source_text=source_text,
            model_name=model_name,
            batch_size=int(batch_size),
            max_batches=1,
            request_pause_seconds=float(pause_seconds),
            status_box=status_box,
            progress_bar=progress_bar,
            live_placeholder=live_placeholder,
        )

    if run_all:
        run_batches(
            source_text=source_text,
            model_name=model_name,
            batch_size=int(batch_size),
            max_batches=None,
            request_pause_seconds=float(pause_seconds),
            status_box=status_box,
            progress_bar=progress_bar,
            live_placeholder=live_placeholder,
        )

    if refresh_only:
        st.rerun()

    if OUTPUT_PATH.exists():
        st.download_button(
            "הורד scoring_baseline_calibrated.py",
            data=OUTPUT_PATH.read_bytes(),
            file_name="scoring_baseline_calibrated.py",
            mime="text/x-python",
            use_container_width=True,
        )

    if PROGRESS_PATH.exists():
        st.download_button(
            "הורד checkpoint progress.json",
            data=PROGRESS_PATH.read_bytes(),
            file_name="calibration_progress.json",
            mime="application/json",
            use_container_width=True,
        )

    if LOG_PATH.exists():
        with st.expander("Run log"):
            st.code(LOG_PATH.read_text(encoding="utf-8"), language="json")

    with st.expander("מה האפליקציה עושה"):
        st.markdown(
            """
            - שומרת עותק מקומי של `scoring_baseline.py`.
            - עוברת רק על דגמים שחסרים להם שדות הכיול החדשים.
            - שולחת אצוות ל-Gemini עם Google Search grounding ו-JSON schema.
            - שומרת checkpoint אחרי כל אצווה מוצלחת.
            - כותבת קובץ Python מלא ומעודכן אחרי כל אצווה.
            - אם יש נפילה/timeout, פשוט לוחצים שוב המשך והעיבוד ממשיך מהמקום האחרון שנשמר.
            """
        )


if __name__ == "__main__":
    main()
