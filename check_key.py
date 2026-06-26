import hashlib
import streamlit as st
from google import genai

TARGET_FP = "sha256:55ca0992720d705f"

key = None
for name in st.secrets.keys():
    v = st.secrets[name]
    if isinstance(v, str) and "sha256:" + hashlib.sha256(v.encode()).hexdigest()[:16] == TARGET_FP:
        key = v
        st.write("using secret name:", name)
        break

if not key:
    st.error("key not found by fingerprint")
    st.stop()

st.write("### Test 1 — AI Studio path (api_key only) — what Yeda+MILO use")
try:
    c1 = genai.Client(api_key=key)
    r1 = c1.models.generate_content(model="gemini-3.5-flash", contents="ping")
    st.success("AI Studio OK: " + (r1.text or "")[:120])
except Exception as e:
    st.error(f"AI Studio FAILED: {type(e).__name__}: {str(e)[:400]}")

st.write("### Test 2 — Vertex express path (vertexai=True)")
try:
    c2 = genai.Client(vertexai=True, api_key=key)
    r2 = c2.models.generate_content(model="gemini-3.5-flash", contents="ping")
    st.success("Vertex express OK: " + (r2.text or "")[:120])
except Exception as e:
    st.error(f"Vertex express FAILED: {type(e).__name__}: {str(e)[:400]}")
