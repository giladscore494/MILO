import hashlib
import streamlit as st

key = st.secrets["GEMINI_API_KEY"]
fp = "sha256:" + hashlib.sha256(key.encode()).hexdigest()[:16]

st.write("fingerprint:", fp)
st.write("length:", len(key))
st.write("starts with:", key[:6])
st.write("ends with:", key[-4:])
