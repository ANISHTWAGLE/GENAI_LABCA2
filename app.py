"""HeatDesk - ask IMD temperature data in plain English.   Run:  streamlit run app.py"""
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from heatdesk_ui import LiveTimeline, active_dataset, get_llm, render_code_tab
from reflexion import dataset
from reflexion.ask import answer_question
from reflexion.llm import LLMUnavailable
from reflexion.loop import LoopConfig
from reflexion.sandbox import SandboxLimits

load_dotenv(Path(__file__).resolve().parent / ".env")
st.set_page_config(page_title="HeatDesk", page_icon=":material/thermostat:", layout="wide",
                   initial_sidebar_state="collapsed")

EXAMPLES = [
    "Which 5 places had the most heatwave days in May 2024?",
    "What was the hottest day recorded near Nagpur, and how hot was it?",
    "How many days above 45 degC did the area near Delhi have each year?",
]
CONFIDENCE = {"HIGH": ("green", ":material/verified:", "Checked twice - both answers agree."),
              "LOW": ("orange", ":material/warning:", "Checked twice - the answers differ. See the note below."),
              "UNVERIFIED": ("blue", ":material/help:", "Not double-checked."),
              "FAILED": ("red", ":material/error:", "No answer could be produced.")}

with st.sidebar:
    st.header("Settings")
    names = [d.name for d in dataset.DATASETS.values() if d.available]
    st.radio("Data", names, key="data_name", format_func=lambda n: dataset.DATASETS[n].description)
    st.selectbox("AI model", ["auto", "groq", "ollama"], key="provider",
                 format_func={"auto": "Groq (cloud), fall back to local", "groq": "Groq (cloud) only",
                              "ollama": "Local qwen2.5-coder:3b only"}.get)
    st.checkbox("Double-check every answer", value=True, key="verify",
                help="Solves the question twice, independently, and compares.")
    st.slider("Max attempts to fix its code", 1, 8, 5, key="max_attempts")

st.title("HeatDesk", icon=":material/thermostat:")
st.write("Ask a question about India's daily maximum temperatures (IMD data, 2015-2024). "
         "The AI writes a small program to find the answer, and fixes the program itself if it fails.")

with st.container(horizontal=True):
    for i, ex in enumerate(EXAMPLES):
        if st.button(ex, key=f"ex_{i}", width="content"):
            st.session_state["question"] = ex
with st.form("ask_form", border=False):
    question = st.text_area("Your question", key="question", height=80,
                            placeholder="e.g. Which 5 places had the most heatwave days in May 2024?")
    submitted = st.form_submit_button("Ask", type="primary", icon=":material/send:")

if submitted and (question or "").strip():
    ds = active_dataset()
    try:
        llm = get_llm(st.session_state.get("provider", "auto"))
    except LLMUnavailable as e:
        st.error(f"No AI model available: {e}", icon=":material/cloud_off:")
        st.stop()
    cfg = LoopConfig(max_attempts=st.session_state.get("max_attempts", 5),
                     time_budget_s=1800.0 if llm.current.name == "ollama" else 600.0,
                     limits=SandboxLimits(timeout_s=60, memory_mb=ds.memory_mb))
    with st.status(":shimmer[Working on it...]", expanded=True) as status:
        timeline = LiveTimeline()
        llm.on_event = timeline.provider_event
        res = answer_question(question, llm, cfg, verify=st.session_state.get("verify", True), on_event=timeline)
        timeline._close()
        status.update(label="Done" if res.primary.success else "Could not answer",
                      state="complete" if res.primary.success else "error", expanded=False)
    st.session_state.setdefault("history", []).insert(0, res)

for i, res in enumerate(st.session_state.get("history", [])[:5]):
    with st.container(border=True):
        st.subheader(res.question)
        if res.answer is None:
            st.error(f"Sorry, no answer: {res.primary.reason}", icon=":material/error:")
        else:
            colour, icon, text = CONFIDENCE[res.confidence]
            st.badge(f"Confidence: {res.confidence} - {text}", icon=icon, color=colour)
            st.info(res.answer.get("answer", ""), icon=":material/lightbulb:")
            left, right = st.columns(2)
            with left:
                result = res.answer.get("result")
                if isinstance(result, list):
                    st.dataframe(pd.DataFrame(result).rename(columns={"value": res.answer.get("unit") or "value"}),
                                 hide_index=True)
                else:
                    st.metric("Result", f"{result} {res.answer.get('unit') or ''}")
                st.caption(f"How it was worked out: {res.answer.get('method', '')}")
            with right:
                chart = res.outputs_dir / "chart.png" if res.outputs_dir else None
                if res.answer.get("chart") and chart and chart.exists():
                    st.image(str(chart))
            if res.confidence == "LOW" and res.check_answer:
                st.warning(f"The double-check said: {res.check_answer.get('answer', '')}",
                           icon=":material/compare_arrows:")
        tries = len(res.primary.attempts)
        with st.expander(f"How the AI got this answer - code, fixes{f' ({tries - 1})' if tries > 1 else ''}, "
                         "checks and safety", icon=":material/code:"):
            render_code_tab(res.primary, key=f"code_{i}")
