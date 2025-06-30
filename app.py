import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
from gpt_researcher import GPTResearcher

import re
import gpt_researcher.actions.agent_creator as agent_creator

# === Patch regex to avoid NoneType crash ===
original = agent_creator.extract_json_with_regex

def safe_extract_json_with_regex(response):
    if not response:
        return None
    return original(response)

agent_creator.extract_json_with_regex = safe_extract_json_with_regex

# === Load API Key ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ OPENAI_API_KEY not found in .env. Please set it!")
    st.stop()

st.set_page_config(page_title="Deep Research Assistant", page_icon="🧠")
st.title("🧠 Deep Research Assistant — GPTResearcher")
st.info("Run deep research with live logs & fallback if no docs load!")

topic = st.text_input("Enter your research topic:", value="Latest trends in AI regulation worldwide")

async def run_gpt_researcher_async(topic: str):
    placeholder = st.empty()
    logs = []

    def stream_log(*args, **kwargs):
        line = " ".join(str(a) for a in args)
        logs.append(line)
        placeholder.text("\n".join(logs[-30:]))

    researcher = GPTResearcher(
        query=topic,
        report_source="web",  # use web to force scraping
        verbosity="high"
    )
    researcher.print = stream_log

    try:
        await researcher.conduct_research()
        report = await researcher.write_report()
    except Exception as e:
        st.error(f"❌ Research failed: {e}")
        return "⚠️ Research failed."

    if not report or report.strip() == "":
        return "⚠️ No report generated — maybe no sources found!"

    return report

if st.button("🚀 Run Deep Research"):
    if not topic.strip():
        st.warning("Please enter a topic first!")
    else:
        with st.spinner("Running deep research..."):
            report = asyncio.run(run_gpt_researcher_async(topic))
            st.download_button("📄 Download Report", report, file_name="DeepResearchReport.md")
            st.write(report)
