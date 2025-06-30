import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
from gpt_researcher import GPTResearcher

# === Load API key ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ OPENAI_API_KEY not found in .env. Please set it!")
    st.stop()

# === Streamlit Page Config ===
st.set_page_config(page_title="Deep Research Assistant", page_icon="🧠")

st.title("🧠 Deep Research Assistant with GPTResearcher")
st.info("Ask a deep question and generate a **live research report** with real-time logs!")

# === Input ===
topic = st.text_input("🔍 Enter your research topic:", value="")

# === Deep Research Function ===
async def run_gpt_researcher_async(topic: str):
    placeholder = st.empty()
    logs = []

    def stream_log(*args, **kwargs):
        line = " ".join(str(a) for a in args)
        logs.append(line)
        # Show last 30 lines
        placeholder.text("\n".join(logs[-30:]))

    researcher = GPTResearcher(
        query=topic,
        report_source="hybrid",
        verbosity="high"
    )
    researcher.print = stream_log

    try:
        await researcher.conduct_research()
        report = await researcher.write_report()
    except Exception as e:
        st.error(f"❌ Research failed: {e}")
        return "⚠️ Research failed."

    return report

# === Run Button ===
if st.button("🚀 Run Deep Research"):
    if not topic.strip():
        st.warning("❗ Please enter a topic first!")
    else:
        with st.spinner("🧠 Running deep research... please wait!"):
            report = asyncio.run(run_gpt_researcher_async(topic))
            st.success("✅ Research complete!")
            st.download_button("📄 Download Report", report, file_name="DeepResearchReport.md")
            st.write(report)
