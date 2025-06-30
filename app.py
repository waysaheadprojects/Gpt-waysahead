import os
import io
import asyncio
from contextlib import redirect_stdout

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from gpt_researcher import GPTResearcher

# === Load env ===
load_dotenv()

# === Secure check for API key ===
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY is missing. Please set it in your .env file.")
    st.stop()

# === Streamlit UI ===
st.set_page_config(page_title="Deep Retail Research — Live Logs + Dynamic Charts", page_icon="📊")
st.title("🧠 Deep Retail Research — Live Logs + Dynamic Charts")

query = st.text_input(
    "🔍 Enter your big research question:",
    "Generate a detailed report on the Dubai retail sector, with a clear markdown table of malls and annual visitors."
)

# === Run GPT-Researcher with logs ===
async def run_researcher_and_capture(query):
    placeholder = st.empty()
    output_buffer = io.StringIO()

    researcher = GPTResearcher(
        query=query,
        report_source="hybrid",
        verbosity="high"
    )

    def capture_log(*args, **kwargs):
        line = " ".join(str(a) for a in args)
        output_buffer.write(line + "\n")
        placeholder.text(output_buffer.getvalue())

    researcher.print = capture_log

    with redirect_stdout(output_buffer):
        await researcher.conduct_research()
        report = await researcher.write_report()

    if not report:
        report = "❌ Sorry, no report generated. Please check your OpenAI API key or try again."

    return report

# === Chart renderer ===
def render_charts_from_markdown(report_text):
    """
    Dynamically detect Markdown tables in the report,
    convert them to DataFrames, and plot bar charts.
    """
    lines = report_text.splitlines()
    tables = []
    temp_table = []
    inside_table = False

    for line in lines:
        if "|" in line:
            inside_table = True
            temp_table.append(line.strip())
        elif inside_table:
            if len(temp_table) > 2:
                tables.append(temp_table)
            temp_table = []
            inside_table = False

    for idx, table in enumerate(tables):
        headers = [h.strip() for h in table[0].split("|")[1:-1]]
        rows = []
        for row in table[2:]:
            cols = [c.strip() for c in row.split("|")[1:-1]]
            rows.append(cols)
        df = pd.DataFrame(rows, columns=headers)

        # Attempt numeric conversion
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")

        st.write(f"📊 **Auto-Detected Table {idx+1}**")
        st.dataframe(df)

        # Plot if numeric data found
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) >= 1:
            fig, ax = plt.subplots()
            df.plot(kind="bar", x=df.columns[0], y=numeric_cols[0], ax=ax)
            ax.set_ylabel(numeric_cols[0])
            st.pyplot(fig)

# === Run button ===
if st.button("🚀 Run Deep Research Now"):
    with st.spinner("🔍 Running GPT-Researcher with live logs..."):
        final_report = asyncio.run(run_researcher_and_capture(query))

    st.subheader("✅ Final Research Report")
    st.markdown(final_report)

    render_charts_from_markdown(final_report)

    st.download_button(
        "📥 Download Full Report",
        data=final_report,
        file_name="deep_research_report.md"
    )
