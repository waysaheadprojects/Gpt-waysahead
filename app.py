import os
import io
import asyncio
from contextlib import redirect_stdout

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from gpt_researcher import GPTResearcher

# === Streamlit settings ===
st.set_page_config(page_title="Retail Research + Dynamic Charts", page_icon="📊")

st.title("🧠 Deep Retail Research — Dynamic Report + Auto Charts")

# === Input ===
query = st.text_input(
    "Enter your big research question:",
    "Generate a detailed report on the Dubai retail sector with a table of malls and annual visitors."
)

# === Run GPT Researcher and stream logs ===
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

    return report

# === Chart renderer ===
def render_charts_from_markdown(report_text):
    """
    Look for markdown tables in the report.
    For each table, parse it and plot as a bar chart.
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
            # Table ended
            if len(temp_table) > 2:
                tables.append(temp_table)
            temp_table = []
            inside_table = False

    # Parse each table to pandas
    for idx, table in enumerate(tables):
        headers = [h.strip() for h in table[0].split("|")[1:-1]]
        rows = []
        for row in table[2:]:  # skip header + separator
            cols = [c.strip() for c in row.split("|")[1:-1]]
            rows.append(cols)
        df = pd.DataFrame(rows, columns=headers)

        # Try numeric conversion
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")

        st.write(f"📊 **Auto-Detected Table {idx+1}**")
        st.dataframe(df)

        # Only plot if numeric
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) >= 1:
            fig, ax = plt.subplots()
            df.plot(kind="bar", x=df.columns[0], y=numeric_cols[0], ax=ax)
            st.pyplot(fig)

# === Button to trigger ===
if st.button("🔍 Run Deep Research & Show Charts"):
    with st.spinner("Running GPT-Researcher with live logs..."):
        final_report = asyncio.run(run_researcher_and_capture(query))

    st.subheader("✅ **Final Research Report**")
    st.markdown(final_report)

    # Try to find any markdown table and chart it
    render_charts_from_markdown(final_report)

    # Offer download
    st.download_button(
        "📥 Download Report",
        data=final_report,
        file_name="deep_research_report.md"
    )
