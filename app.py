import os
import re
import glob
import asyncio
import random
import requests
import feedparser
import pandas as pd
import gpt_researcher.actions.agent_creator as agent_creator
import streamlit as st

from dotenv import load_dotenv
from tqdm import tqdm
from bs4 import BeautifulSoup

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from gpt_researcher import GPTResearcher
import os
os.environ["REPORT_SOURCE"] = "local"
# === Patch regex crash ===
original = agent_creator.extract_json_with_regex
def safe_extract_json_with_regex(response):
    if not response:
        return None
    return original(response)
agent_creator.extract_json_with_regex = safe_extract_json_with_regex

# === Env ===
load_dotenv()
st.set_page_config(page_title="Hybrid Research — FAISS + Web", page_icon="🧠")

st.markdown("""
<style>
[data-testid="stSidebar"] { display: none !important; }
.block-container { padding-top: 2rem; }
#log-box { height: 300px; overflow-y: scroll; background: #f9f9f9;
 border: 1px solid #ddd; padding: 1rem; font-size: 0.8rem; white-space: pre-wrap; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings()

@st.cache_resource
def get_vectorstore():
    embeddings = get_embeddings()
    vs = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vs

@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)

# === Make sure uploads folder exists ===
UPLOADS_DIR = "./uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ✅ Debug: Show files to prove files exist
st.write("**Uploads folder files:**", glob.glob(f"{UPLOADS_DIR}/**/*", recursive=True))

# If folder is empty, add dummy.txt to guarantee no error
files = glob.glob(f"{UPLOADS_DIR}/**/*", recursive=True)
if not files:
    with open(f"{UPLOADS_DIR}/dummy.txt", "w") as f:
        f.write("placeholder text")

# === Answer using vector ===
def get_chunks(q, vs):
    return vs.similarity_search(q, k=8)

def get_retriever_chain(vs):
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 50})
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        ("user", "Generate a precise search query.")
    ])
    return create_history_aware_retriever(get_llm(), retriever, prompt)

def get_rag_chain(chain):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Use ONLY context below. No hallucination.\nContext: {context}"),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ])
    return create_retrieval_chain(chain, create_stuff_documents_chain(get_llm(), prompt))

def get_answer(q, vs):
    docs = get_chunks(q, vs)
    if not docs:
        return "❌ No data found."
    merged_context = "\n\n".join([d.page_content for d in docs])
    chain = get_retriever_chain(vs)
    rag = get_rag_chain(chain)
    result = rag.invoke({
        "chat_history": st.session_state.chat_history,
        "input": q,
        "context": merged_context
    })
    return result["answer"]

# === Hybrid Research ===
async def run_gpt_researcher_hybrid(topic, vs):
    log_box = st.empty()
    chart_box = st.empty()
    logs, metrics = [], []

    def capture_log(*args, **kwargs):
        line = " ".join(str(a) for a in args)
        logs.append(line)
        if "Insight:" in line:
            metrics.append({"Step": len(metrics)+1, "Score": random.randint(1, 10)})
        log_box.markdown(f"<div id='log-box'>{''.join(logs[-50:])}</div>", unsafe_allow_html=True)
        if metrics:
            df = pd.DataFrame(metrics)
            chart_box.line_chart(df, x="Step", y="Score")

    researcher = GPTResearcher(
        query=query,
        report_type="research_report",
        report_source="langchain_vectorstore",
        vector_store=vector_store,
    )
    researcher.print = capture_log

    await researcher.conduct_research()
    return await researcher.write_report()

# === UI ===
vs = get_vectorstore()
st.title("🧠 Retail Hybrid Research — FAISS + Web")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="💡 Ready. Ask me or run deep research.")]

q = st.chat_input("Quick question?")
if q:
    a = get_answer(q, vs)
    st.session_state.chat_history.extend([HumanMessage(content=q), AIMessage(content=a)])

for msg in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(msg, AIMessage) else "Human"):
        st.markdown(msg.content)

# === Hybrid Research Block ===
st.divider()
st.header("🔍 Deep Research — FAISS + Web")
topic = st.text_input("Topic for Deep Research:")

if st.button("🚀 Run Hybrid Research Now"):
    if not topic.strip():
        st.warning("❗ Please enter a topic.")
    else:
        with st.spinner("⏳ Running hybrid research..."):
            report = asyncio.run(run_gpt_researcher_hybrid(topic, vs))
            st.download_button("📄 Download Report", report, file_name="DeepResearchReport.md")
            st.write(report)
