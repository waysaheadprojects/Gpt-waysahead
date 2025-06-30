import os
import re
import asyncio
import random
import requests
import feedparser
import gpt_researcher.actions.agent_creator as agent_creator
import streamlit as st
import pandas as pd

from dotenv import load_dotenv
from bs4 import BeautifulSoup
from tqdm import tqdm

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from gpt_researcher import GPTResearcher

# Fix regex bug
original = agent_creator.extract_json_with_regex
def safe_extract_json_with_regex(response):
    if not response:
        return None
    return original(response)
agent_creator.extract_json_with_regex = safe_extract_json_with_regex

# === ENV ===
load_dotenv()
st.set_page_config(page_title="Retail Hybrid Research", page_icon="🧠")

st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        .block-container { padding-top: 2rem; }
        #log-box { height: 300px; overflow-y: scroll; background: #f9f9f9; 
                   border: 1px solid #ddd; padding: 1rem; font-size: 0.8rem; 
                   white-space: pre-wrap; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings()

# === Load your existing FAISS ===
@st.cache_resource
def get_vectorstore():
    embeddings = get_embeddings()
    vs = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vs

# === Regular answer using FAISS ===
def get_best_chunks(q, vs):
    docs = vs.similarity_search(q, k=8)
    return docs

@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)

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
        ("system", """Use ONLY the context below. 
Do not hallucinate. Use tables if needed. 
If no info, say: "Not found in docs."
Context: {context}"""),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ])
    return create_retrieval_chain(chain, create_stuff_documents_chain(get_llm(), prompt))

def get_answer(q, vs):
    docs = get_best_chunks(q, vs)
    if not docs:
        return "❌ Not found in docs."
    merged_context = "\n\n".join([d.page_content for d in docs])
    chain = get_retriever_chain(vs)
    rag = get_rag_chain(chain)
    result = rag.invoke({
        "chat_history": st.session_state.chat_history,
        "input": q,
        "context": merged_context
    })
    return result["answer"]

# === Deep Research Hybrid ===

async def run_gpt_researcher_hybrid(topic, vs):
    log_box = st.empty()
    chart_box = st.empty()
    logs = []
    metrics = []

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
        query=topic,
        report_type="research_report",
        report_source="hybrid",
        vector_store=vs,
        doc_path="./uploads"  # ✅ Use your existing uploads folder!
    )
    researcher.print = capture_log

    await researcher.conduct_research()
    return await researcher.write_report()


# === UI ===
vs = get_vectorstore()
st.title("🧠 Retail Research Assistant — FAISS + Web Hybrid")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="💡 Welcome! Ask anything about retail, or run deep research.")]

q = st.chat_input("Quick Question?")
if q:
    a = get_answer(q, vs)
    st.session_state.chat_history.extend([HumanMessage(content=q), AIMessage(content=a)])

for msg in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(msg, AIMessage) else "Human"):
        st.markdown(msg.content)

# === Deep Research ===
st.divider()
st.header("🔍 Full Deep Research")
topic = st.text_input("Topic for Deep Research:")

if st.button("🚀 Run Deep Research (FAISS + Web)"):
    if not topic or topic.strip() == "":
        st.warning("❗ Enter a topic first.")
    else:
        with st.spinner("⏳ Running deep hybrid research..."):
            report = asyncio.run(run_gpt_researcher_hybrid(topic, vs))
            st.download_button("📄 Download Deep Report", report, file_name="DeepResearchReport.md")
            st.write(report)
