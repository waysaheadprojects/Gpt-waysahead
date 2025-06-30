import os
import asyncio
import random
from dotenv import load_dotenv

import streamlit as st

from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from gpt_researcher import GPTResearcher
import gpt_researcher.actions.agent_creator as agent_creator

from langgraph.graph import StateGraph, END

# === Safe regex fix ===
original = agent_creator.extract_json_with_regex
def safe_extract_json_with_regex(response):
    if not response:
        return None
    return original(response)
agent_creator.extract_json_with_regex = safe_extract_json_with_regex

load_dotenv()
os.environ["REPORT_SOURCE"] = "local"

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.3)
embeddings = OpenAIEmbeddings()
vs = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)

# === Retriever chain
def get_retriever_chain():
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        ("user", "Generate a precise search query.")
    ])
    return create_history_aware_retriever(llm, retriever, prompt)

def get_rag_chain(chain):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """
You are a senior research analyst for retail & consumer trends.
Use ONLY the given context. If none, say: ❌ I have no data for that.
Use clear markdown tables, bullet points, and short chart suggestions.
No hallucination.
Context: {context}
         """),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ])
    return create_retrieval_chain(chain, create_stuff_documents_chain(llm, prompt))

# === Tool: Vector lookup
@tool
async def vector_lookup_tool(query: str) -> str:
    """
    Uses the local FAISS vector store to answer a question.
    Returns a formatted short answer or ❌ if not found.
    """
    docs = vs.similarity_search(query, k=5)
    if not docs:
        return "❌ No vector answer."
    merged_context = "\n\n".join([d.page_content for d in docs])
    chain = get_rag_chain(get_retriever_chain())
    result = chain.invoke({
        "chat_history": [],
        "input": query,
        "context": merged_context
    })
    return result["answer"]

# === Tool: Deep Research fallback
@tool
async def deep_research_tool(query: str) -> str:
    """
    Runs GPTResearcher in hybrid mode to create a detailed report using local + web data.
    Also logs progress to the server.
    """
    log_file = "./research_logs.txt"
    def capture_log(*args, **kwargs):
        line = " ".join(str(a) for a in args)
        with open(log_file, "a") as f:
            f.write(line + "\n")

    researcher = GPTResearcher(
        query=query,
        report_type="research_report",
        report_source="hybrid",
        vector_store=vs,
        doc_path="./uploads"
    )
    researcher.print = capture_log
    await researcher.conduct_research()
    return await researcher.write_report()

# === LangGraph nodes
async def node_vector(query):
    return await vector_lookup_tool.ainvoke({"query": query})

async def run_deep(query):
    return await deep_research_tool.ainvoke({"query": query})

# === Streamlit app ===
st.set_page_config(page_title="Retail Hybrid Research", page_icon="🧠")
st.title("🧠 Retail Research — Hybrid Agent")

query = st.text_input("Ask any question:")
run_query = st.button("Run Research")

if "vector_result" not in st.session_state:
    st.session_state.vector_result = ""

if run_query and query:
    with st.spinner("Checking local knowledge..."):
        vector_result = asyncio.run(node_vector(query))
    st.session_state.vector_result = vector_result

    if vector_result.startswith("❌"):
        st.warning("No clear local match found. This may need deeper web + local research.")
        if st.button("Run Deep Research Now"):
            with st.spinner("Running Deep Research..."):
                deep_result = asyncio.run(run_deep(query))
                st.session_state.vector_result = deep_result

if st.session_state.vector_result:
    st.markdown("### Answer:")
    st.markdown(st.session_state.vector_result)
