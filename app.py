import os
import asyncio
from dotenv import load_dotenv

import streamlit as st
from pydantic import BaseModel
from typing import Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.tools import web  # use web search tool
from langgraph.graph import StateGraph, END
from gpt_researcher import GPTResearcher
import gpt_researcher.actions.agent_creator as agent_creator

# === Patch JSON regex bug for GPTResearcher ===
original = agent_creator.extract_json_with_regex
def safe_extract_json_with_regex(response):
    if not response:
        return None
    return original(response)
agent_creator.extract_json_with_regex = safe_extract_json_with_regex

load_dotenv()
os.environ["REPORT_SOURCE"] = "local"

# Initialize LLM and vector store
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.2)
embeddings = OpenAIEmbeddings()
vs = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)

# Build retriever and RAG chain
def get_retriever_chain():
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 8})
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        ("user", "Generate a precise search query.")
    ])
    return create_history_aware_retriever(llm, retriever, prompt)

def get_rag_chain(chain):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a retail research assistant.
Use ONLY the provided context. If none, reply ❌.
Include bullet points, tables, or chart ideas if helpful.
Context: {context}
        """),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ])
    return create_retrieval_chain(chain, create_stuff_documents_chain(llm, prompt))

# === Define tools with docstrings ===

async def vector_lookup(query: str) -> str:
    """
    Search the local FAISS vector store for relevant context.
    Returns a markdown-formatted answer or ❌ if nothing is found.
    """
    docs = vs.similarity_search(query, k=5)
    if not docs:
        return "❌ No vector answer."
    context = "\n\n".join([d.page_content for d in docs])
    result = get_rag_chain(get_retriever_chain()).invoke({
        "chat_history": [], "input": query, "context": context
    })
    return result["answer"]

async def chitchat_tool(query: str) -> str:
    """
    Provide a brief, warm response for casual conversation or greetings.
    """
    prompt = f'User said: "{query}". Reply kindly in one or two sentences.'
    return (await llm.ainvoke(prompt)).content.strip()

async def web_search_tool(query: str) -> str:
    """
    Perform a live web search using web.run.
    Returns the headline/title of the top result or a "no results" message.
    """
    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(
        None,
        lambda: web.run({
            "search_query": [{"q": query, "recency": 1, "domains": None}]
        })
    )
    hits = resp.get("search_query", [])
    if not hits:
        return "🌐 No web results found."
    return f"🌐 Top result: {hits[0]['q']}"

async def run_gpt_researcher(query: str) -> str:
    """
    Execute a deep hybrid research with GPTResearcher.
    Combines local FAISS and web sources. Logs progress to research_logs.txt.
    """
    log_file = "./research_logs.txt"
    def capture_log(*args, **kwargs):
        with open(log_file, "a") as f:
            f.write(" ".join(str(a) for a in args) + "\n")

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

# Wrap them as StructuredTools
vector_tool = StructuredTool.from_function(vector_lookup)
chitchat = StructuredTool.from_function(chitchat_tool)
web_tool = StructuredTool.from_function(web_search_tool)
deep_tool = StructuredTool.from_function(run_gpt_researcher)

# === LangGraph State & Graph ===
class State(BaseModel):
    query: str
    route: Optional[str] = None
    answer: Optional[str] = None

async def router(state: State):
    """
    Use LLM to classify user input:
    - 'vector' for FAISS research queries
    - 'web' for live info
    - 'chitchat' for greetings or casual
    """
    res = await llm.ainvoke(f'''
Classify this input: "{state.query}"
Return exactly one of: vector, web, chitchat
''')
    return {"route": res.content.strip().lower()}

# Node handlers
async def vector_node(state: State):
    return {"answer": await vector_tool.ainvoke({"query": state.query})}

async def web_node(state: State):
    return {"answer": await web_tool.ainvoke({"query": state.query})}

async def chitchat_node(state: State):
    return {"answer": await chitchat.ainvoke({"query": state.query})}

async def deep_node(state: State):
    return {"answer": await deep_tool.ainvoke({"query": state.query})}

# Build graph
graph = StateGraph(State)
graph.add_node("router", router)
graph.add_node("vector", vector_node)
graph.add_node("web", web_node)
graph.add_node("chitchat", chitchat_node)
graph.add_node("deep", deep_node)

graph.set_entry_point("router")
graph.add_conditional_edges("router", lambda s: s.route, {
    "vector": "vector",
    "web": "web",
    "chitchat": "chitchat"
})
graph.add_edge("vector", END)
graph.add_edge("web", END)
graph.add_edge("chitchat", END)
graph.add_edge("deep", END)

agent = graph.compile()

# === Streamlit Interface ===
st.set_page_config(page_title="Retail Hybrid Agent", page_icon="🧠")
st.title("🧠 Retail Research — Hybrid LangGraph Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input handler
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.last_query = prompt
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        res = asyncio.run(agent.ainvoke(State(query=prompt)))
    answer = res["answer"]

    if answer.startswith("❌"):
        st.session_state.messages.append({
            "role": "assistant",
            "content": "I couldn't find enough locally. Click below for Deep Research!"
        })
    else:
        st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

# Fallback deep research option
if st.session_state.messages and "Deep Research" in st.session_state.messages[-1]["content"]:
    if st.button("🔍 Run Deep Research for this"):
        with st.spinner("Running Deep Research..."):
            report = asyncio.run(deep_tool.ainvoke({"query": st.session_state.last_query}))
            st.session_state.messages.append({"role": "assistant", "content": report})
            st.rerun()

# Manual deep research
st.divider()
with st.expander("💡 Manual Deep Research"):
    manual_q = st.text_input("Your topic:")
    if st.button("Run Manual Research"):
        if manual_q.strip():
            with st.spinner("Running Deep Research..."):
                report = asyncio.run(deep_tool.ainvoke({"query": manual_q}))
                st.session_state.messages.append({"role": "user", "content": f"Manual topic: {manual_q}"})
                st.session_state.messages.append({"role": "assistant", "content": report})
                st.rerun()
        else:
            st.warning("Please enter a topic.")
