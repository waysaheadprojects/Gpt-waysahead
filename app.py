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

from langgraph.graph import StateGraph, END

from gpt_researcher import GPTResearcher
import gpt_researcher.actions.agent_creator as agent_creator

from web import run as web_run  # Use web tool

# === PATCH ===
original = agent_creator.extract_json_with_regex
def safe_extract_json_with_regex(response):
    if not response:
        return None
    return original(response)
agent_creator.extract_json_with_regex = safe_extract_json_with_regex

load_dotenv()
os.environ["REPORT_SOURCE"] = "local"

# === LLM and vector store setup ===
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.2)
embeddings = OpenAIEmbeddings()
vs = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)

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
You are a retail research agent.
Use ONLY the provided context. If none, reply ❌.
Use markdown tables, bullet points, chart ideas when helpful.
Context: {context}
        """),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ])
    return create_retrieval_chain(chain, create_stuff_documents_chain(llm, prompt))

# === Tools ===
async def vector_lookup(query: str) -> str:
    """
    Query the FAISS vector store for research content.
    Returns a well-formatted answer or ❌ if nothing relevant.
    """
    docs = vs.similarity_search(query, k=5)
    if not docs:
        return "❌ No vector answer."
    context = "\n\n".join([d.page_content for d in docs])
    chain = get_rag_chain(get_retriever_chain())
    result = chain.invoke({"chat_history": [], "input": query, "context": context})
    return result["answer"]

async def chitchat_tool(query: str) -> str:
    """
    Provide a polite, short response to greetings or casual conversation.
    """
    prompt = f'User asked: "{query}". Reply warmly in 1–2 friendly sentences.'
    return (await llm.ainvoke(prompt)).content.strip()

async def web_search_tool(query: str) -> str:
    """
    Perform a fresh web search and return top result summary.
    """
    # Use the web tool to search
    results = web_run({"search_query":[{"q":query,"recency":1,"domains":None}]})
    hits = results.get("search_query", [])
    if not hits:
        return "🌐 No web results found."
    return "🌐 Top result: " + hits[0]["q"]

async def run_gpt_researcher(query: str) -> str:
    """
    Run a deep hybrid research using GPTResearcher (local + web).
    Logs each step into 'research_logs.txt'.
    """
    log_file = "./research_logs.txt"
    def capture_log(*args, **kwargs):
        with open(log_file, "a") as f:
            f.write(" ".join(str(a) for a in args) + "\n")
    researcher = GPTResearcher(
        query=query, report_type="research_report",
        report_source="hybrid", vector_store=vs, doc_path="./uploads"
    )
    researcher.print = capture_log
    await researcher.conduct_research()
    return await researcher.write_report()

# === Wrap as tools ===
vector_tool = StructuredTool.from_function(vector_lookup)
chitchat = StructuredTool.from_function(chitchat_tool)
web_tool = StructuredTool.from_function(web_search_tool)
deep_tool = StructuredTool.from_function(run_gpt_researcher)

class State(BaseModel):
    query: str
    route: Optional[str] = None
    answer: Optional[str] = None

# === LangGraph nodes and routing ===
async def router(state: State):
    prompt = f'''
Classify input:
"{state.query}"

Return:
- vector → if it's a research query.
- web → if it requires up-to-date info.
- chitchat → for greetings or casual talk.

Just respond with one word: vector, web, or chitchat.
'''
    res = await llm.ainvoke(prompt)
    return {"route": res.content.strip().lower()}

async def vector_node(state: State):
    return {"answer": await vector_tool.ainvoke({"query": state.query})}

async def web_node(state: State):
    return {"answer": await web_tool.ainvoke({"query": state.query})}

async def chitchat_node(state: State):
    return {"answer": await chitchat.ainvoke({"query": state.query})}

async def deep_node(state: State):
    return {"answer": await deep_tool.ainvoke({"query": state.query})}

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

# === Streamlit UI ===
st.set_page_config(page_title="Retail Hybrid Agent", page_icon="🧠")
st.title("🧠 Retail Research — Hybrid LangGraph Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"]=="user" else "assistant"):
        st.markdown(msg["content"])

# Handle new input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.session_state.last_query = prompt
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.spinner("Thinking..."):
        result = asyncio.run(agent.ainvoke(State(query=prompt)))
    answer = result["answer"]
    if answer.startswith("❌"):
        st.session_state.messages.append({"role":"assistant","content":"I couldn’t find enough locally. Click below to run Deep Research!"})
    else:
        st.session_state.messages.append({"role":"assistant","content":answer})
    st.rerun()

# Offer fallback deep research if needed
if st.session_state.messages and "Deep Research" in st.session_state.messages[-1]["content"]:
    if st.button("🔍 Run Deep Research for this"):
        with st.spinner("Running Deep Research..."):
            report = asyncio.run(deep_tool.ainvoke({"query":st.session_state.last_query}))
            st.session_state.messages.append({"role":"assistant","content":report})
            st.rerun()

# Manual deep research UI
st.divider()
with st.expander("💡 Manual Deep Research"):
    manual_q = st.text_input("Your deep research topic:")
    if st.button("Run Now"):
        if manual_q.strip():
            with st.spinner("Running Deep Research..."):
                report = asyncio.run(deep_tool.ainvoke({"query":manual_q}))
                st.session_state.messages.append({"role":"user","content":f"Manual topic: {manual_q}"})
                st.session_state.messages.append({"role":"assistant","content":report})
                st.rerun()
        else:
            st.warning("Enter a topic first.")
