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

# === PATCH ===
original = agent_creator.extract_json_with_regex
def safe_extract_json_with_regex(response):
    if not response:
        return None
    return original(response)
agent_creator.extract_json_with_regex = safe_extract_json_with_regex

load_dotenv()
os.environ["REPORT_SOURCE"] = "local"

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.2)
embeddings = OpenAIEmbeddings()
vs = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)

# === Retrieval chain ===
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
        ("system",
         """
You are a retail research agent.
Use ONLY the context given. If none, say ❌.
Use markdown tables, bullet points, chart ideas where helpful.
Context: {context}
         """),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ])
    return create_retrieval_chain(chain, create_stuff_documents_chain(llm, prompt))

# === Tools ===
async def vector_lookup(query: str) -> str:
    docs = vs.similarity_search(query, k=5)
    if not docs:
        return "❌ No vector answer."
    context = "\n\n".join(d.page_content for d in docs)
    chain = get_rag_chain(get_retriever_chain())
    return chain.invoke({"chat_history": [], "input": query, "context": context})["answer"]

async def chitchat_tool(query: str) -> str:
    prompt = f'User asked: "{query}". Reply warmly in 1–2 lines.'
    return (await llm.ainvoke(prompt)).content.strip()

async def web_search(query: str) -> str:
    """
    Fetch latest info from the web using search tool.
    """
    search_query = [{"q": query, "recency": 1, "domains": None}]
    results = await asyncio.get_event_loop().run_in_executor(None, lambda: web.run({"search_query": search_query}))
    top = results.get("search_query", [])[0]
    return f"🌐 Web result: {top['q']}"

async def run_gpt_researcher(query: str) -> str:
    log_file = "./research_logs.txt"
    def capture_log(*args, **kwargs):
        with open(log_file, "a") as f:
            f.write(" ".join(str(a) for a in args) + "\n")
    researcher = GPTResearcher(query=query, report_type="research_report", report_source="hybrid", vector_store=vs, doc_path="./uploads")
    researcher.print = capture_log
    await researcher.conduct_research()
    return await researcher.write_report()

vector_tool = StructuredTool.from_function(vector_lookup)
chitchat = StructuredTool.from_function(chitchat_tool)
web_tool = StructuredTool.from_function(web_search)
deep_tool = StructuredTool.from_function(run_gpt_researcher)

class State(BaseModel):
    query: str
    route: Optional[str] = None
    answer: Optional[str] = None

async def router(state):
    q = state.query
    prompt = f'''
Classify input:
"{q}"
Return:
- vector → research
- chitchat → small talk
- web → needs live web update
Answer with one word.
'''
    return {"route": (await llm.ainvoke(prompt)).content.strip().lower()}

async def vector_node(state):
    return {"answer": await vector_tool.ainvoke({"query": state.query})}

async def chitchat_node(state):
    return {"answer": await chitchat.ainvoke({"query": state.query})}

async def web_node(state):
    return {"answer": await web_tool.ainvoke({"query": state.query})}

async def deep_node(state):
    return {"answer": await deep_tool.ainvoke({"query": state.query})}

graph = StateGraph(State)
graph.add_node("router", router)
graph.add_node("vector", vector_node)
graph.add_node("chitchat", chitchat_node)
graph.add_node("web", web_node)
graph.add_node("deep", deep_node)

graph.set_entry_point("router")
graph.add_conditional_edges("router", lambda s: s.route, {
    "vector": "vector",
    "chitchat": "chitchat",
    "web": "web"
})
graph.add_edge("vector", END)
graph.add_edge("chitchat", END)
graph.add_edge("web", END)
graph.add_edge("deep", END)

agent = graph.compile()

# === Streamlit UI ===
st.set_page_config(page_title="Retail Hybrid Agent", page_icon="🧠")
st.title("🧠 Retail Research — Hybrid LangGraph Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.last_query = prompt
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        result = asyncio.run(agent.ainvoke(State(query=prompt)))
    answer = result["answer"]

    if answer.startswith("❌"):
        st.session_state.messages.append({
            "role": "assistant",
            "content": "I couldn’t find local info. Click below for Deep Research!"
        })
    else:
        st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

if st.session_state.messages and "Deep Research" in st.session_state.messages[-1]["content"]:
    if st.button("🔍 Run Deep Research"):
        with st.spinner("Running Deep Research..."):
            report = asyncio.run(deep_tool.ainvoke({"query": st.session_state.last_query}))
            st.session_state.messages.append({"role": "assistant", "content": report})
            st.rerun()

st.divider()
with st.expander("💡 Manual Deep Research"):
    manual_q = st.text_input("Your topic:")
    if st.button("Run Now"):
        if manual_q.strip():
            with st.spinner("Running Deep Research..."):
                report = asyncio.run(deep_tool.ainvoke({"query": manual_q}))
                st.session_state.messages.append({"role": "user", "content": f"Manual topic: {manual_q}"})
                st.session_state.messages.append({"role": "assistant", "content": report})
                st.rerun()
        else:
            st.warning("Enter a topic.")
