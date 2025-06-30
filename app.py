import os
import asyncio
import time
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

# === Load ===
load_dotenv()
os.environ["REPORT_SOURCE"] = "local"

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
You are a retail research assistant.
Use ONLY the provided context. If none, reply ❌.
Use markdown, bullet points, or tables if helpful.
Context: {context}
        """),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ])
    return create_retrieval_chain(chain, create_stuff_documents_chain(llm, prompt))

async def vector_lookup(query: str) -> str:
    """Local vector store search."""
    docs = vs.similarity_search(query, k=5)
    if not docs:
        return "❌ No vector match."
    context = "\n\n".join([d.page_content for d in docs])
    chain = get_rag_chain(get_retriever_chain())
    result = await chain.ainvoke({"chat_history": [], "input": query, "context": context})
    return result["answer"]

async def chitchat_tool(query: str) -> str:
    """Quick chit-chat fallback."""
    prompt = f'User said: "{query}". Reply nicely in 1–2 short lines.'
    return (await llm.ainvoke(prompt)).content.strip()

async def run_gpt_researcher(query: str) -> str:
    """Run the GPT Researcher deep report."""
    log_file = "./research_logs.txt"
    open(log_file, "w").close()

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

async def deep_tool_fn(query: str) -> str:
    """Run deep GPT researcher report."""
    return await run_gpt_researcher(query)

vector_tool = StructuredTool.from_function(vector_lookup)
chitchat = StructuredTool.from_function(chitchat_tool)
deep_tool = StructuredTool.from_function(deep_tool_fn)

class State(BaseModel):
    query: str
    route: Optional[str] = None
    answer: Optional[str] = None

async def router(state: State):
    res = await llm.ainvoke(f'''
Classify this input:
"{state.query}"
Return: vector OR chitchat.
''')
    return {"route": res.content.strip().lower()}

async def vector_node(state: State):
    return {"answer": await vector_tool.ainvoke({"query": state.query})}

async def chitchat_node(state: State):
    return {"answer": await chitchat.ainvoke({"query": state.query})}

async def deep_node(state: State):
    return {"answer": await deep_tool.ainvoke({"query": state.query})}

graph = StateGraph(State)
graph.add_node("router", router)
graph.add_node("vector", vector_node)
graph.add_node("chitchat", chitchat_node)
graph.add_node("deep", deep_node)

graph.set_entry_point("router")
graph.add_conditional_edges("router", lambda s: s.route, {
    "vector": "vector",
    "chitchat": "chitchat"
})
graph.add_edge("vector", END)
graph.add_edge("chitchat", END)
graph.add_edge("deep", END)

agent = graph.compile()

# === Streamlit ===
st.set_page_config(page_title="Retail Hybrid Agent", page_icon="🧠")
st.title("🧠 Retail Research — Hybrid LangGraph Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

async def main():
    if prompt := st.chat_input("Ask anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.last_query = prompt
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            result = await agent.ainvoke(State(query=prompt))
            answer = result["answer"]
            if asyncio.iscoroutine(answer):
                answer = await answer

            if answer.startswith("❌"):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I couldn’t find enough locally. Click below for Deep Research!"
                })
            else:
                st.session_state.messages.append({"role": "assistant", "content": answer})

        st.rerun()

    st.divider()
    with st.expander("💡 Manual Deep Research"):
        manual_q = st.text_input("Your topic:")
        if st.button("Run Manual Research"):
            if manual_q.strip():
                def stream_research():
                    log_file = "./research_logs.txt"
                    task = asyncio.run(deep_tool.ainvoke({"query": manual_q}))

                    # While running, stream log file
                    last_content = ""
                    while True:
                        time.sleep(1)
                        if os.path.exists(log_file):
                            with open(log_file) as f:
                                content = f.read()
                                if content != last_content:
                                    last_content = content
                                    yield f"```\n{content}\n```"
                        if not asyncio.iscoroutine(task):
                            break
                    yield f"\n\n## ✅ Final Report:\n\n{task}"

                st.write_stream(stream_research)
            else:
                st.warning("Please enter a topic.")

asyncio.run(main())
