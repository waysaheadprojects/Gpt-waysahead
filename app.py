import os
import asyncio
from dotenv import load_dotenv

import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from gpt_researcher import GPTResearcher
import gpt_researcher.actions.agent_creator as agent_creator

# === PATCH ===
original = agent_creator.extract_json_with_regex
def safe_extract_json_with_regex(response):
    if not response:
        return None
    return original(response)
agent_creator.extract_json_with_regex = safe_extract_json_with_regex

# === Load env ===
load_dotenv()
os.environ["REPORT_SOURCE"] = "local"

# === Setup ===
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.3)
embeddings = OpenAIEmbeddings()
vs = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)


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
You are a trusted retail & consumer research assistant.
Use ONLY the context. If missing, say ❌ I have no data for that.
Never hallucinate. Use markdown tables, bullet points, chart suggestions.
Context: {context}
         """),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ])
    return create_retrieval_chain(chain, create_stuff_documents_chain(llm, prompt))


async def vector_lookup(query: str) -> str:
    """
    Answer user retail research queries using the local FAISS vector store.
    Return a clear answer if found. If nothing relevant, say ❌.
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


async def chitchat_reply(query: str) -> str:
    """
    Answer casual greetings, meta, or chit-chat with a friendly short message.
    Example: Who are you? Hi! How are you?
    """
    prompt = f"""
A user asked: "{query}"

Reply warmly in 1-2 lines as a helpful assistant. No vector search needed.
"""
    result = await llm.ainvoke(prompt)
    return result.content.strip()


async def deep_research(query: str) -> str:
    """
    Run full GPTResearcher hybrid mode for in-depth research combining web + local.
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


# === Wrap as structured tools ===
vector_tool = StructuredTool.from_function(vector_lookup)
chitchat_tool = StructuredTool.from_function(chitchat_reply)

tools = [vector_tool, chitchat_tool]

# === Streamlit ===
st.set_page_config(page_title="Retail Hybrid Chatbot", page_icon="🧠")
st.title("🧠 Retail Research — Hybrid Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

# === Tool routing prompt ===
route_prompt = """You are a smart router.
Decide if the user wants:
- normal research (use vector_lookup)
- or just chit-chat (use chitchat_reply)
Reply ONLY with the tool name: 'vector_lookup' or 'chitchat_reply'.
User input: {query}"""

# === Main input ===
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.last_query = prompt

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        route_decision = asyncio.run(llm.ainvoke(route_prompt.format(query=prompt)))
        route = route_decision.content.strip().lower()

        if "vector" in route:
            answer = asyncio.run(vector_tool.ainvoke({"query": prompt}))
        else:
            answer = asyncio.run(chitchat_tool.ainvoke({"query": prompt}))

    if answer.startswith("❌"):
        st.session_state.messages.append({
            "role": "assistant",
            "content": "I couldn’t find this locally. Click below for full Deep Research!"
        })
    else:
        st.session_state.messages.append({"role": "assistant", "content": answer})

    st.rerun()

# Fallback Deep Research button
if st.session_state.messages and "Deep Research" in st.session_state.messages[-1]["content"]:
    if st.button("🔍 Run Deep Research for this"):
        with st.spinner("Running Deep Research..."):
            deep_report = asyncio.run(deep_research(st.session_state.last_query))
            st.session_state.messages.append({"role": "assistant", "content": deep_report})
            st.rerun()
