import os
import asyncio
from dotenv import load_dotenv

import streamlit as st

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from gpt_researcher import GPTResearcher
import gpt_researcher.actions.agent_creator as agent_creator

# === PATCH for JSON regex bug ===
original = agent_creator.extract_json_with_regex
def safe_extract_json_with_regex(response):
    if not response:
        return None
    return original(response)
agent_creator.extract_json_with_regex = safe_extract_json_with_regex

# === Load environment ===
load_dotenv()
os.environ["REPORT_SOURCE"] = "local"

# === Load models & vector store ===
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.3)
embeddings = OpenAIEmbeddings()
vs = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)


def get_rag_chain(chain):
    """
    Creates a retrieval-augmented generation (RAG) chain.
    This chain uses a strong system prompt instructing the LLM to:
    - Only use the provided context.
    - Not hallucinate data.
    - Reply with clear tables, bullet points, and chart suggestions if useful.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """
You are a trusted retail and consumer research assistant.
Use ONLY the context below. If no context, say: ❌ I have no data for that.
Never make up numbers or facts.
Use bullet points, markdown tables, or chart suggestions if it helps.
Context: {context}
         """),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ])
    return create_retrieval_chain(
        chain,
        create_stuff_documents_chain(llm, prompt)
    )


def get_retriever_chain():
    """
    Creates a retriever chain that takes the user query,
    generates a precise search query, and retrieves similar chunks from the vector store.
    """
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        ("user", "Generate a precise search query.")
    ])
    return create_history_aware_retriever(llm, retriever, prompt)


async def vector_lookup_tool(query: str) -> str:
    """
    Uses the local FAISS vector store to answer the question.
    If nothing relevant is found, returns a ❌ indicator.
    If found, runs a RAG chain to craft a clear answer.
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


async def deep_research_tool(query: str) -> str:
    """
    Runs a full GPTResearcher hybrid research pipeline.
    Combines local vector store + live web search.
    Logs each step to a server-side text file for auditing.
    Returns a detailed multi-part research report.
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


# === Streamlit UI ===
st.set_page_config(page_title="Retail Hybrid Chatbot", page_icon="🧠")
st.title("🧠 Retail Research — Hybrid Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# Display full chat history
for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

# Main chat input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.last_query = prompt

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Checking local knowledge..."):
        answer = asyncio.run(vector_lookup_tool(prompt))

    if answer.startswith("❌"):
        st.session_state.messages.append({
            "role": "assistant",
            "content": "I couldn’t find this in my local knowledge. Click below to run Deep Research!"
        })
    else:
        st.session_state.messages.append({"role": "assistant", "content": answer})

    st.rerun()

# If vector store failed, offer fallback button
if st.session_state.messages and "Deep Research" in st.session_state.messages[-1]["content"]:
    if st.button("🔍 Run Deep Research for this"):
        with st.spinner("Running Deep Research..."):
            deep_report = asyncio.run(deep_research_tool(st.session_state.last_query))
            st.session_state.messages.append({"role": "assistant", "content": deep_report})
            st.rerun()

# ✅ Subtle manual Deep Research block BELOW the chat — collapsible
with st.expander("💡 Or run Deep Research on any custom topic"):
    manual_topic = st.text_input("Topic for manual Deep Research:")
    if st.button("Run Deep Research Manually"):
        if manual_topic.strip():
            with st.spinner("Running Deep Research..."):
                manual_report = asyncio.run(deep_research_tool(manual_topic))
                st.session_state.messages.append(
                    {"role": "user", "content": f"Run Deep Research for: {manual_topic}"}
                )
                st.session_state.messages.append({"role": "assistant", "content": manual_report})
                st.rerun()
        else:
            st.warning("Please enter a topic.")
