import os
import re
import asyncio
import random
import glob
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
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

# === Common retriever chain ===
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
        ("system", "Use ONLY context below. If none, say '❌ No vector answer.'\nContext: {context}"),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ])
    return create_retrieval_chain(chain, create_stuff_documents_chain(llm, prompt))

# === TOOL 1: Vector Store
@tool
async def vector_lookup_tool(query: str) -> str:
    """
    Attempt to answer a user question using the local FAISS vector store.
    Returns relevant context snippets or a 'No vector answer' signal.
    This tool should be used for any normal lookup: general Q&A, fact-check, simple facts.
    """
    docs = vs.similarity_search(query, k=5)
    if not docs:
        return "❌ No vector answer."
    context = "\n\n".join([d.page_content for d in docs])
    return f"✅ Vector answer:\n\n{context[:800]}..."

# === TOOL 2: Deep Research (fallback)
@tool
async def deep_research_tool(query: str) -> str:
    """
    Run a full deep hybrid research workflow using GPTResearcher.
    Combines the FAISS vector store and online web search to create a detailed report.
    Use this ONLY if the vector store has no good match AND user approves deeper research.
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

# === LangGraph: Define states ===
class ResearchState:
    query: str
    vector_result: str
    approved_deep: bool

# === Build the Graph ===
graph = StateGraph(ResearchState)

# Node 1: Vector lookup
async def node_vector(state):
    query = state["query"]
    result = await vector_lookup_tool.ainvoke({"query": query})
    return {"vector_result": result}

# Node 2: Branch check
def branch_check(state):
    if state["vector_result"].startswith("✅"):
        return "end"
    return "ask_user"

# Node 3: Ask user whether to proceed with deep research
async def ask_user_node(state):
    print("\n🤖 The vector store didn’t find a clear match.")
    print("This question might benefit from deeper research.")
    approved = input("Run Deep Research? [yes/no]: ").strip().lower() == "yes"
    return {"approved_deep": approved}

# Node 4: If approved, run GPTResearcher
async def run_deep_node(state):
    query = state["query"]
    result = await deep_research_tool.ainvoke({"query": query})
    return {"vector_result": result}

# === Add nodes & edges ===
graph.add_node("vector_lookup", node_vector)
graph.add_node("ask_user", ask_user_node)
graph.add_node("deep_research", run_deep_node)

graph.set_entry_point("vector_lookup")
graph.add_edge("vector_lookup", branch_check)
graph.add_conditional_edges("vector_lookup", branch_check, {
    "end": END,
    "ask_user": "ask_user"
})
graph.add_edge("ask_user", "deep_research")
graph.add_edge("deep_research", END)

agent = graph.compile()

# === Run CLI ===
if __name__ == "__main__":
    query = input("Ask your question: ")
    result = asyncio.run(agent.invoke({"query": query}))
    print("\n=== Final Answer ===")
    print(result["vector_result"])
