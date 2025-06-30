import os
import re
import asyncio
from dotenv import load_dotenv

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

# === Env ===
load_dotenv()
os.environ["REPORT_SOURCE"] = "local"

# === LLM & Vector store ===
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.3)
embeddings = OpenAIEmbeddings()
vs = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)

# === Retriever & RAG Chain with Strong Analytical Prompt ===
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
You are a senior research analyst for retail, consumer, and brand intelligence.
When answering:
- Use ONLY the provided context. If there is no context, reply: '❌ I have no data for that.'
- Do NOT make up numbers.
- Use clear bullet points, short paragraphs, or markdown tables if relevant.
- If helpful, describe simple charts (bar, pie, trend line) that could illustrate your answer.
- Always break down your reasoning step by step and highlight key takeaways in **bold**.

Context: {context}
         """),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ])
    return create_retrieval_chain(
        chain,
        create_stuff_documents_chain(llm, prompt)
    )

# === TOOL 1: Vector store lookup ===
@tool
async def vector_lookup_tool(query: str) -> str:
    """
    Attempt to answer using the local FAISS vector store.
    If nothing found, returns '❌ No vector answer.'
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

# === TOOL 2: Deep Research fallback ===
@tool
async def deep_research_tool(query: str) -> str:
    """
    Runs a detailed hybrid research report using GPTResearcher.
    Combines vector store + online search. Saves logs to server.
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

# === LangGraph state ===
class ResearchState:
    query: str
    vector_result: str
    approved_deep: bool

# === Nodes ===
async def node_vector(state):
    query = state["query"]
    result = await vector_lookup_tool.ainvoke({"query": query})
    return {"vector_result": result}

def branch_check(state):
    if not state["vector_result"].startswith("❌"):
        return "end"
    return "ask_user"

async def ask_user_node(state):
    print("\n🤖 I couldn’t find enough trusted info in local data alone.")
    approved = input("Run full Deep Research using web + local? [yes/no]: ").strip().lower() == "yes"
    return {"approved_deep": approved}

async def run_deep_node(state):
    query = state["query"]
    result = await deep_research_tool.ainvoke({"query": query})
    return {"vector_result": result}

# === Graph ===
graph = StateGraph(ResearchState)

graph.add_node("vector_lookup", node_vector)
graph.add_node("ask_user", ask_user_node)
graph.add_node("deep_research", run_deep_node)

graph.set_entry_point("vector_lookup")

graph.add_conditional_edges(
    "vector_lookup",
    branch_check,
    {
        "end": END,
        "ask_user": "ask_user"
    }
)

graph.add_edge("ask_user", "deep_research")
graph.add_edge("deep_research", END)

agent = graph.compile()

# === CLI Runner ===
if __name__ == "__main__":
    query = input("Ask any question: ")
    result = asyncio.run(agent.invoke({"query": query}))
    print("\n=== Final Answer ===")
    print(result["vector_result"])
