import os
import glob
import random
import requests
import feedparser
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import streamlit as st
from tqdm import tqdm
from dotenv import load_dotenv

import asyncio
from gpt_researcher import GPTResearcher

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# === Load env ===
load_dotenv()
st.set_page_config(page_title="Retail Research Assistant — Hybrid Tool", page_icon="🧠")

st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)

@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings()

FIXED_DOMAINS = [
    "https://en.wikipedia.org/wiki/Mall_of_the_Emirates",
    "https://www.malloftheemirates.com/en",
    "https://www.brandsforless.com/en-in/",
    "https://en.wikipedia.org/wiki/BFL_Group",
    "https://en.wikipedia.org/wiki/The_Dubai_Mall",
    "https://thedubaimall.com/"
]

WELCOME_FACTS = [
    "Mall of the Emirates attracts over 42 million visitors every year.",
    "BFL Group runs over 74 stores across the Middle East & Europe.",
    "Omnichannel retail is growing by 14% every year.",
    "78% of shoppers prefer sustainable brands.",
    "Brands For Less recently expanded its online footprint in India."
]

def get_fact_from_rss():
    try:
        feed = feedparser.parse("https://www.retaildive.com/rss/")
        if feed.entries:
            return f"📰 Retail Headline: {feed.entries[0].title}"
    except:
        return None

def get_duckduckgo_fact():
    try:
        url = "https://duckduckgo.com/html/?q=latest+retail+news"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        link = soup.find("a", {"class": "result__a"})
        if link:
            return f"📰 Retail Insight: {link.text.strip()}"
    except:
        return None

def get_fact_from_vectorstore(vs):
    try:
        docs = vs.similarity_search(
            "Share one interesting fact about retail, BFL Group or malls.", k=5)
        chunks = [d.page_content.strip() for d in docs if len(d.page_content.strip()) > 50]
        if not chunks: return None
        combined = " ".join(chunks[:3])
        prompt = f"Extract one short, interesting fact. Under 30 words. Text: {combined[:800]}"
        response = get_llm().invoke(prompt)
        return f"📌 Retail Fact: {response.content.strip()}"
    except: return None

def get_best_welcome_fact(vs):
    return get_fact_from_vectorstore(vs) or get_fact_from_rss() or get_duckduckgo_fact() or f"💡 {random.choice(WELCOME_FACTS)}"

def crawl_links(start_url, max_pages=5):
    seen, to_visit = set(), [start_url]
    base = urlparse(start_url).netloc
    while to_visit and len(seen) < max_pages:
        current = to_visit.pop(0)
        if current in seen or urlparse(current).netloc != base: continue
        try:
            soup = BeautifulSoup(requests.get(current, timeout=10).text, "html.parser")
            to_visit.extend(
                urljoin(current, a['href']) for a in soup.find_all("a", href=True)
                if urlparse(urljoin(current, a['href'])).netloc == base
            )
        except: pass
        seen.add(current)
    return list(seen)

@st.cache_resource
def get_or_create_vectorstore():
    path = "./faiss_index"
    embeddings = get_embeddings()
    if os.path.exists(path):
        if "web_pages_indexed" not in st.session_state:
            try: st.session_state.web_pages_indexed = int(open(".faiss_urls.txt").read())
            except: st.session_state.web_pages_indexed = 0
        if "pdf_pages_indexed" not in st.session_state:
            try: st.session_state.pdf_pages_indexed = int(open(".faiss_pdfs.txt").read())
            except: st.session_state.pdf_pages_indexed = 0
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

    all_urls, docs = [], []
    for url in FIXED_DOMAINS:
        all_urls.extend(crawl_links(url, max_pages=5))
    for url in tqdm(all_urls):
        try: docs.extend(WebBaseLoader(url).load())
        except: pass

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(path)

    with open(".faiss_urls.txt", "w") as f: f.write(str(len(all_urls)))
    st.session_state.web_pages_indexed = len(all_urls)
    st.session_state.pdf_pages_indexed = 0

    st.success(f"✅ Indexed {len(all_urls)} website pages.")
    return vs

def add_pdfs_to_vectorstore(files, vs):
    os.makedirs("./uploads", exist_ok=True)
    pdf_docs, count = [], 0
    for file in files:
        save = f"./uploads/{file.name}"
        with open(save, "wb") as f: f.write(file.getbuffer())
        loaded = PyPDFLoader(save).load()
        count += len(loaded)
        for doc in loaded:
            heading = next((line.strip() for line in doc.page_content.splitlines() if line.strip().isupper()), None)
            if heading: doc.metadata["heading"] = heading
            doc.metadata["source_pdf"] = file.name
            pdf_docs.append(doc)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    chunks = splitter.split_documents(pdf_docs)
    vs.add_documents(chunks)
    vs.save_local("./faiss_index")

    st.session_state.pdf_pages_indexed = st.session_state.get("pdf_pages_indexed", 0) + count
    with open(".faiss_pdfs.txt", "w") as f: f.write(str(st.session_state.pdf_pages_indexed))
    st.success(f"✅ Added {len(files)} PDF(s) with {count} pages!")
    return vs

def get_best_relevant_chunks(q, vs):
    docs = vs.similarity_search(q, k=8)
    if not docs:
        hits = [d for d in vs.docstore._dict.values() if q.lower() in d.page_content.lower()]
        if hits: return hits
    return docs

def get_retriever_chain(vs):
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 200})
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        ("user", "Generate a precise search query.")
    ])
    return create_history_aware_retriever(get_llm(), retriever, prompt)

def get_rag_chain(chain):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a deep retail researcher.
Use ONLY the context below — from vector store OR fresh external fallback.
Never hallucinate.
Always break down numbers, brands, categories in tables or lists.
If info missing, say so clearly.
Context: {context}"""),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ])
    return create_retrieval_chain(chain, create_stuff_documents_chain(get_llm(), prompt))

async def get_fresh_context_from_gpt_researcher(topic: str):
    researcher = GPTResearcher(
        query=topic,
        report_source="hybrid",
        verbosity="high"
    )
    await researcher.conduct_research()
    context = await researcher.write_report()
    return context

def get_answer(user_input):
    vs = st.session_state.vector_store
    docs = get_best_relevant_chunks(user_input, vs)

    if docs:
        merged_context = "\n\n".join([d.page_content for d in docs])
    else:
        with st.spinner("🔍 No match — using GPT Researcher Tool..."):
            merged_context = asyncio.run(get_fresh_context_from_gpt_researcher(user_input))

    chain = get_retriever_chain(vs)
    rag = get_rag_chain(chain)

    with st.spinner("🤖 Generating final answer..."):
        result = rag.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_input,
            "context": merged_context
        })
    return result["answer"]

# === UI ===
st.title("🧠 Retail Research Assistant — FAISS + GPT Researcher Fallback")
st.session_state.vector_store = get_or_create_vectorstore()

if not os.path.exists("./faiss_index") or not glob.glob("./uploads/*.pdf"):
    uploaded_files = st.file_uploader(
        "📄 Upload PDF(s) — permanently stored",
        type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files and len(uploaded_files) > 0:
        add_pdfs_to_vectorstore(uploaded_files, st.session_state.vector_store)

st.info(
    f"📊 Total indexed — Websites: {st.session_state.get('web_pages_indexed',0)} | PDFs: {st.session_state.get('pdf_pages_indexed',0)} pages"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content=get_best_welcome_fact(st.session_state.vector_store))]

q = st.chat_input("Ask me anything…")
if q:
    a = get_answer(q)
    st.session_state.chat_history.extend([HumanMessage(content=q), AIMessage(content=a)])

for msg in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(msg, AIMessage) else "Human"):
        st.markdown(msg.content)
