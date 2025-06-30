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

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# === Load env ===
load_dotenv()
st.set_page_config(page_title="Retail Chatbot", page_icon="🧠")

st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o")  # or your model of choice

@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings()

FIXED_DOMAINS = [
    "https://en.wikipedia.org/wiki/Mall_of_the_Emirates",
    "https://www.malloftheemirates.com/en",
    "https://www.brandsforless.com/en-in/",
    "https://en.wikipedia.org/wiki/BFL_Group",
]

# === 1️⃣ RSS HEADLINE ===
def get_fact_from_rss():
    try:
        feed = feedparser.parse("https://www.retaildive.com/rss/")
        if feed.entries:
            headline = feed.entries[0].title
            return f"📰 Retail Headline: {headline}"
    except Exception as e:
        print(f"RSS failed: {e}")
    return None

# === 2️⃣ DuckDuckGo fallback ===
def get_duckduckgo_fact():
    try:
        query = "latest retail news"
        url = f"https://duckduckgo.com/html/?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        links = soup.find_all("a", {"class": "result__a"})
        if links:
            headline = links[0].text.strip()
            return f"📰 Retail Insight: {headline}"
    except Exception as e:
        print(f"DDG failed: {e}")
    return None

# === 3️⃣ Static fallback ===
WELCOME_FACTS = [
    "Mall of the Emirates attracts 42 million visitors every year.",
    "BFL Group runs over 74 stores across the Middle East & Europe.",
    "Omnichannel retail is growing by 14% every year.",
    "78% of shoppers prefer sustainable brands.",
    "Brands For Less recently expanded its online footprint in India."
]

# === 4️⃣ Vector store + LLM tiny summarizer ===
def get_fact_from_vectorstore(vs):
    try:
        retriever = vs.as_retriever()
        docs = retriever.get_relevant_documents("Share one interesting fact about BFL or Mall of the Emirates.")
        chunks = [d.page_content.strip() for d in docs if len(d.page_content.strip()) > 50]
        if not chunks:
            return None
        combined = " ".join(chunks[:3])
        prompt = f"Extract one clear, recent fact about BFL Group or Mall of the Emirates from this text. Keep it under 25 words. Text: {combined[:1000]}"
        response = get_llm().invoke(prompt)
        clean_fact = response.strip().replace("\n", " ")
        if clean_fact:
            return f"📌 Retail Fact: {clean_fact}"
    except Exception as e:
        print(f"Vectorstore fact failed: {e}")
    return None

def get_best_welcome_fact(vs):
    fact = get_fact_from_vectorstore(vs)
    if fact:
        return fact
    fact = get_fact_from_rss()
    if fact:
        return fact
    fact = get_duckduckgo_fact()
    if fact:
        return fact
    return f"💡 {random.choice(WELCOME_FACTS)}"

# === Crawl helper ===
def crawl_links(start_url, max_pages=10):
    seen = set()
    to_visit = [start_url]
    base_domain = urlparse(start_url).netloc

    while to_visit and len(seen) < max_pages:
        current = to_visit.pop(0)
        if current in seen or urlparse(current).netloc != base_domain:
            continue
        try:
            r = requests.get(current, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a['href']
                full_url = urljoin(current, href)
                if full_url.startswith("http") and urlparse(full_url).netloc == base_domain:
                    to_visit.append(full_url)
        except Exception:
            continue
        seen.add(current)
    return list(seen)

@st.cache_resource
def get_or_create_vectorstore():
    path = ".chroma_cache/main"
    os.makedirs(path, exist_ok=True)

    if os.path.exists(os.path.join(path, "index")):
        return Chroma(persist_directory=path, embedding_function=get_embeddings())

    all_urls = []
    with st.spinner("🌐 Crawling sites..."):
        for url in FIXED_DOMAINS:
            urls = crawl_links(url, max_pages=10)
            all_urls.extend(urls)

        docs = []
        for url in tqdm(all_urls, desc="Loading pages"):
            try:
                loader = WebBaseLoader(url)
                for doc in loader.load():
                    doc.metadata["source_url"] = url
                    docs.append(doc)
            except Exception:
                continue

        chunks = RecursiveCharacterTextSplitter().split_documents(docs)
        vs = Chroma.from_documents(chunks, get_embeddings(), persist_directory=path)
        vs.persist()
    st.success(f"✅ Indexed {len(all_urls)} pages.")
    return vs

def add_pdfs_to_vectorstore(uploaded_files, vs):
    os.makedirs("./uploads", exist_ok=True)
    docs = []
    for file in uploaded_files:
        save_path = f"./uploads/{file.name}"
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader(save_path)
        for doc in loader.load():
            doc.metadata["source_pdf"] = file.name
            docs.append(doc)
    chunks = RecursiveCharacterTextSplitter().split_documents(docs)
    vs.add_documents(chunks)
    vs.persist()
    return vs

def pdfs_already_uploaded():
    return len(glob.glob("./uploads/*.pdf")) > 0

def get_retriever_chain(vs):
    retriever = vs.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        ("user", "Generate a smart search query to retrieve the best answer.")
    ])
    return create_history_aware_retriever(get_llm(), retriever, prompt)

def get_rag_chain(retriever_chain):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a smart, conversational retail knowledge assistant.
Use only the context below to answer with reliable facts.
Do not invent. If you don’t find info, say: "Sorry, I couldn’t find that in the documents I have."
Use clear markdown for lists or tables.
Be warm, concise, and human.
Context:
{context}
"""),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ])
    return create_retrieval_chain(retriever_chain, create_stuff_documents_chain(get_llm(), prompt))

def get_answer(user_input):
    retriever_chain = get_retriever_chain(st.session_state.vector_store)
    rag_chain = get_rag_chain(retriever_chain)
    result = rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return result["answer"]

# === UI ===
st.title("🧠 Retail Chatbot — BFL, Mall & Permanent PDFs")

st.session_state.vector_store = get_or_create_vectorstore()

if not pdfs_already_uploaded():
    uploaded_files = st.file_uploader(
        "📄 Upload PDF(s) — these will be stored permanently for everyone:",
        type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files:
        add_pdfs_to_vectorstore(uploaded_files, st.session_state.vector_store)
        st.success(f"✅ Added {len(uploaded_files)} PDF(s)! Reload page to use them.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content=get_best_welcome_fact(st.session_state.vector_store))
    ]

user_input = st.chat_input("Type your question here...")
if user_input:
    answer = get_answer(user_input)
    st.session_state.chat_history.extend([
        HumanMessage(content=user_input),
        AIMessage(content=answer)
    ])

for msg in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(msg, AIMessage) else "Human"):
        st.markdown(msg.content)
