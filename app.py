import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import streamlit as st
from tqdm import tqdm
from dotenv import load_dotenv
import hashlib

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- Load env ---
load_dotenv()

st.set_page_config(page_title="Crawled Brand Chat", page_icon="🏬")
# --- Hide Sidebar Completely ---
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# --- Embedding / LLM ---
@st.cache_resource
def get_llm():
    return ChatOpenAI()

@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings()

# --- Fixed domains to crawl ---
FIXED_DOMAINS = [
    "https://en.wikipedia.org/wiki/Mall_of_the_Emirates",
    "https://www.malloftheemirates.com/en",
    "https://www.brandsforless.com/en-in/",
    "https://en.wikipedia.org/wiki/BFL_Group",
]

# --- Crawl internal links from a domain ---
def crawl_links(start_url, max_pages=20):
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


# --- Vectorstore loader (crawler + embedder) ---
@st.cache_resource
def get_vectorstore_from_crawled_domains():
    cache_path = ".chroma_cache/crawled_fixed"
    os.makedirs(cache_path, exist_ok=True)

    if os.path.exists(os.path.join(cache_path, "index")):
        return Chroma(persist_directory=cache_path, embedding_function=get_embeddings())

    all_urls = []
    with st.spinner("🌐 Crawling and indexing all pages..."):
        for url in FIXED_DOMAINS:
            # st.write(f"Crawling: {url}")  # ← REMOVE this line to hide individual link
            urls = crawl_links(url, max_pages=20)
            all_urls.extend(urls)

        documents = []
        for url in tqdm(all_urls, desc="Loading pages"):
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source_url"] = url
                documents.extend(docs)
            except Exception:
                continue

        chunks = RecursiveCharacterTextSplitter().split_documents(documents)
        vectorstore = Chroma.from_documents(chunks, get_embeddings(), persist_directory=cache_path)
        vectorstore.persist()

    st.success(f"✅ Indexed {len(all_urls)} pages.")
    return vectorstore


# --- RAG pipeline ---
def get_context_retriever_chain(vector_store):
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to get relevant information."),
    ])
    return create_history_aware_retriever(get_llm(), retriever, prompt)

def get_conversational_rag_chain(retriever_chain):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based on the context:\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ])
    return create_retrieval_chain(retriever_chain, create_stuff_documents_chain(get_llm(), prompt))

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    rag_chain = get_conversational_rag_chain(retriever_chain)
    result = rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return result["answer"]

# --- UI ---

st.title("🧠 Ask Me About BFL & Mall of the Emirates")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore_from_crawled_domains()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hi! Ask me anything about Mall of the Emirates or BFL Group.")]

user_query = st.chat_input("Ask your question...")
if user_query:
    response = get_response(user_query)
    st.session_state.chat_history.extend([
        HumanMessage(content=user_query),
        AIMessage(content=response)
    ])

for msg in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(msg, AIMessage) else "Human"):
        st.markdown(msg.content)
