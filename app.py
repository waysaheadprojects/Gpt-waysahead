import os
import requests
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
    return ChatOpenAI(model="gpt-4.1-nano", temperature = 0.9)

@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings()

FIXED_DOMAINS = [
    "https://en.wikipedia.org/wiki/Mall_of_the_Emirates",
    "https://www.malloftheemirates.com/en",
    "https://www.brandsforless.com/en-in/",
    "https://en.wikipedia.org/wiki/BFL_Group",
]

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
def get_base_vectorstore():
    path = ".chroma_cache/base"
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

@st.cache_resource
def get_pdf_vectorstore():
    path = ".chroma_cache/pdfs"
    os.makedirs(path, exist_ok=True)

    # If we already have embedded PDFs, just load them:
    if os.path.exists(os.path.join(path, "index")):
        return Chroma(persist_directory=path, embedding_function=get_embeddings())

    # Otherwise, initialize empty store:
    return Chroma(persist_directory=path, embedding_function=get_embeddings())

def add_pdfs_to_store(uploaded_files, pdf_vs):
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
    pdf_vs.add_documents(chunks)
    pdf_vs.persist()
    return pdf_vs

def get_merged_vectorstore():
    base_vs = get_base_vectorstore()
    pdf_vs = get_pdf_vectorstore()
    base_vs.merge_from(pdf_vs)
    return base_vs

def get_retriever_chain(vs):
    retriever = vs.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        ("user", "Generate best search query."),
    ])
    return create_history_aware_retriever(get_llm(), retriever, prompt)

def get_rag_chain(retriever_chain):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a retail assistant. Use only the context below.
If the user asks for lists or tables, use clean Markdown.
If info is missing, say: 'I couldn’t find that in the available information.'

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

# 1️⃣ Always load merged vector store:
st.session_state.vector_store = get_merged_vectorstore()

# 2️⃣ Allow PDF upload if new:
uploaded_files = st.file_uploader(
    "📄 Upload new PDF(s) — these will be saved permanently:",
    type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    pdf_vs = get_pdf_vectorstore()
    add_pdfs_to_store(uploaded_files, pdf_vs)
    st.success(f"✅ Added {len(uploaded_files)} PDF(s)! Reload page to use them.")

# 3️⃣ Init chat history:
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hi 👋! Ask me anything about BFL, Mall, or all uploaded PDFs.")
    ]

# 4️⃣ Handle chat:
user_input = st.chat_input("Type your question here...")
if user_input:
    answer = get_answer(user_input)
    st.session_state.chat_history.extend([
        HumanMessage(content=user_input),
        AIMessage(content=answer)
    ])

# 5️⃣ Show chat:
for msg in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(msg, AIMessage) else "Human"):
        st.markdown(msg.content)
