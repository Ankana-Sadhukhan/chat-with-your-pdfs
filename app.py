import math
import os
import re
import tempfile
import time
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(page_title="PDF RAG Chat", page_icon="📄", layout="wide")

# ---------------------------------------------------------------- resources
@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


@st.cache_resource
def get_llm():
    return init_chat_model("groq:openai/gpt-oss-20b")


PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.
Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""),
    ("human", """Context:
{context}

Question:
{question}"""),
])

# ------------------------------------------------------------ session state
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


# ------------------------------------------------------------------ helpers
def add_with_retry(vectorstore, batch, max_attempts=5, on_wait=None):
    """Add documents, waiting and retrying if the embedding quota is hit."""
    for attempt in range(1, max_attempts + 1):
        try:
            vectorstore.add_documents(batch)
            return
        except GoogleGenerativeAIError as error:
            message = str(error)
            lower = message.lower()
            is_quota = ("429" in message or "resource_exhausted" in lower) and (
                "quota" in lower or "per minute" in lower
            )
            if not is_quota or attempt == max_attempts:
                raise

            delays = re.findall(
                r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s", message
            )
            delays.extend(re.findall(r"retry in (\d+(?:\.\d+)?)s", lower))
            wait = math.ceil(max(map(float, delays))) + 1 if delays else 60
            if on_wait:
                on_wait(wait)
            time.sleep(wait)


def reset_index():
    vs = st.session_state.vectorstore
    if vs is not None:
        try:
            vs.delete_collection()
        except Exception:
            pass
    st.session_state.vectorstore = None
    st.session_state.indexed_files = []
    st.session_state.messages = []
    st.session_state.session_id = uuid.uuid4().hex[:12]


def build_index(uploaded_files, chunk_size, chunk_overlap):
    # fresh collection for every indexing run
    reset_index()

    all_docs = []
    for f in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getvalue())
            tmp_path = tmp.name
        try:
            pages = PyPDFLoader(tmp_path).load()
            for page in pages:
                page.metadata["source"] = f.name  # show real filename, not temp path
            all_docs.extend(pages)
        finally:
            os.remove(tmp_path)

    all_docs = [d for d in all_docs if d.page_content.strip()]
    if not all_docs:
        st.error(
            "No text could be extracted. These PDFs may be scanned images "
            "(they would need OCR first)."
        )
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(all_docs)

    vectorstore = Chroma(
        collection_name=f"session_{st.session_state.session_id}",
        embedding_function=get_embeddings(),
        persist_directory="chroma-db",
    )

    batch_size = 100
    progress = st.progress(0.0, text="Creating embeddings...")
    status = st.empty()
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        add_with_retry(
            vectorstore,
            batch,
            on_wait=lambda s: status.warning(
                f"Embedding quota reached, retrying in {s} seconds..."
            ),
        )
        status.empty()
        progress.progress(
            min((start + batch_size) / len(chunks), 1.0),
            text=f"Embedded {min(start + batch_size, len(chunks))}/{len(chunks)} chunks",
        )
    progress.empty()

    st.session_state.vectorstore = vectorstore
    st.session_state.indexed_files = [f.name for f in uploaded_files]
    st.success(f"Indexed {len(uploaded_files)} file(s) into {len(chunks)} chunks.")


def answer_question(query, k):
    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": max(10, k * 4), "lambda_mult": 0.5},
    )
    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)
    response = get_llm().invoke(PROMPT.invoke({"context": context, "question": query}))
    return response.content, docs


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("📄 Your documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs", type=["pdf"], accept_multiple_files=True
    )

    with st.expander("Settings"):
        chunk_size = st.slider("Chunk size", 200, 2000, 1000, step=100)
        chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, step=50)
        top_k = st.slider("Chunks retrieved per question", 1, 8, 4)

    if st.button("Process documents", type="primary", disabled=not uploaded_files):
        with st.spinner("Reading and indexing your PDFs..."):
            try:
                build_index(uploaded_files, chunk_size, min(chunk_overlap, chunk_size - 1))
            except Exception as e:
                st.error(f"Indexing failed: {e}")

    if st.session_state.indexed_files:
        st.markdown("**Indexed:**")
        for name in st.session_state.indexed_files:
            st.markdown(f"- {name}")

    if st.button("Clear documents & chat"):
        reset_index()
        st.rerun()

# --------------------------------------------------------------------- main
st.title("Chat with your PDFs")
st.caption("Upload PDFs in the sidebar, click **Process documents**, then ask questions.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['file']}**, page {s['page']}")
                    st.caption(s["text"])

ready = st.session_state.vectorstore is not None
query = st.chat_input(
    "Ask a question about your documents..." if ready else "Process a PDF first",
    disabled=not ready,
)

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, docs = answer_question(query, top_k)
            except Exception as e:
                answer, docs = f"Something went wrong: {e}", []
        st.markdown(answer)

        sources = [
            {
                "file": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page", 0) + 1,
                "text": d.page_content[:300] + ("..." if len(d.page_content) > 300 else ""),
            }
            for d in docs
        ]
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"**{s['file']}**, page {s['page']}")
                    st.caption(s["text"])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )