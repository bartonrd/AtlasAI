
import os
import re
import streamlit as st

# Loaders: PDF + DOCX
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

# Splitter (v1 package)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings (v1 package)
from langchain_huggingface import HuggingFaceEmbeddings

# Vector store
from langchain_community.vectorstores import FAISS

# Legacy chain (still supported via langchain_classic)
from langchain_classic.chains import RetrievalQA

# Prompt (v1 package)
from langchain_core.prompts import PromptTemplate

# Local HF LLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

# ---------------------------
# Configuration (edit paths)
# ---------------------------
# Get the script directory and construct relative paths to documents folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(SCRIPT_DIR, "documents")

PDF_PATHS = [
    os.path.join(DOCUMENTS_DIR, "distribution_model_manager_user_guide.pdf"),
    os.path.join(DOCUMENTS_DIR, "adms-16-20-0-modeling-overview-and-converter-user-guide.pdf"),
]
DOCX_PATHS = [
    # Add DOCX files from documents folder if needed
]

# Local, offline models
EMBEDDING_MODEL = r"C:\models\all-MiniLM-L6-v2"
LOCAL_TEXT_GEN_MODEL = r"C:\models\flan-t5-base"  # or flan-t5-small for faster CPU runs

TOP_K = 2  # number of chunks retrieved (reduced to avoid token limit issues)
CHUNK_SIZE = 800  # reduced from 1000 to fit within model token limits
CHUNK_OVERLAP = 150

# ---------------------------
# Helpers
# ---------------------------
def strip_boilerplate(text: str) -> str:
    """
    Optional: remove common footer/header boilerplate that may appear in PDFs,
    e.g., 'Proprietary - See Copyright Page' or repeated document titles.
    Keeps function conservative: only strips obvious boilerplate phrases.
    """
    if not text:
        return text
    patterns = [
        r"\bProprietary\s*-\s*See\s*Copyright\s*Page\b",
        r"\bContents\b",
        r"\bADMS\s*[\d\.]+\s*Modeling\s*Overview\s*and\s*Converter\s*User\s*Guide\b",
        r"\bDistribution\s*Model\s*Manager\s*User\s*Guide\b",
        r"^\s*Page\s*\d+\s*$",
    ]
    cleaned = text
    for pat in patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
    # Remove extra spaces due to deletions
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned

def to_bullets(text: str, min_items: int = 3, max_items: int = 10) -> str:
    """
    Convert a free-form LLM answer into Markdown bullets (one per line).
    - Splits on common bullet glyphs (• ▪ ● ·) and line-start hyphens.
    - Falls back to sentence splitting if no glyphs are found.
    - Strips leading bullet/numbering characters.
    - Enforces 3–10 bullets when possible.
    """
    if not text:
        return ""

    # Optional: remove boilerplate if the model echoed it
    text = strip_boilerplate(text)

    # Normalize whitespace for the fallback splitter
    normalized = re.sub(r"\s+", " ", text).strip()

    # Primary: split on explicit bullet glyphs or line-start hyphens/newlines
    parts = re.split(
        r"(?:\n+|[•▪●·]\s+|(?:^|\s)-\s+)",
        text,
        flags=re.UNICODE
    )
    parts = [p.strip() for p in parts if p and p.strip()]

    # Fallback: sentence splitting if bullets weren't found
    if len(parts) <= 1:
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalized)
        parts = [p.strip() for p in parts if p and p.strip()]

    # Final cleanup: strip any leading bullets or numbering
    cleaned = []
    for p in parts:
        p = re.sub(r"^[•\-–\*\u2022\u25AA\u25CF\u00B7]+\s*", "", p)            # bullets
        p = re.sub(r"^(?:\d+|[A-Za-z])[\.\)\:]\s*", "", p)                     # 1. 2) a) etc.
        p = p.strip()
        if p:
            cleaned.append(p)

    # If still too few, try splitting by '•' or semicolons (common in inline lists)
    if len(cleaned) < min_items:
        if "•" in text:
            cleaned = [re.sub(r"^[•\-\*\s]+", "", p).strip() for p in text.split("•") if p.strip()]
        elif ";" in text:
            cleaned = [re.sub(r"^[•\-\*\s]+", "", p).strip() for p in text.split(";") if p.strip()]

    # Emit Markdown bullets, capped
    bullets = [f"- {p}" for p in cleaned[:max_items]]
    return "\n".join(bullets)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Ask a question")

# Drag-drop upload for PDF/DOCX
uploaded_files = st.file_uploader(
    "Upload PDF or DOCX files (optional)",
    type=["pdf", "docx"],
    accept_multiple_files=True,
)

temp_dir = os.path.join(os.getcwd(), "tmp_docs")
os.makedirs(temp_dir, exist_ok=True)

if uploaded_files:
    for uf in uploaded_files:
        save_path = os.path.join(temp_dir, uf.name)
        with open(save_path, "wb") as f:
            f.write(uf.read())
        ext = os.path.splitext(save_path)[1].lower()
        if ext == ".pdf":
            PDF_PATHS.append(save_path)
        elif ext == ".docx":
            DOCX_PATHS.append(save_path)

prompt_text = st.chat_input("Pass Your Prompt here")

if prompt_text:
    st.chat_message("user").markdown(prompt_text)

    # ---------------------------
    # 1) Load documents (PDF + DOCX)
    # ---------------------------
    docs = []
    missing = []

    for p in PDF_PATHS:
        if not os.path.exists(p):
            missing.append(p)
        else:
            try:
                docs.extend(PyPDFLoader(p).load())
            except Exception as e:
                st.warning(f"Failed to read PDF {p}: {e}")

    for p in DOCX_PATHS:
        if not os.path.exists(p):
            missing.append(p)
        else:
            try:
                docs.extend(Docx2txtLoader(p).load())
            except Exception as e:
                st.warning(f"Failed to read DOCX {p}: {e}")

    if missing:
        st.error("The following files were not found:\n- " + "\n- ".join(missing))
        st.stop()
    if not docs:
        st.error("No documents loaded. Please add PDF/DOCX paths or upload files.")
        st.stop()

    # ---------------------------
    # 2) Split documents
    # ---------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    splits = splitter.split_documents(docs)
    if not splits:
        st.error("No text chunks produced. If files are scanned images, run OCR first.")
        st.stop()

    # ---------------------------
    # 3) Embeddings + FAISS
    # ---------------------------
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _ = embeddings.embed_query("probe")  # quick check
    except Exception as e:
        st.error(f"Embedding model failed: {e}")
        st.stop()

    try:
        vectorstore = FAISS.from_documents(splits, embeddings)
    except IndexError:
        st.error("Embedding list was empty—ensure files have extractable text.")
        st.stop()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # ---------------------------
    # 4) Local Hugging Face pipeline LLM (deterministic for clean bullets)
    # ---------------------------
    try:
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_TEXT_GEN_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_TEXT_GEN_MODEL)
        gen_pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=384,  # tighter output
            do_sample=False,     # deterministic; avoids temperature warnings for T5
            truncation=True,     # truncate inputs that exceed max length
            max_length=512,      # FLAN-T5 max input length
            # For variety:
            # do_sample=True, temperature=0.2, top_p=0.9, top_k=50
        )
        llm = HuggingFacePipeline(pipeline=gen_pipe)
    except Exception as e:
        st.error(f"Failed to load local HF model from '{LOCAL_TEXT_GEN_MODEL}': {e}")
        st.stop()

    # ---------------------------
    # 5) Retrieval-aware prompt (forces bullet points & newlines)
    # ---------------------------
    template = """You are a concise, helpful assistant for a RAG system.

Rules:
- If the question is unrelated to the context, reply briefly to the user without using the context.
- Otherwise, answer using ONLY the retrieved context.
- FORMAT your final answer as 3–7 Markdown bullet points.
- Each bullet MUST start with "- " and be followed by a newline.
- Keep each bullet to one sentence. No preface, no closing remarks—bullets only.

Question:
{question}

Context:
{context}

Answer (markdown bullets only):
"""
    qa_prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=template,
    )

    # ---------------------------
    # 6) RetrievalQA (classic chain API)
    # ---------------------------
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",       # for larger corpora consider "map_reduce"
        retriever=retriever,
        chain_type_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
    )

    # ---------------------------
    # 7) Run the chain (use invoke to avoid deprecation warning)
    # ---------------------------
    result = qa.invoke({"query": prompt_text})
    answer = result.get("result", "")
    sources = result.get("source_documents", [])

    # ---------------------------
    # 8) Display (enforce bullets)
    # ---------------------------
    st.chat_message("assistant").markdown(to_bullets(answer))

    if sources:
        with st.expander("Sources"):
            for i, doc in enumerate(sources, start=1):
                meta = doc.metadata or {}
                page = meta.get("page", "unknown")
                source = meta.get("source", "unknown")
                st.write(f"{i}. {source} (page {page})")
