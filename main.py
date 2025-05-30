"""
---------------------------------------------------------------------
Prerequisites
-------------
```bash
pip install langchain langgraph langchain-community chromadb arxiv mysql-connector-python \
            python-dotenv google-generativeai pdfminer.six tiktoken \
            pypdf streamlit
```

Environment variables (e.g. via `.env`):
```
GEMINI_API_KEY="<your‑key>"
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=password
MYSQL_DB=paper_db
CHROMA_PERSIST_DIR=./chroma_db
```

*MySQL* schema (automatically created on first run):
```sql
CREATE TABLE papers (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title TEXT,
    abstract TEXT,
    source ENUM('arxiv','local'),
    arxiv_id VARCHAR(32) NULL,
    pdf_path TEXT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

"""

import os
import re
os.environ.update({
    "GEMINI_API_KEY": "XXX",
    "MYSQL_HOST": "XXX.XXX.XXX.XXX",
    "MYSQL_PORT": "XXXX",
    "MYSQL_USER": "XXX",
    "MYSQL_PASSWORD": "XXX",
    "MYSQL_DB": "XXX",
    "CHROMA_PERSIST_DIR": "./chroma_db"
})

import fitz
import tempfile
import logging
from typing import Dict, List, Tuple

import arxiv  # arXiv API wrapper
import chromadb
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

import mysql.connector as mysql

# import google.generativeai as genai
# genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# models = genai.list_models()
# for m in models:
#     print(f"✅ {m.name}")
#     print(f"  supports generation: {'generateContent' in m.supported_generation_methods}")
#     print(f"  generation methods: {m.supported_generation_methods}")
#     print()

# ---------------------------------------------------------------------------
# 0. Environment & globals
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG‑ARXIV")
name = "models/gemini-2.5-flash-preview-04-17"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", name)

# Create the LLM & embedding objects once (they are thread‑safe)
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.2,
)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",
                                  google_api_key=os.environ["GEMINI_API_KEY"])

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
vector_store = Chroma(
    client=chroma_client,
    collection_name="papers",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

# MySQL connection helper ----------------------------------------------------

def get_mysql_conn():
    return mysql.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", 3306)),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DB", "paper_db"),
        autocommit=True,
    )

# Ensure table exists
with get_mysql_conn() as _conn:
    cur = _conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS papers (
            id INT AUTO_INCREMENT PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            source ENUM('arxiv','local'),
            arxiv_id VARCHAR(32) NULL,
            pdf_path TEXT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
    )

# Utility --------------------------------------------------------------------

def store_paper(title: str, abstract: str, source: str, *, arxiv_id=None, pdf_path=None):
    """Persist metadata to MySQL and vector store; returns paper_id."""
    with get_mysql_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO papers (title, abstract)
            VALUES (%s, %s)
            """,
            (title, abstract),
        )
        paper_id = cur.lastrowid

    # Add to vector store (ID is MySQL id string)
    vector_store.add_texts(
        texts=[title, abstract],
        ids=[f"{paper_id}_title", f"{paper_id}_abstract"],
        metadatas=[{"id": str(paper_id)}, {"id": str(paper_id)}]
    )


    vector_store.persist()
    return paper_id


def search_mysql_by_ids(ids: List[str]) -> List[Tuple[int, str, str]]:
    if not ids:
        return []
    with get_mysql_conn() as conn:
        cur = conn.cursor()
        format_ids = ",".join(["%s"] * len(ids))
        cur.execute(
            f"SELECT id, title, abstract FROM papers WHERE id IN ({format_ids})",
            ids,
        )
        return cur.fetchall()

def extract_title_and_abstract(pdf_path):
    def extract_title_by_font_consistency(first_page):
        blocks = first_page.get_text("dict")["blocks"]

        title_lines = []
        initial_font_size = None
        found_first_text = False

        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    font_size = span["size"]
                    
                    if not found_first_text:
                        initial_font_size = font_size
                        found_first_text = True

                    # 若字體大小與初始相同，繼續視為標題
                    if abs(font_size - initial_font_size) < 4:
                        title_lines.append(text)
                    else:
                        # 字體大小不同，視為進入新段落
                        return " ".join(title_lines).strip()

        return " ".join(title_lines).strip()

    def extract_abstract_block(lines):
        abstract_started = False
        abstract_lines = []

        for i, line in enumerate(lines):
            line_lower = line.lower().strip()

            # Step 1: Abstract 起點
            if not abstract_started and "abstract" in line_lower:
                abstract_started = True
                continue

            # Step 2: Abstract 結束條件
            if abstract_started:
                # Look ahead to下一行是否為 "introduction"
                if i + 1 < len(lines):
                    next_line = lines[i+1].strip().lower()
                    if line in ["1", "i", "1.", "i."] and next_line == "introduction":
                        break
                    if re.match(r"^(1\.?\s*)?(i\.?\s*)?introduction$", line_lower):
                        break
                abstract_lines.append(line.strip())

        return " ".join(abstract_lines)

    doc = fitz.open(pdf_path)

    title = extract_title_by_font_consistency(doc[0])
    lines = []
    for page in doc:
        lines += page.get_text().split('\n')
        if len(lines) > 100:
            break
    lines = [line.strip() for line in lines if line.strip()]

    abstract = extract_abstract_block(lines)

    return title, abstract


# ---------------------------------------------------------------------------
# 1. Graph State & Intent classification
# ---------------------------------------------------------------------------
class RAGState(Dict):
    """Graph state object used by LangGraph."""
    user_input: str
    intent: str  # one of search_arxiv | upload_pdf | local_query | compare | other
    arxiv_results: List[Dict]
    pdf_info: Dict
    rag_answer: str
    history: List

# Intent detection prompt -----------------------------------------------------

INTENT_PROMPT = PromptTemplate.from_template(
    """You are a routing assistant. The user said: \"{question}\".
    Classify the user's intent into EXACTLY one of the following keys:
    1. search_arxiv  they want to search arXiv.
    2. upload_pdf    they uploaded or want to upload a PDF.
    3. local_query   they want information from already‑stored papers.
    4. compare       they want a comparative report across papers.
    5. other         anything else.

    Reply with ONLY the intent key (no extra words)."""
)

def classify_intent(state: RAGState) -> RAGState:
    question = state["user_input"]
    response = llm.invoke(INTENT_PROMPT.format(question=question))
    intent = response.content.strip().lower()

    if intent not in {
        "search_arxiv",
        "upload_pdf",
        "local_query",
        "compare",
        "other",
    }:
        intent = "other"
    state["intent"] = intent
    return state

# ---------------------------------------------------------------------------
# 2. Node implementations
# ---------------------------------------------------------------------------

def node_search_arxiv(state: RAGState) -> RAGState:
    """Search arXiv, persist, return top 3 results."""
    query = state["user_input"]
    history = state["history"]
    history.append(("human", query))
    # print(query)
    search = arxiv.Search(query=query, max_results=3, sort_by=arxiv.SortCriterion.Relevance)
    results = []
    for result in search.results():
        paper_id = store_paper(
            title=result.title,
            abstract=result.summary,
            source="arxiv",
            arxiv_id=result.get_short_id(),
        )
        results.append({
            "db_id": paper_id,
            "arxiv_id": result.get_short_id(),
            "title": result.title,
            "abstract": result.summary,
        })
    state["arxiv_results"] = results
    state["history"] = history
    return state


def node_handle_pdf(state: RAGState) -> RAGState:
    import glob
    query = state["user_input"]
    history = state["history"]
    history.append(("human", query))
    print("📄 Please enter a full PDF file path, or a directory containing PDFs:")
    path_input = input(">> PDF file or folder path: ").strip()

    if not os.path.exists(path_input):
        raise ValueError(f"❌ Path does not exist: {path_input}")

    pdf_paths = []

    if os.path.isdir(path_input):
        pdf_paths = glob.glob(os.path.join(path_input, "*.pdf"))
    elif os.path.isfile(path_input) and path_input.lower().endswith(".pdf"):
        pdf_paths = [path_input]
    else:
        raise ValueError("❌ Input must be a PDF file or a folder containing PDFs")

    if not pdf_paths:
        raise ValueError("❌ No PDF files found in the specified location.")

    results = []

    for pdf_path in pdf_paths:
        print(f"📑 Processing: {pdf_path}")
        try:
            title, abstract = extract_title_and_abstract(pdf_path)
            paper_id = store_paper(title, abstract, source="local", pdf_path=pdf_path)
            results.append({"db_id": paper_id, "title": title, "abstract": abstract})

        except Exception as e:
            print(f"⚠️ Failed to process {pdf_path}: {e}")

    if results:
        state["pdf_info"] = results
    else:
        state["pdf_info"] = {"error": "No valid PDFs processed."}

    state["history"] = history
    return state


def node_local_query(state: RAGState) -> RAGState:

    history = state["history"]
    """Similarity search vector store → RAG answer."""
    print("Please enter keyword of the paper you want to search in local database: ")
    path_input = input(">> Keyword of the paper: ").strip()

    docs_and_scores = vector_store.similarity_search_with_score(path_input, k = 1)
    # print(docs_and_scores)
    ids = [doc.metadata.get("id") for doc, _ in docs_and_scores if "id" in doc.metadata]
    rows = search_mysql_by_ids(ids)
    
    print("Here is the paper store in local database: ")
    print([f"TITLE: {r[1]}\nABSTRACT: {r[2]}" for r in rows])

    context = "\n\n".join([f"TITLE: {r[1]}\nABSTRACT: {r[2]}" for r in rows])
    # answer_prompt = PromptTemplate.from_template(
    #     """
    #     You are a research assistant helping to summarize academic papers.
    #     ONLY use the information from the context below.

    #     Context:
    #     {context}
    #     """
    # )
    # prompt_str = answer_prompt.format(context=context)
    answer_prompt = """ 
        You are a research assistant helping to summarize academic papers.
        ONLY use the information from the context below.

        Context:
        {context}
        """
    # message = [
    #     ("human", answer_prompt.format(context=context))
    # ]
    history.append(("human", answer_prompt.format(context=context)))
    # print(message)
    answer = llm.invoke(history).content
    history.append(("assistant", answer))

    state["rag_answer"] = answer
    state["history"] = history
    return state


def node_compare(state: RAGState) -> RAGState:

    history = state["history"]
    """Compare multiple papers based on similarity search in the vector store."""
    question = state["user_input"]

    docs_and_scores = vector_store.similarity_search_with_score(question, k=3)
    ids = [doc.metadata.get("id") for doc, _ in docs_and_scores if "id" in doc.metadata]
    rows = search_mysql_by_ids(ids)

    if not rows:
        state["rag_answer"] = "I couldn't find any relevant papers to compare."
        return state

    # 組成上下文
    context = "\n\n".join([f"TITLE: {r[1]}\nABSTRACT: {r[2]}" for r in rows])

    compare_prompt = PromptTemplate.from_template(
        """
        You are a research assistant.
        Using the paper summaries below, generate a structured comparison report focusing on:
        - Problem statement
        - Methodology
        - Results
        - Limitations
        - Key differences between papers

        Only use the context below. Do not make up content.

        Context:
        {context}

        Question: {question}
        Comparison:
        """
    )
    # prompt_str = compare_prompt.format(context=context, question=question)
    history.append(("human", compare_prompt.format(context=context, question=question)))

    answer = llm.invoke(history).content
    history.append(("assistant", answer))

    state["rag_answer"] = answer
    state["history"] = history
    return state


def node_other(state: RAGState) -> RAGState:
    state["rag_answer"] = llm.invoke(state["user_input"]).content
    return state

# ---------------------------------------------------------------------------
# 3. Build LangGraph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(RAGState)
    graph.add_node("router", classify_intent)
    graph.add_node("search_arxiv", node_search_arxiv)
    graph.add_node("upload_pdf", node_handle_pdf)
    graph.add_node("local_query", node_local_query)
    graph.add_node("compare", node_compare)
    graph.add_node("other", node_other)

    # ✅ 這一行是必要的：設定入口
    graph.set_entry_point("router")

    # ✅ 用 add_conditional_edges 做 intent routing
    graph.add_conditional_edges(
        source="router",
        path=lambda s: s["intent"],
        path_map={
            "search_arxiv": "search_arxiv",
            "upload_pdf": "upload_pdf",
            "local_query": "local_query",
            "compare": "compare",
            "other": "other",
        }
    )

    # 所有終點
    for node_key in ["search_arxiv", "upload_pdf", "local_query", "compare", "other"]:
        graph.add_edge(node_key, END)

    return graph.compile()

compiled_graph = build_graph()

# ---------------------------------------------------------------------------
# 4. Tiny CLI helper 
# ---------------------------------------------------------------------------
def run_cli():
    print("Welcome to the paper‑RAG assistant. Type 'exit' to quit.")
    history = []
    while True:
        # print(history)
        try:
            msg = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg.lower() in {"exit", "quit"}:
            break
        out_state: RAGState = compiled_graph.invoke({"user_input": msg, "history": history})
        history = out_state.get("history", history)
        if len(history) > 2:
            history = history[-2:]
        print(history)
        
        if out_state.get("arxiv_results"):
            message = ""
            for p in out_state["arxiv_results"]:
                print(f"[{p['db_id']}] {p['title']}\n{p['abstract'][:300]}...\n")
                message += f"[{p['db_id']}] {p['title']}\n{p['abstract']}\n"
            
            history.append(("assistant", message))

        elif out_state.get("pdf_info"):
            infos = out_state["pdf_info"]
            if isinstance(infos, dict) and "error" in infos:
                print("❌", infos["error"])
            elif isinstance(infos, list):
                message = ""
                for p in infos:
                    print(f"✅ Stored PDF as paper ID {p.get('db_id')}: {p.get('title')}")
                    message += f"title: {p.get('title')}, abstract: {p.get('abstract')}\n"
                history.append(("assistant", message))
            else:
                print("⚠️ Unexpected pdf_info format:", infos)

        elif out_state.get("compare"):
            print("📊 Comparative analysis:\n")
            print(out_state.get("rag_answer"))

        elif out_state.get("local_query"):
            print("📖 Answer based on local papers:\n")
            print(out_state.get("rag_answer"))

        else:  # fallback (e.g. other/general)
            print("💬", out_state.get("rag_answer"))


if __name__ == "__main__":
    run_cli()

