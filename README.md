# 📚 Paper‑RAG Assistant

A research‑oriented Retrieval‑Augmented Generation (RAG) system that lets you **search arXiv**, ingest **local PDFs**, and ask **conversational questions** over your personalised paper collection.
It pairs **LangGraph** state machines with **Google Gemini** models, **Chroma DB** vector storage and a lightweight **MySQL** catalogue, giving you a *batteries‑included* pipeline from raw PDF to high‑level insight.

---

## ✨ Key Features

| Capability                    | Description                                                                                                                            |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Multi‑source ingestion**    | ‑ Search and persist top‑*k* arXiv papers.‑ Drop any PDF (single file or folder) and auto‑extract **title + abstract** via heuristics. |
| **Structured metadata store** | All papers are catalogued into a MySQL table and simultaneously embedded into a Chroma collection (`papers`).                          |
| **LangGraph orchestration**   | An explicit state machine routes user queries to one of five intents: `search_arxiv`, `upload_pdf`, `local_query`, `compare`, `other`. |
| **Google Gemini integration** | • `models/gemini‑2.5‑flash‑preview‑04‑17` for chat tasks.• `models/embedding‑001` for text embeddings.                                 |
| **CLI**     | Use the built‑in terminal loop.                                                            |
| **Deterministic persistence** | Vector store is stored on disk (`./chroma_db`) so nothing is lost between sessions.                                                    |

---

---

## 🛠️ Prerequisites


Install Python dependencies:

```bash
pip install \
  langchain langgraph langchain-community chromadb arxiv \
  mysql-connector-python python-dotenv google-generativeai pdfminer.six \
  tiktoken pypdf streamlit
```

```txt
langchain==0.1.17
langgraph==0.0.26
chromadb==0.5.0
arxiv==2.1.0
mysql-connector-python==8.4.0
python-dotenv==1.0.1
google-generativeai==0.5.1
pdfminer.six==20221105
pypdf==4.2.0
tiktoken==0.6.0
```

---

## 🔐 Environment Variables

Create a **`.env`** in the project root (or export vars in your shell):

```env
# Google Gemini
GEMINI_API_KEY="<your‑key>"
GEMINI_MODEL="models/gemini-2.5-flash-preview-04-17"  # optional override

# MySQL connection
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=llmuser
MYSQL_PASSWORD=LLMproject_
MYSQL_DB=paper_db

# Chroma persistence directory
CHROMA_PERSIST_DIR=./chroma_db
```

> **Tip:** The script will auto‑create the `paper_db` database if it does not exist (given sufficient privileges).

---

## 🗄️ MySQL Schema

The table is created automatically on first run, shown here for reference:

```sql
CREATE TABLE papers (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title TEXT,
    abstract TEXT,
);
```

---

## 🚀 Quick Start (CLI)

```bash
python main.py
```

Then interact:

```text
Welcome to the paper‑RAG assistant. Type 'exit' to quit.
>>> upload pdf
📄 Please enter a full PDF file path, or a directory containing PDFs:
>> PDF file or folder path: ./pdf
📑 Processing: ./pdf/SiLK- Simple Learned Keypoints.pdf...
✅ Stored PDF as paper ID 1: SiLK: Simple Learned Keypoints Pierre Gleize
...

>>> Compare dedode and dedodev2
💬 **Comparison Report: DeDoDe vs. DeDoDe v2**
**Problem Statement**
*   **DeDoDe:** Addresses the challenge of keypoint detection for 3D reconstruction, specifically the ....
...
```

The CLI keeps only the last two turns in memory, making it suitable for long sessions without hitting token limits.

---

## 🖼️ Architecture Overview

```
┌───────────────────────────────┐
│           User CLI            │
└──────────────┬────────────────┘
               │  user_input
        ┌──────▼──────┐ intent
        │  Router     │─────────┐
        └──────┬──────┘         │
  search_arxiv │     upload_pdf │ compare/local/other
        ┌──────▼──────┐         │
        │  arXiv API  │         │
        └──────┬──────┘         │
               │ meta & text    │
        ┌──────▼──────┐         │
        │ MySQL +     │         │
        │  Chroma     │◄────────┘
        └──────┬──────┘
               │ RAG context
        ┌──────▼──────┐
        │  Gemini LLM │
        └─────────────┘
```

* **StateGraph**   – provides explicit edges between nodes for traceability.
* **Google Gemini** – used both for intent routing and final answer generation.
* **Chroma DB**     – fast cosine similarity search on embeddings.
* **MySQL**         – relational metadata store (source, ids, timestamps).

---

## 🏃‍♂️ Common Workflows

| Task                                     | Command                                                             | Notes                                                                     |
| ---------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Search latest papers on Arxiv | `>>> Arxiv search ...`                                                | Top‑3 arXiv hits are persisted automatically                              |
| Ingest a folder of local PDFs            | Choose `upload_pdf` intent or type `upload`                         | Titles & abstracts are auto‑extracted; fallback to file‑name if OCR fails |
| Ask a question about stored papers       | `>>> What are the main limitations of current 3D diffusion models?` | If nothing found, system will state so                                    |
| Compare approaches                       | `>>> compare How do SCORE‑based vs DDPM methods differ?`            | Produces a 5‑bullet structured report                                     |

---

## 🙏 Acknowledgements

* [LangChain](https://github.com/langchain-ai/langchain) + [LangGraph](https://github.com/normal-computing/langgraph)
* [ChromaDB](https://github.com/chroma-core/chroma)
* Google Gemini & GenerativeAI APIs
* The amazing [arXiv](https://arxiv.org) open access repository
