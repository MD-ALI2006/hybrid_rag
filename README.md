# 🔬 Hybrid RAG System

> **Production-Style Hybrid Retrieval-Augmented Generation for PDF Documents**
>
> SkillCraft Internship — Task 1

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Problem Statement](#-problem-statement)
3. [What is RAG?](#-what-is-rag)
4. [What is Hybrid RAG?](#-what-is-hybrid-rag)
5. [System Architecture](#-system-architecture)
6. [Technology Stack](#-technology-stack)
7. [Project Structure](#-project-structure)
8. [Installation Guide (Windows)](#-installation-guide-windows)
9. [How to Run](#-how-to-run)
10. [How to Use](#-how-to-use)
11. [Technical Concepts](#-technical-concepts)
12. [Evaluation Methodology](#-evaluation-methodology)
13. [Limitations](#-limitations)
14. [Troubleshooting](#-troubleshooting)
15. [Future Improvements](#-future-improvements)

---

## 🎯 Project Overview

This project is a **Hybrid Retrieval-Augmented Generation (RAG)** system that lets you:

1. **Upload** PDF research papers or technical documents
2. **Ask questions** about the content in natural language
3. **Get accurate answers** with page-level citations
4. **Compare** three different retrieval strategies side-by-side

The system processes your PDFs locally, finds the most relevant passages using multiple search methods, then uses Google Gemini to generate grounded answers.

---

## ❓ Problem Statement

Large language models (LLMs) are powerful, but they:
- **Hallucinate** — make up facts that sound convincing
- **Lack access** to your private documents
- **Cannot cite** specific pages or sections

RAG solves this by **retrieving** relevant information from your documents first, then giving that information to the LLM as context. The LLM answers based on the retrieved evidence, not from memory.

---

## 📖 What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that combines:

1. **Retrieval** — Finding relevant document chunks that relate to your question
2. **Generation** — Using an LLM to synthesize an answer from those chunks

```
User Question → Retrieve Relevant Chunks → Send to LLM → Answer with Citations
```

This ensures the answer is **grounded** in your actual documents rather than the model's training data.

---

## 🔀 What is Hybrid RAG?

Standard RAG uses only **one** retrieval method. **Hybrid RAG** combines multiple retrieval strategies:

| Method | Type | Strength |
|--------|------|----------|
| **BM25** | Keyword (lexical) | Exact terms, technical jargon |
| **Dense Retrieval** | Semantic (neural) | Meaning, paraphrases, conceptual similarity |
| **Hybrid** | Combined | Best of both worlds |

By fusing keyword and semantic search, Hybrid RAG catches documents that either method alone would miss.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     Streamlit UI                            │
│  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌─────────────┐  │
│  │Dashboard │ │ Upload   │ │   Chat    │ │ Comparison  │  │
│  └──────────┘ └──────────┘ └───────────┘ └─────────────┘  │
├────────────────────────────────────────────────────────────┤
│                     RAG Pipeline                            │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Document   │→ │  Chunking    │→ │  Index Building  │   │
│  │ Processor  │  │ (800 chars)  │  │  (BM25 + Vector) │   │
│  └────────────┘  └──────────────┘  └──────────────────┘   │
│                                                             │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   BM25     │  │   Dense      │  │  Hybrid Fusion   │   │
│  │ Retriever  │  │  Retriever   │  │  (RRF)           │   │
│  └────────────┘  └──────────────┘  └──────────────────┘   │
│                           │                                 │
│                    ┌──────────────┐                         │
│                    │  Reranker    │                         │
│                    │(Cross-Encoder)│                        │
│                    └──────────────┘                         │
│                           │                                 │
│                    ┌──────────────┐                         │
│                    │  Gemini LLM  │                         │
│                    │  (Answer +   │                         │
│                    │  Citations)  │                         │
│                    └──────────────┘                         │
└────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Frontend | Streamlit | Web UI |
| PDF Extraction | PyMuPDF (fitz) | Extract text with page numbers |
| Keyword Search | rank_bm25 | BM25 lexical retrieval |
| Semantic Search | Sentence Transformers | Dense embeddings |
| Vector Store | ChromaDB | Store and query embeddings |
| Reranker | BAAI/bge-reranker-base | Cross-encoder reranking |
| LLM | Google Gemini (via google-genai) | Answer generation |
| Visualization | Plotly | Interactive charts |
| Config | python-dotenv | Environment variables |

---

## 📁 Project Structure

```
hybrid_rag/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .env                        # API keys (not committed to git)
├── .env.example                # Template for .env
├── .gitignore                  # Git ignore rules
│
├── config/
│   └── settings.py             # All configuration in one place
│
├── src/
│   ├── document_processor.py   # PDF extraction & validation
│   ├── chunking.py             # Text chunking with metadata
│   ├── bm25_retriever.py       # BM25 keyword retrieval
│   ├── dense_retriever.py      # Semantic vector retrieval
│   ├── hybrid_retriever.py     # Hybrid fusion (RRF)
│   ├── reranker.py             # Cross-encoder reranking
│   ├── llm_service.py          # Gemini API integration
│   ├── rag_pipeline.py         # Orchestrates the full pipeline
│   ├── citations.py            # Citation extraction & verification
│   ├── evaluation.py           # Metrics and analysis
│   └── utils.py                # Shared utilities
│
├── ui/
│   ├── dashboard.py            # Dashboard page
│   ├── upload_page.py          # Upload & process page
│   ├── chat_page.py            # Chat interface
│   ├── comparison_page.py      # Retrieval comparison
│   └── evaluation_page.py      # Evaluation dashboard
│
├── data/
│   ├── uploads/                # Uploaded PDFs
│   └── processed/              # Processed data
│
├── vector_db/                  # ChromaDB persistent storage
│
└── tests/
    ├── test_chunking.py        # PDF & chunking tests
    ├── test_retrieval.py       # Retrieval tests
    └── test_citations.py       # Citation tests
```

---

## 💻 Installation Guide (Windows)

### Prerequisites

- **Python 3.11** installed ([python.org](https://www.python.org/downloads/))
- **Google AI Studio API key** ([aistudio.google.com](https://aistudio.google.com/apikey))

### Step-by-Step Setup

#### 1. Open PowerShell / Command Prompt

Navigate to the project folder:

```powershell
cd C:\Users\YourName\Desktop\hybrid_rag
```

#### 2. Create a Virtual Environment

```powershell
python -m venv venv
```

#### 3. Activate the Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

> If you get an execution policy error, run:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

#### 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

This installs all required libraries. The first run will also download AI models (~100 MB for the embedding model, ~1 GB for the reranker).

#### 5. Set Up Your Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Create a new API key
3. Open the `.env` file in the project root
4. Replace `your_api_key_here` with your actual key:

```
GEMINI_API_KEY=AIzaSy...your_key_here
```

> ⚠️ **Never share your API key or commit it to git.**

---

## 🚀 How to Run

```powershell
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

The application opens in your browser at `http://localhost:8501`.

---

## 📖 How to Use

### 1. Upload Documents

- Go to **📄 Upload & Process** in the sidebar
- Drag and drop one or more PDF files
- Click **🚀 Process Documents**
- Wait for extraction, chunking, and indexing to complete

### 2. Ask Questions

- Go to **💬 Chat** in the sidebar
- Select a retrieval mode (Hybrid recommended)
- Type your question in the chat input
- Read the answer with page citations
- Expand sources to see the original text

### 3. Compare Retrieval Methods

- Go to **🔍 Retrieval Comparison**
- Enter a question
- Click **🔎 Compare**
- See side-by-side results from BM25, Dense, and Hybrid
- Review overlap analysis and score charts

### 4. Review Evaluation

- Go to **📈 Evaluation**
- View timing breakdowns, chunk counts, and per-query metrics

---

## 🔬 Technical Concepts

### BM25 (Best Matching 25)

BM25 is a **keyword-based** scoring function. It ranks documents by how often query words appear in them, adjusted for document length.

**Strengths:** Exact term matching, technical jargon, acronyms.
**Weaknesses:** Cannot understand synonyms or paraphrases.

### Dense Retrieval

Dense retrieval converts both the query and documents into **numerical vectors** (embeddings) using a neural network. Documents with similar meaning have similar vectors.

**Model used:** `sentence-transformers/all-MiniLM-L6-v2` (a compact, fast model that runs locally).

**Strengths:** Understands meaning, handles paraphrases.
**Weaknesses:** May miss exact technical terms.

### Hybrid Retrieval

Combines BM25 and Dense retrieval to get the best of both. The key challenge is merging two ranked lists with incompatible scores.

### Reciprocal Rank Fusion (RRF)

RRF is the fusion method used to combine BM25 and Dense results:

```
RRF_score(document) = 1/(k + BM25_rank) + 1/(k + Dense_rank)
```

Where `k = 60` (a smoothing constant). This gives every document a comparable score regardless of which retriever found it.

**Why RRF?** Raw BM25 and cosine similarity scores are on different scales. Simply adding them would be meaningless. RRF uses ranks instead of scores, making fusion fair.

### Cross-Encoder Reranking

After hybrid retrieval, a **cross-encoder** (BAAI/bge-reranker-base) re-scores each (query, chunk) pair. Unlike the embedding model that encodes query and document separately, the cross-encoder processes them together for higher accuracy.

**Model used:** `BAAI/bge-reranker-base` — a 278M parameter model that runs locally.

### Citation Mechanism

1. Retrieved chunks carry metadata: document name, page number, section title.
2. The LLM prompt instructs Gemini to cite using `[Source: filename, Page X]`.
3. After generation, citations are **extracted** from the answer text with regex.
4. Each citation is **verified** against the actually retrieved chunks.
5. Unverified citations are flagged.

---

## 📊 Evaluation Methodology

This system provides **exploratory retrieval analysis** — real timing and retrieval statistics from your session.

### Measured Metrics

- Retrieval time per method (BM25, Dense, Hybrid)
- Reranking time
- LLM response time
- Total end-to-end time
- Number of chunks retrieved vs. reranked
- Overlap between retrieval methods

### Formal IR Metrics (Available with Ground Truth)

If you provide a labeled dataset with query-relevance pairs:

- **Precision@K** — fraction of top-K results that are relevant
- **Recall@K** — fraction of relevant results found in top-K
- **MRR** — Mean Reciprocal Rank
- **Hit Rate@K** — whether any relevant result appears in top-K

> ⚠️ Without ground-truth labels, the system does NOT claim one method is objectively better. All comparisons are exploratory.

---

## ⚠️ Limitations

1. **Scanned PDFs** — Cannot extract text from image-only PDFs (OCR not included).
2. **Model download** — First run downloads ~1.3 GB of models (embedding + reranker).
3. **API dependency** — Answer generation requires a working Google Gemini API key and internet connection.
4. **Context window** — Very large documents may produce many chunks; only the top reranked chunks are sent to the LLM.
5. **Section detection** — Heading detection is heuristic and may not catch all section titles.
6. **Language** — Optimized for English documents.

---

## 🔧 Troubleshooting

### "GEMINI_API_KEY not configured"

Edit `.env` and add your key: `GEMINI_API_KEY=AIzaSy...`

### "Embedding model failed to load"

Ensure you have internet on first run (the model downloads once). Check disk space (~500 MB needed).

### "API quota exceeded"

Google Gemini free tier has rate limits. Wait a minute and try again, or use a paid API key.

### "No extractable text found"

The PDF may be scanned (image-only). This system requires PDFs with selectable text.

### Virtual environment activation fails

Run: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

### Streamlit port conflict

Run with a different port: `streamlit run app.py --server.port 8502`

---

## 🚀 Future Improvements

- OCR integration for scanned PDFs (Tesseract / EasyOCR)
- Conversation memory across sessions
- Multi-modal support (images, tables in PDFs)
- Custom evaluation dataset builder
- Export answers as reports
- Support for more document formats (DOCX, TXT)
- GPU acceleration for faster embedding and reranking
- Advanced chunking strategies (semantic chunking)

---

## 📝 Privacy Note

- **Document processing** (extraction, chunking, embedding) happens **entirely on your machine**.
- **Only the retrieved text chunks** (not the full documents) are sent to Google Gemini for answer generation.
- Your API key is stored locally in `.env` and never transmitted anywhere except to Google's API.

---

> Built as part of the SkillCraft Technology Internship Program.
