<<<<<<< HEAD
# 🧬 The Clinical Oracle
### NIH Clinical Intelligence System — RAG Agent for Clinical Trial Protocols

> Ask complex clinical questions. Get answers grounded in NIH protocol documents — with sources, similarity scores, and LLM-as-Judge evaluation.

---

## 🎯 What it does

**The Clinical Oracle** is an Agentic RAG system built on top of NIH clinical trial protocol PDFs. It allows clinical researchers to query dense protocol documents in natural language and get:

- ✅ Answers grounded exclusively in the retrieved context (no hallucinations)
- ✅ Source citations with similarity scores per chunk
- ✅ Real-time LLM-as-Judge quality evaluation (Faithfulness, Relevance, Completeness, Citation)
- ✅ Session archiving and full report download

---

## 🏗️ Architecture

```
PDF Documents (NIH Protocols)
        ↓
  Unstructured (PDF parsing)
        ↓
  RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
        ↓
  all-MiniLM-L6-v2 Embeddings
        ↓
  ChromaDB Vector Store
        ↓
  Semantic Retrieval (k=12 chunks)
        ↓
  Mistral-small-latest (Answer Generation)
        ↓
  LLM-as-Judge (Quality Evaluation)
        ↓
  Streamlit UI
```

---

## 📊 Evaluation Results (LLM-as-Judge)

| Metric | Score |
|---|---|
| Faithfulness | 9.4/10 |
| Relevance | 8.8/10 |
| Completeness | 9.0/10 |
| Citation | 8.0/10 |
| **Overall** | **8.8/10** |

---

## 🚀 Quick Start — Local

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/clinical-oracle.git
cd clinical-oracle
```

### 2. Set up environment

```bash
pip install -r requirements.txt
```

### 3. Configure API key

Create a `.env` file at the root:

```
MISTRAL_API_KEY=your_mistral_api_key_here
```

### 4. Add your PDF documents

Place your NIH clinical trial PDF files in a `docs/` folder:

```
clinical-oracle/
├── docs/
│   ├── NCT06145295_Prot_SAP_000.pdf
│   ├── NCT06152900_Prot_SAP_000.pdf
│   └── ...
```

### 5. Parse PDFs and build the vector database

Run the notebook `Clinical_Oracle_FINAL.ipynb` cells 1 to 5, or run the parsing script directly.

The `chroma_db/` and `docs_txt/` folders will be created automatically.

### 6. Launch the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🐳 Docker Deployment

### Build

```bash
docker build -t clinical-oracle .
```

### Run

```bash
docker run -p 8501:8501 \
  -e MISTRAL_API_KEY=your_key_here \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v $(pwd)/docs_txt:/app/docs_txt \
  clinical-oracle
```

> ⚠️ The `chroma_db/` and `docs_txt/` folders must exist before running Docker — generate them from the notebook first.

Open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
clinical-oracle/
├── app.py                          # Streamlit application
├── Clinical_Oracle_FINAL.ipynb     # Full RAG pipeline notebook
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration
├── .env                            # API keys (not committed)
├── logo.png                        # App logo (optional)
├── archives_oracle.json            # Session archives (auto-generated)
├── docs/                           # Raw PDF files (not committed)
├── docs_txt/                       # Parsed text files (auto-generated)
└── chroma_db/                      # Vector database (auto-generated)
```

---

## ⚙️ Configuration

| Parameter | Default | Description |
|---|---|---|
| `k` chunks | 12 | Number of chunks retrieved per query |
| Chunk size | 1000 | Characters per chunk |
| Chunk overlap | 200 | Overlap between chunks |
| Embedding model | `all-MiniLM-L6-v2` | Sentence transformer model |
| LLM | `mistral-small-latest` | Generation model |
| Temperature | 0 | Deterministic output |

---

## 🔑 Requirements

- Python 3.11+
- Mistral API key — get one at [console.mistral.ai](https://console.mistral.ai)
- NIH clinical trial protocol PDFs

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Mistral AI (mistral-small-latest) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | ChromaDB |
| PDF Parsing | Unstructured |
| RAG Framework | LangChain |
| UI | Streamlit |
| Evaluation | LLM-as-Judge (custom) |

---

## 📝 .gitignore

Add this to avoid committing large files:

```
.env
chroma_db/
docs/
docs_txt/
archives_oracle.json
__pycache__/
*.pyc
```
=======
---
title: Clinical Oral RAG
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Streamlit template space
---

# Welcome to Streamlit!

Edit `/src/streamlit_app.py` to customize this app to your heart's desire. :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).
>>>>>>> a1ba6223919726b0aaba3be9d5a3abb83420de02
