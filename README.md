
# AI Document Assistant using RAG

An AI-powered document assistant that allows users to upload documents and ask questions using Retrieval-Augmented Generation (RAG).

The application retrieves relevant information from uploaded documents and generates context-aware answers using open-source language models.

## Project Objectives

- Upload and process multiple document formats.
- Extract text from PDF, DOCX, TXT, and Markdown files.
- Split documents into smaller chunks.
- Generate embeddings using open-source embedding models.
- Store and retrieve document vectors using ChromaDB.
- Generate answers using a local LLM.
- Display source documents and citations with answers.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Document Processing | PyPDF, python-docx |
| Text Chunking | Custom Python module |
| Embedding Model | Sentence Transformers |
| Vector Database | ChromaDB |
| LLM | Open-source model |
| Version Control | Git and GitHub |

## Project Architecture

```text
User
 |
 v
Document Upload
 |
 v
Document Loader
 |
 v
Text Chunking
 |
 v
Embedding Generation
 |
 v
ChromaDB Vector Store
 |
 v
Similarity Search
 |
 v
LLM Answer Generation
 |
 v
Answer with Source Citations
```

## Project Structure

```text
ai-document-assistant/
|
|-- src/
|   |-- __init__.py
|   |-- document_loader.py
|   |-- text_chunker.py
|
|-- data/
|   |-- sample.md
|
|-- tests/
|   |-- __init__.py
|   |-- test_document_loader.py
|   |-- test_text_chunker.py
|   |-- test_pipeline_loading.py
|
|-- requirements.txt
|-- .gitignore
|-- README.md
```

## Implementation Progress

### Step 1: Document Loading and Text Extraction

**Objective:** Extract text from different document formats and convert it into a common structured format.

**Supported formats:**
- PDF
- DOCX
- TXT
- Markdown

**Libraries used:**
- pypdf
- python-docx

**Implementation:**

Created `src/document_loader.py` with the following functions:

- `load_pdf()` – extracts text page by page.
- `load_text_file()` – reads TXT and Markdown files.
- `load_docx()` – extracts Word document paragraphs.
- `load_document()` – identifies the file format and selects the appropriate loader.

**Important learning:**

1. Different document formats require different extraction methods.
2. PDF text extraction can preserve page-level information.
3. Document metadata is important for source citations.
4. A common document structure makes the RAG pipeline modular.

**Testing:**

```powershell
python -m tests.test_document_loader
```

**Status:** Completed.

### Step 2: Text Chunking

**Objective:** Divide extracted documents into smaller text passages for efficient retrieval.

**Concepts learned:**

- Chunk size
- Chunk overlap
- Word boundary preservation
- Metadata preservation

**Implementation:**

Created `src/text_chunker.py` with:

- `split_text()` – splits text into overlapping chunks.
- `chunk_documents()` – processes document records and preserves source metadata.

**Initial configuration:**

| Parameter | Value |
|---|---:|
| Chunk size | 500 characters |
| Chunk overlap | 100 characters |

These are initial experimental values, not final optimized settings.

**Why chunking is important:**

- Reduces the amount of irrelevant information.
- Makes semantic retrieval more efficient.
- Helps retrieve relevant passages.
- Preserves context using overlapping text.

**Testing:**

```powershell
python -m tests.test_text_chunker
python -m tests.test_pipeline_loading
```

**Status:** Completed.


### Step 3: Open-Source Embedding Generation

**Objective:** Convert document chunks into numerical vector representations using a locally executed open-source embedding model.

**Model used:**
- Model: all-MiniLM-L6-v2
- Framework: Sentence Transformers
- Embedding dimension: 384
- Execution: Local CPU/GPU

**Implementation:**

Created `src/embeddings.py` containing:

- `EmbeddingGenerator` – manages the embedding model.
- `generate_embeddings()` – converts text into numerical vectors.
- `embed_chunks()` – generates embeddings while preserving chunk metadata.

**Concepts learned:**

1. Embeddings represent the semantic meaning of text numerically.
2. Semantically similar sentences should have similar vector representations.
3. Embedding dimensions determine the size of the resulting vectors.
4. Normalized embeddings simplify cosine similarity calculations.
5. Open-source embedding models can run locally without an external API.

**Testing:**

```powershell
python -m tests.test_embeddings
python -m tests.test_embedding_pipeline
```

**Important observation:**

The selected model produces 384-dimensional embeddings and has a maximum input sequence length of 256 word pieces. Input length must be considered when designing chunking strategies.

**Status:** Completed.


## How to Run the Project

Clone the repository:

```powershell
git clone https://github.com/YOUR_USERNAME/ai-document-assistant.git
cd ai-document-assistant
```

Create a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Learning Outcomes

Through this project, I am learning:

- Retrieval-Augmented Generation architecture.
- Document processing and text extraction.
- Chunking strategies for RAG.
- Metadata management.
- Modular Python development.
- Testing and version control.

## Future Improvements

- Open-source embedding generation.
- ChromaDB vector storage.
- Semantic similarity search.
- Local LLM integration.
- Conversational document assistant.
- Source citations and document references.
- Streamlit user interface.
