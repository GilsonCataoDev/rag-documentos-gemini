# Document Q&A Pipeline (RAG) with LangChain & Gemini

A lightweight, context-aware Retrieval-Augmented Generation (RAG) pipeline designed for question-answering over PDF documents. Built with LangChain, ChromaDB, and Google Gemini, this project extracts text from local documents, indexes them into a vector space, and generates strictly grounded responses to minimize hallucinations.

---

## Architecture & Data Flow

```mermaid
flowchart LR
    A[PDF Document] --> B[PyPDFLoader]
    B --> C[Recursive Text Splitter\nchunk: 1000 | overlap: 200]
    C --> D[gemini-embedding-001]
    D --> E[(ChromaDB Vector Store)]
    
    Q[User Query] --> F[Similarity Retriever\nk=3]
    E --> F
    F --> G[Context-Grounded Prompt]
    Q --> G
    G --> H[gemini-3.8-flash\ntemp=0.2]
    H --> I[StrOutputParser]
    I --> J[Terminal Response]
```

---

## Technical Design Decisions

- **Recursive Text Splitting:** Uses `RecursiveCharacterTextSplitter` with double-newlines, single-newlines, and spaces as hierarchical delimiters. This preserves paragraph and sentence boundaries, avoiding fragmented clauses that distort semantic embeddings.
- **Overlap Window (200 characters):** Guarantees continuity across chunk boundaries so cross-sentence concepts are not split between adjacent vectors.
- **Vector Indexing (`gemini-embedding-001`):** Generates dense vector representations optimized for semantic similarity and information retrieval tasks.
- **Hallucination Suppression via Prompt Grounding:** The system prompt constrains the LLM to only answer from the provided context. If evidence is absent, the model explicitly acknowledges the lack of information rather than inferring facts.
- **Declarative Pipeline (LCEL):** Implemented using LangChain Expression Language (`RunnablePassthrough`, pipes `|`), enabling modular testing and zero-friction model swapping.

---

## Repository Structure

```text
├── .env.example          # Sample configuration file for environment variables
├── .gitignore            # Excludes secrets (.env), caches, and virtual environments
├── rag_basico.py         # Main pipeline script (ingestion, indexing, retrieval, generation)
├── requirements.txt      # Pinned project dependencies
└── README.md             # Project documentation
```

---

## Quickstart

### Prerequisites

- Python 3.11+
- A Google AI Studio API key ([Get an API key](https://aistudio.google.com/))

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/GilsonCataoDev/rag-documentos-gemini.git
cd rag-documentos-gemini

python -m venv venv
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# Linux/macOS
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Secrets

Copy `.env.example` to `.env` and insert your Gemini API key:

```bash
cp .env.example .env
```

```env
GEMINI_API_KEY="your_actual_api_key_here"
```

### 4. Run the Pipeline

Place any target PDF in the root directory named `documento.pdf` and execute:

```bash
python rag_basico.py
```

---

## Future Roadmap

- [ ] Add persistent on-disk vector storage with ChromaDB to bypass re-indexing on successive runs.
- [ ] Implement re-ranking (e.g., Cohere Rerank or BGE Reranker) to optimize top-k precision.
- [ ] Integrate automated evaluation benchmarks using RAGAS (faithfulness, answer relevance, context recall).
- [ ] Expose the retrieval chain via a FastAPI endpoint and a lightweight Streamlit interface.

---

## License

MIT
