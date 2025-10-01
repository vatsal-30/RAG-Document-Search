# 🤖 Agentic RAG System

An **Agentic Retrieval-Augmented Generation (RAG)** system built with **LangChain, LangGraph, and FAISS**, featuring a **Streamlit UI**.  
This project demonstrates how to ingest documents, index them into a vector store, and query them via an **agent-based workflow** that retrieves relevant context and generates grounded answers.  

---

## 🚀 Features
- **Document ingestion** from URLs, PDFs, and text files.  
- **Recursive text chunking** for optimized retrieval.  
- **Vector database with FAISS** and **OpenAI embeddings**.  
- **Agentic RAG pipeline** built with **LangGraph** and a **ReAct agent**.  
- **Wikipedia integration** for general knowledge fallback.  
- **Streamlit web app** for interactive search and Q&A.  

---

## 🛠 Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-username/agentic-rag.git
cd agentic-rag
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### Environment Variables

Create a .env file in the project root with your OpenAI API key:
```bash
OPENAI_API_KEY=your_openai_api_key
```

## Usage
Run via Streamlit UI
```bash
streamlit run streamlit_app.py
```


Open the local URL (usually http://localhost:8501) in your browser.

Run via CLI
```bash
python main.py
```

## 💡 Example Queries
 - "Summarize the contents of the Lilian Weng agentic AI blog."
 - "What is diffusion in video generation?"
 - "Tell me about retrieval-augmented generation with citations."
