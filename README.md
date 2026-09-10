# Agentic AI — Portfolio & Coursework Repository

This repository contains the end-to-end coursework projects for the **Agentic AI** program, demonstrating autonomous multi-step reasoning, tool-augmented execution, retrieval-augmented generation (RAG), and domain-specific decision support.

---

## 📂 Repository Architecture

```text
Agentic-Ai/
├── Project-1/                   # Intelligent Task Execution Agent (ReAct Framework)
│   ├── intelligent-task-agent/
│   │   ├── app.py               # Streamlit interactive UI
│   │   ├── agent/               # ReAct reasoning loop & tool selection
│   │   ├── tools/               # Safe Math AST, Web Search, Date/Time anchor
│   │   └── tests/               # 37 automated tests
│   ├── requirements.txt
│   └── README.md
│
├── Project-2/                   # Knowledge-Based Decision Agent (Agentic RAG)
│   ├── knowledge-based-decision-agent/
│   │   ├── app.py               # Streamlit interactive UI
│   │   ├── agent/               # LangGraph 3-node state machine
│   │   ├── rag/                 # Chroma vector store, semantic retriever, chunking
│   │   ├── data/                # Authoritative university policy documents
│   │   └── tests/               # 29 automated tests
│   ├── requirements.txt
│   └── README.md
│
└── README.md                    # Portfolio overview & navigation
```

---

## 🚀 Projects Overview

### [Project 1: Intelligent Task Execution Agent](./Project-1)
* **Architecture**: Autonomous ReAct reasoning loop with dynamic tool calling orchestrated via LangGraph.
* **Core Capabilities**: Decomposes natural language goals, executes AST-based mathematical operations, retrieves real-time web knowledge with 3-tier fallback, temporally grounds date/time queries, and autonomously decides when to `CONTINUE` or `FINISH`.
* **Stack**: Python 3.11, LangGraph, LangChain Ollama (`llama3.2:3b` / `qwen3:4b`), Streamlit, AST.
* **Testing**: 37 automated unit and integration tests (100% pass).

### [Project 2: Knowledge-Based Decision Agent](./Project-2)
* **Architecture**: Agentic Retrieval-Augmented Generation (RAG) with a 3-node LangGraph state machine (`analyze_query` ➔ `retrieve_context` ➔ `reason_and_decide`).
* **Core Capabilities**: Ingests authoritative university policy documentation into a persistent Chroma vector database, evaluates student eligibility (e.g., merit scholarships), delivers proactive policy recommendations, and deterministically refuses out-of-domain queries without hallucination.
* **Stack**: Python 3.11, LangGraph (`StateGraph`), ChromaDB, LangChain Ollama, Streamlit.
* **Testing**: 29 automated unit and integration tests (100% pass).

---

## 🛠️ Global Prerequisites

* **Python 3.11+**
* **Local Ollama Service** with local models:
  ```bash
  ollama pull llama3.2:3b
  ollama pull qwen3:4b
  ```

---

## 📜 Author & License

* **Author**: AbdulRazzaq Sanwal ([@Razzaq696](https://github.com/Razzaq696))
* **License**: MIT License