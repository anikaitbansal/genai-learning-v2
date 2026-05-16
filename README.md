# GenAI RouterBot
Production-style GenAI chatbot with FastAPI, LangGraph, RAG (FAISS + embeddings), PDF ingestion, SQLite memory, Streamlit UI, and Docker.

# Features
- Intent classification
- FastAPI backend
- RAG with FAISS
- PDF ingestion
- LangGraph workflow
- Evaluation + retry
- SQLite persistence
- Streamlit UI
- Dockerized deployment

# Tech Stack
Python,
FastAPI, 
LangChain,
LangGraph,
Groq API,
FAISS,
SentenceTransformers,
SQLite,
Streamlit, 
Docker,

# Run Instructions
- docker-compose up --build
- streamlit run streamlit_app.py

# Limitations
- Retrieval depends on chunk quality
- Local-scale FAISS
- PDF extraction limitations
- Evaluator adds latency
