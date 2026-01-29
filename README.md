# 📄 PDF Chat RAG (Streamlit + FAISS + Gemini)

A lightweight “Chat with your PDF” app built from scratch.  
Upload any PDF, it chunks the content, creates embeddings, stores them in a FAISS index, retrieves the most relevant chunks, and answers your question using **Google Gemini**.

## ✅ Features
- Upload a PDF and ask questions
- Paragraph-wise chunking (or chunking with overlap, depending on your code)
- Semantic search using **SentenceTransformers + FAISS**
- Answer generation using **Gemini API**
- Chat UI (enter to send + thinking spinner)
- Show retrieved chunks inside a dropdown (for transparency)

---

## 🧱 Tech Stack
- **Streamlit** (UI)
- **PyPDF** (PDF text extraction)
- **SentenceTransformers** (embeddings)
- **FAISS** (vector search)
- **Google Gemini API** (LLM answers)
- **Python 3.10+**

---

## 🚀 Setup (Local)

### 1) Clone the repo
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
