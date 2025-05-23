🧠 RAG Agent Backend

This is the backend component of a Retrieval-Augmented Generation (RAG) system designed to provide intelligent, context-aware answers based on PDF documents. Built with Python and FastAPI, this service handles embedding, retrieval, and interaction with OpenAI’s language models.


🔍 What It Does

Parses and splits PDF files into individual sentences.

Generates sentence embeddings using Hugging Face's BAAI/bge-small-en-v1.5 model.

Stores and indexes embeddings in a MongoDB database for fast semantic search.

Accepts user queries via API, retrieves the most relevant content based on similarity, and sends it along with the query to OpenAI for final answer generation.

Returns the response to the front-end through a RESTful API.


⚙️ Tech Stack
Language: Python

Framework: FastAPI

Embeddings: HuggingFace Transformers (BAAI/bge-small-en-v1.5)

Database: MongoDB (used for storing and searching vector embeddings)

AI Integration: OpenAI API for final response generation

PDF Handling: Llama_index / pypdf


📡 API Overview

POST /query: Accepts a user question, performs semantic retrieval + OpenAI call, and returns an answer.

POST /upload: Allows uploading and processing of new PDF documents (optional route depending on your setup).

GET /pdfs: Get's a list of uploaded and embeded pdfs

Structured and ready for integration with any front-end via HTTP requests.

<a href="https://github.com/SebuBergman/RAG-Project-frontend">RAG front-end github</a>

🧰 Getting Started

‼️ Prerequisites
This project uses python venv

```bash
 python -m venv rag_project
```

### :running: Run Locally
Create directories (data & embeddings)
```bash
New-Item -Name "data" -Type Directory
New-Item -Name "embeddings" -Type Directory
```

Start the env
```bash
rag_project/Scripts/activate
```

Install required packages
```bash
pip install -r requirements.txt
```

Start backend application
```bash
uvicorn rag_agent:app --reload
```

