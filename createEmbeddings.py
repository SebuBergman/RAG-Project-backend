import os
import requests
from llama_index.core import SimpleDirectoryReader
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Set the environment paths
DATA_PATH = r"data"

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DATABASE_NAME = "RAG"
COLLECTION_NAME = "embeddings"

# Hugging Face API Configuration
HF_API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5"
HF_TOKEN = os.getenv("HF_TOKEN")
print(f"HF_TOKEN: {HF_TOKEN}")

def query_hf_api(payload):
    print(f"HF_TOKEN: {HF_TOKEN}")
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=360)
    return response.json()

# Read PDF files and extract sentencess
reader = SimpleDirectoryReader(
    input_files=["./data/Designingwebapis.pdf"]
)

documents = reader.load_data()
print(f"Loaded {len(documents)} docs")

# Perform sentence embedding using the API
embeddings = []
for doc in documents:
    sentences = doc.text.split('. ')  # Split content into sentences
    payload = {"inputs": sentences}
    response = query_hf_api(payload)
    if "error" not in response:
        doc_embeddings = response
        embeddings.append({
            "document_id": doc.doc_id,
            "sentences": sentences,
            "embeddings": doc_embeddings
        })
    else:
        print(f"Error embedding document {doc.doc_id}: {response['error']}")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Store embeddings in MongoDB
collection.insert_many(embeddings)