from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import requests
import numpy as np
from pymongo import MongoClient
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
# pylint:disable=relative-beyond-top-level

load_dotenv()

# FastAPI app initialization
app = FastAPI()

# Setting the environment
# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DATABASE_NAME = "RAG"
COLLECTION_NAME = "embeddings"
HF_API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5"
HF_TOKEN = os.getenv("HF_TOKEN")

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# OpenAI client initialization
openai_client = OpenAI()

# CORS configuration
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model for query input
class QueryRequest(BaseModel):
    question: str

# Helper function to query Hugging Face API
def query_hf_api(payload):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=360)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error querying Hugging Face API")
    return response.json()

PREFIX = "/api"

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG Assistant API. Use the /query endpoint to interact."}

@app.post("/query")
async def query(request: QueryRequest):
    user_query = request.question

    # Generate the embedding for the user query using Hugging Face API
    payload = {"inputs": [user_query]}
    response = query_hf_api(payload)
    query_embedding = np.array(response).reshape(1, -1)

    # Retrieve all embeddings from MongoDB
    cursor = collection.find({})
    documents = []
    embeddings = []
    metadata = []

    for doc in cursor:
        documents.extend(doc['sentences'])
        if 'embeddings' in doc:
            embeddings.extend(doc['embeddings'])
        metadata.append(doc.get('metadata', {}))

    # Check if there are embeddings in the database
    if not embeddings:
        raise HTTPException(status_code=404, detail="No embeddings found in the database.")

    # Convert embeddings to NumPy array and ensure it is 2D
    embeddings = np.array(embeddings)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    # Calculate cosine similarity between query embedding and document embeddings
    similarities = cosine_similarity(query_embedding, embeddings).flatten()

    # Get top 3 most similar documents (handle cases with fewer than 3 documents)
    top_indices = similarities.argsort()[-3:][::-1]
    top_documents = [documents[i] for i in top_indices if i < len(documents)]
    top_metadata = [metadata[i] for i in top_indices if i < len(metadata)]

    # Preparing data for OpenAI
    system_prompt = """
    You are a helpful assistant. You answer questions about Web API development. 
    But you only answer based on knowledge I'm providing you. You don't use your internal 
    knowledge and you don't make things up.
    If you don't know the answer, just say: I don't know
    --------------------
    The data:
    """ + "<br>".join(["".join(doc) for doc in top_documents]) + """
    --------------------
    The question:
    """ + user_query + """
    """

    # Get the response from OpenAI
    try:
        openai_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        )
        answer = openai_response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying OpenAI: {str(e)}")

    # Return the response
    return {
        "answer": answer,
        "top_documents": top_documents,
        "file_name": "Designingwebapis.pdf",
        "metadata": top_metadata
    }