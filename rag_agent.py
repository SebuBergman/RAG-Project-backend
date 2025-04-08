import os
import requests
import numpy as np
from pymongo import MongoClient
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

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

def query_hf_api(payload):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=360)
    return response.json()

# Getting the user query
user_query = input("What do you want to know about designing web API's?\n\n")

# Generate the embedding for the user query using the Hugging Face API
payload = {"inputs": [user_query]}
response = query_hf_api(payload)
query_embedding = response

# Convert query embedding to numpy array and ensure it is 2D
query_embedding = np.array(query_embedding).reshape(1, -1)

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

# Debugging: Check the number of documents and embeddings
print(f"Number of documents: {len(documents)}")
print(f"Number of embeddings: {len(embeddings)}")

# Convert embeddings to numpy array and ensure it is 2D
embeddings = np.array(embeddings)
if embeddings.ndim == 1:
    embeddings = embeddings.reshape(1, -1)

# Debugging: Check the shape of the embeddings array
print(f"Shape of embeddings array: {embeddings.shape}")

# Check if embeddings array is empty
if embeddings.size == 0:
    print("No embeddings found in the database.")
    exit()

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
""" + "\n".join(["".join(doc) for doc in top_documents]) + """
--------------------
The question:
""" + user_query + """
"""

# Printing the results for debugging
print(system_prompt)

# Getting response from OpenAI
client = OpenAI()

openai_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
)

print("\n\n---------------------\n\n")

print(openai_response.choices[0].message.content)