from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import os
import posixpath  # For consistent forward slash paths
import requests
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
# pylint:disable=relative-beyond-top-level
from create_embeddings import process_pdf, upload_to_s3
from mongo_db import get_all_embeddings, insert_pdf_metadata, get_pdf_metadata, keyword_search, hybrid_search

load_dotenv()

# FastAPI app initialization
app = FastAPI()

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

# Upload folder
UPLOAD_PATH = "./data"
os.makedirs(UPLOAD_PATH, exist_ok=True)

stored_pdfs = []

# Model for query input
class QueryRequest(BaseModel):
    question: str
    keyword: str
    file_name: str

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG Assistant API. Use the /query endpoint to interact."}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF for embedding creation."""
    try:
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_PATH, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Upload the file to S3 and process it
        bucket_name = os.getenv("S3_BUCKET_NAME")
        s3_key = f"pdfs/{file.filename}"
        s3_url = upload_to_s3(file_path, bucket_name, s3_key)

        # Process the uploaded PDF
        process_pdf(file_path)

        # Save metadata with S3
        pdf_metadata = {
            "file_name": file.filename,
            "file_path": s3_url,
        }
        insert_pdf_metadata(pdf_metadata)

        # Clean up the local file
        os.remove(file_path)

        return {"message": f"File {file.filename} uploaded and processed successfully.", "s3_url": s3_url}

    except Exception as e:
        print(f"Error in /upload endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
    
@app.get("/pdfs")
def get_pdfs():
    """Get the list of available PDFs."""
    try:
        pdfs = get_pdf_metadata()

        return {"pdfs": pdfs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving PDFs: {str(e)}")

@app.post("/query")
async def query(request: QueryRequest):
    """Answer a question using both vector and keyword search."""
    user_query = request.question
    user_keyword = request.keyword
    file_name = request.file_name

    # Generate the embedding for the user query using Hugging Face API
    payload = {"inputs": [user_query]}
    response = requests.post(
        os.getenv("HF_API_URL"),
        headers={"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"},
        json=payload,
    )
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error querying Hugging Face API")
    query_embedding = np.array(response.json()).reshape(1, -1)

    # Retrieve all embeddings from MongoDB
    cursor = get_all_embeddings()
    documents, embeddings, metadata = [], [], []

    for doc in cursor:
        documents.extend(doc['sentences'])
        embeddings.extend(doc['embeddings'])
        metadata.append({"file_name": doc['file_name'], "file_path": doc['file_path']})

    # Check if there are embeddings in the database
    if not embeddings:
        raise HTTPException(status_code=404, detail="No embeddings found in the database.")
    
    # Debug: Log lengths of retrieved data
    print(f"Number of Embeddings: {len(embeddings)}")
    print(f"Number of Metadata: {len(metadata)}")
    print(f"Number of Documents: {len(documents)}")

    # Convert embeddings to NumPy array
    embeddings = np.array(embeddings)

    # Debug: Log embedding shape
    print(f"Document Embedding Shape: {embeddings.shape}")

    # Calculate cosine similarity between query embedding and document embeddings
    similarities = cosine_similarity(query_embedding, embeddings).flatten()

    # Debug: Log similarity scores
    print(f"Similarity Scores: {similarities}")

    # Get top 5 most similar documents
    top_indices = similarities.argsort()[-5:][::-1]
    vector_results = [
        {
            "file_name": file_name,
            "sentences": documents[i],
            "score": similarities[i],
        }
        for i in top_indices
    ]

    # Debug: Log vector results
    print(f"Vector Results: {vector_results}")

    # Perform keyword search
    keyword_results = keyword_search(user_keyword, file_name, limit=5)

    # Debug: Log keyword results
    print(f"Keyword Results: {keyword_results}")

    # Combine vector and keyword search results
    hybrid_results = hybrid_search(vector_results, keyword_results, limit=5)

    # Debug: Log hybrid results
    print(f"Hybrid Results: {hybrid_results}")

    # Prepare response
    top_documents = [result for result in hybrid_results]

    print(f"Top documents: {top_documents}")

    # Preparing data for OpenAI
    system_prompt = """
    You are a helpful assistant. 
    You only answer based on knowledge I'm providing you. You don't use your internal 
    knowledge and you don't make things up.
    If you don't know the answer, just say: I don't know
    --------------------
    The data:
    """ + "<br>".join(top_documents) + """
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
        "results": hybrid_results,
        "file_name": ({"file_name": doc['file_name'], "file_path": doc['file_path']}),
        "metadata": metadata,
    }

    