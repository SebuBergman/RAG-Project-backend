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
from mongo_db import get_all_embeddings, insert_pdf_metadata, get_pdf_metadata

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

        # Save metadata with S3 URL
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

        # Normalize file paths to use forward slashes
        for pdf in pdfs:
            pdf["file_path"] = posixpath.normpath(pdf["file_path"].replace("\\", "/"))

        return {"pdfs": pdfs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving PDFs: {str(e)}")

@app.post("/query")
async def query(request: QueryRequest):
    """Answer a question using the RAG model based on embeddings."""
    user_query = request.question

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

    # Convert embeddings to NumPy array
    embeddings = np.array(embeddings)

    # Calculate cosine similarity between query embedding and document embeddings
    similarities = cosine_similarity(query_embedding, embeddings).flatten()

    # Get top 3 most similar documents (handle cases with fewer than 3 documents)
    top_indices = similarities.argsort()[-4:][::-1]
    top_documents = [documents[i] for i in top_indices]

    # Preparing data for OpenAI
    system_prompt = """
    You are a helpful assistant. 
    You only answer based on knowledge I'm providing you. You don't use your internal 
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
        "file_name": ({"file_name": doc['file_name'], "file_path": doc['file_path']}),
        "top_documents": top_documents,
        "metadata": metadata,
    }

    