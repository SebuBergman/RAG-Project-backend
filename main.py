import os
import json
import requests
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import uvicorn
from langchain_openai.embeddings import OpenAIEmbeddings


# your modules
from create_embeddings import process_pdf, upload_to_s3, delete_all_s3_files
from database import (
    get_cache_collection,
    get_pdf_metadata,
    store_query_result,
    find_similar_cached_query,
    clear_cache_entries,
    insert_pdf_metadata,
    keyword_search_local,
    hybrid_search,
    clear_all_embeddings,
    clear_all_pdfs,
    vector_search,
    get_cache_stats as db_get_cache_stats  # avoid name collision
)

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
    cached: bool = False # default to False

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
    """Answer a question using both vector and keyword search using OpenAI embeddings."""
    try:
        print(f"Received query: {request.question}, Keyword: {request.keyword}, File: {request.file_name}")
        user_query = request.question
        user_keyword = request.keyword
        file_name = request.file_name

        # --- Generate query embedding using OpenAI ---
        embed_model = OpenAIEmbeddings(model="text-embedding-3-large")
        try:
            query_embedding = embed_model.embed_query(user_query).reshape(1, -1)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI embedding error: {e}")

        # Check cache first
        print("Checking cache for similar queries...")
        cached_result = find_similar_cached_query(query_embedding)
        if cached_result:
            print(f"Cache hit: Similar query found - {cached_result['query']}")
            return {
                "answer": cached_result["answer"],
                "results": cached_result["context"],
                "file_metadata": {},
                "cached": True
            }
        print("No similar queries found in cache.")

        # Perform vector search using Milvus
        vector_results = vector_search(query_embedding, limit=7)
        print(f"Vector search returned {len(vector_results)} results")

        # Perform keyword search
        keyword_results = keyword_search_local(user_keyword, file_name, limit=5)
        print(f"Keyword search returned {len(keyword_results)} results")

        # Combine results using hybrid search
        hybrid_results = hybrid_search(vector_results, keyword_results, alpha=0.7, limit=7)
        print(f"Hybrid search returned {len(hybrid_results)} results")

        # Build context from top hybrid results
        context_lines = []
        for i, doc in enumerate(hybrid_results):
            context_lines.append(
                f"Document {i+1} (Hybrid Score: {doc['hybrid_score']:.2f} | "
                f"Semantic: {doc['vector_score']:.2f} | Keyword: {doc['keyword_score']:.2f})\n"
                f"{doc['sentence']}\n"
            )
        context = "\n".join(context_lines)

        # System prompt
        system_prompt = f"""
        Answer the question based only on the following context:
        
        Context:
        {context}

        Question: {user_query}

        Rules:
        - Provide a concise answer based on the context.
        - If the answer is not in the context, say "I don't know."
        - Never make up information
        - Be concise and factual
        """

        # Get file metadata
        file_metadata = {}
        pdf_metadata = get_pdf_metadata()
        if pdf_metadata and file_name:
            file_metadata = next(
                (doc for doc in pdf_metadata if doc['file_name'] == file_name),
                {}
            )

        # Get OpenAI completion
        try:
            openai_response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.3
            )
            answer = openai_response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            answer = "I encountered an error while processing your question."

        # Store query result in cache
        print("Storing query result in cache...")
        store_query_result(
            query_text=user_query,
            query_embedding=query_embedding,
            answer=answer,
            context=context,
        )
        print("Query result stored successfully.")

        return {
            "answer": answer,
            "results": context,
            "file_metadata": file_metadata,
            "cached": False
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in query endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    
@app.get("/cache/stats")
def cache_stats_endpoint():
    """Endpoint to return cache stats."""
    try:
        stats = db_get_cache_stats()
        if "error" in stats:
            raise HTTPException(status_code=500, detail=stats["error"])
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cache/clear_old")
def clear_old_cache_entries(days: int = 30):
    """Clear cache entries older than specified days (default: 30)."""
    try:
        if days <= 0:
            raise HTTPException(
                status_code=400,
                detail="Days parameter must be positive"
            )
            
        deleted_count = clear_cache_entries(days=days)
        return {
            "status": "success",
            "message": f"Deleted {deleted_count} entries older than {days} days",
            "deleted_count": deleted_count,
            "days": days
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing old cache entries: {str(e)}"
        )

@app.post("/cache/clear_all")
def clear_whole_cache():
    """Clear ALL cache entries."""
    try:
        deleted_count = clear_cache_entries()  # No days parameter means clear all
        return {
            "status": "success",
            "message": f"Deleted all {deleted_count} cache entries",
            "deleted_count": deleted_count
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing cache: {str(e)}"
        )

@app.get("/cache/entries")
def list_cache_entries(limit: int = 10):
    """List all recent cache entries."""
    try:
        collection = get_cache_collection()
        entries = list(collection.find(
            {},
            {"_id": 0, "query": 1, "timestamp": 1}
        ).sort("timestamp", -1).limit(limit))
        return {"entries": entries}
    except Exception as e:
        print(f"Error listing cache entries: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/clear_all_data")
def clear_all_data():
    """Clear all embeddings,PDF metadata, and files from S3."""
    try:
        embeddings_deleted = clear_all_embeddings()
        cache_deleted = clear_cache_entries()
        pdf_metadata_deleted = clear_all_pdfs()
        s3_files_deleted = delete_all_s3_files()
        return {
            "status": "success",
            "message": "All data cleared successfully.",
            "details": {
                "embeddings_deleted": embeddings_deleted,
                "cache_deleted": cache_deleted,
                "pdf_metadata_deleted": pdf_metadata_deleted,
                "s3_files_deleted": s3_files_deleted
            }
        }
    except Exception as e:
        print(f"Error clearing all data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear all data: {e}")
    
def main():
    """Main function to run the FastAPI application with uvicorn server."""
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )
    
if __name__ == "__main__":
    main()