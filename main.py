import os
from typing import List, Dict
import uvicorn
import boto3
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# langchain importss
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# your modules
from db import (
    get_cache_collection,
    get_pdf_metadata,
    insert_pdf_metadata,
    clear_cache_entries,
    get_cache_stats as db_get_cache_stats,
    store_query_result,
    find_similar_cached_query,
    clear_all_embeddings,
    clear_all_pdfs,
)
from S3_bucket import upload_to_s3, delete_all_s3_files
from rag_search import (
    vector_search,
    keyword_search,
    hybrid_search,
)

load_dotenv()

# FastAPI app initialization
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_PATH = "./data"
os.makedirs(UPLOAD_PATH, exist_ok=True)

MILVUS_CONNECTION = {
    "uri": os.getenv("MILVUS_DB_PATH", "milvus_local.db"),
}
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "embeddings")

# Initialize LangChain components
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# Global vectorstore reference
vectorstore = None

def get_vectorstore():
    """Get or create vectorstore instance"""
    global vectorstore
    if vectorstore is None:
        try:
            vectorstore = Milvus(
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
                connection_args=MILVUS_CONNECTION,
            )
        except Exception as e:
            print(f"Error initializing vectorstore: {e}")
    return vectorstore

# Model for query input
class QueryRequest(BaseModel):
    question: str
    keyword: str
    file_name: str
    cached: bool = False # default to False

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Hybrid RAG API (Vector + Keyword Search)"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF"""
    try:
        # Save file temporarily
        file_path = os.path.join(UPLOAD_PATH, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Upload to S3
        bucket_name = os.getenv("S3_BUCKET_NAME")
        s3_key = f"pdfs/{file.filename}"
        s3_url = upload_to_s3(file_path, bucket_name, s3_key)
        
        # Load and process PDF with LangChain
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata["file_name"] = file.filename
            doc.metadata["source"] = s3_url
        
        # Split documents
        splits = text_splitter.split_documents(documents)
        
        # Add to vectorstore
        global vectorstore
        if vectorstore is None:
            vectorstore = Milvus.from_documents(
                documents=splits,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                connection_args=MILVUS_CONNECTION,
            )
        else:
            vectorstore.add_documents(splits)
        
        # Store PDF metadata in MongoDB
        insert_pdf_metadata({
            "file_name": file.filename,
            "file_path": s3_url,
            "chunks_count": len(splits)
        })
        
        # Clean up local file
        os.remove(file_path)
        
        return {
            "message": f"Successfully processed {file.filename}",
            "s3_url": s3_url,
            "chunks_created": len(splits)
        }
    
    except Exception as e:
        print(f"Error in upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
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
    """Query the RAG system with custom MongoDB caching"""
    try:
        print(f"Query: {request.question}, Keyword: {request.keyword}, File: {request.file_name}, Cached: {request.cached}")
        
        vs = get_vectorstore()
        if vs is None:
            raise HTTPException(
                status_code=400,
                detail="No documents have been uploaded yet"
            )
        
        # Check cache if enabled
        cached_result = None
        query_embedding = None
        
        if request.cached:
            # Generate embedding for the query
            query_embedding = embeddings.embed_query(request.question)
            
            # Search for similar cached query
            cached_result = find_similar_cached_query(
                query_embedding=query_embedding,
                threshold=0.85
            )
            
            if cached_result:
                print("✓ Using cached result")
                return {
                    "answer": cached_result["answer"],
                    "search_method": "cached",
                    "results_count": 0,
                    "sources": [],
                    "context": cached_result.get("context", ""),
                    "from_cache": True
                }
        
        # Perform vector search
        vector_results = vector_search(
            query=request.question,
            file_name=request.file_name,
            limit=7
        )
        
        # Perform keyword search (only if keyword provided)
        keyword_results = []
        if request.keyword:
            keyword_results = keyword_search(
                query=request.keyword,
                file_name=request.file_name,
                limit=5
            )
        
        # Combine with hybrid search
        if keyword_results:
            final_results = hybrid_search(
                vector_results=vector_results,
                keyword_results=keyword_results,
                alpha=request.alpha,
                limit=7
            )
            search_method = "hybrid"
        else:
            final_results = vector_results
            search_method = "vector_only"
        
        # Build context from results
        context_lines = []
        for i, result in enumerate(final_results):
            if search_method == "hybrid":
                context_lines.append(
                    f"Document {i+1} (Hybrid: {result['hybrid_score']:.3f} | "
                    f"Vector: {result['vector_score']:.3f} | "
                    f"Keyword: {result['keyword_score']:.3f})\n"
                    f"File: {result['file_name']}\n"
                    f"{result['content']}\n"
                )
            else:
                context_lines.append(
                    f"Document {i+1} (Score: {result['score']:.3f})\n"
                    f"File: {result['file_name']}\n"
                    f"{result['content']}\n"
                )
        
        context = "\n".join(context_lines)
        
        # Create prompt
        prompt = f"""Use the following context to answer the question.
If you don't know the answer, say so. Be concise and factual.

Context:
{context}

Question: {request.question}

Answer:"""
        
        # Get answer from LLM
        response = llm.invoke(prompt)
        answer = response.content
        
        # Store in cache if enabled
        if request.cached and query_embedding is not None:
            store_query_result(
                query_text=request.question,
                query_embedding=query_embedding,
                answer=answer,
                context=context,
                threshold=0.85
            )
        
        return {
            "answer": answer,
            "search_method": search_method,
            "results_count": len(final_results),
            "sources": final_results,
            "context": context,
            "from_cache": False
        }
    
    except Exception as e:
        print(f"Error in query: {e}")
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
def clear_old_cache_entries_endpoint(days: int = 30):
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
        deleted_count = clear_cache_entries()
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
    """List recent cache entries with their answers."""
    try:
        collection = get_cache_collection()
        entries = list(collection.find(
            {},
            {"_id": 0, "query": 1, "answer": 1, "timestamp": 1}
        ).sort("timestamp", -1).limit(limit))
        
        # Format timestamps for readability
        for entry in entries:
            if "timestamp" in entry:
                entry["timestamp"] = entry["timestamp"].isoformat()
        
        return {"entries": entries, "count": len(entries)}
    except Exception as e:
        print(f"Error listing cache entries: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/clear_all")
async def clear_all():
    """Clear all data (embeddings, PDFs, and S3)"""
    try:
        global vectorstore
        
        # Clear Milvus embeddings
        embeddings_cleared = clear_all_embeddings()
        vectorstore = None
        
        # Clear PDF metadata from MongoDB
        pdfs_deleted = clear_all_pdfs()
        
        # Clear S3 files
        s3_deleted = delete_all_s3_files()
        
        return {
            "status": "success",
            "message": "All data cleared",
            "embeddings_cleared": embeddings_cleared,
            "pdf_metadata_deleted": pdfs_deleted,
            "s3_files_deleted": s3_deleted
        }
    
    except Exception as e:
        print(f"Error clearing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
def main():
    """Run the FastAPI application"""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()