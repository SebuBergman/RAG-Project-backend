import os
import time
import numpy as np
from dotenv import load_dotenv
from pymilvus import MilvusClient, DataType

load_dotenv()

# Zilliz Cloud / Milvus config
ZILLIZ_CLOUD_URI = os.getenv("ZILLIZ_CLOUD_URI")
ZILLIZ_CLOUD_TOKEN = os.getenv("ZILLIZ_CLOUD_TOKEN")

# Milvus collection names
MILVUS_COLLECTION_NAME = os.getenv("EMBEDDINGS_COLLECTION_NAME", "embeddings")
PDF_COLLECTION = os.getenv("PDFS_COLLECTION_NAME", "pdf_metadata")
QUERY_CACHE_COLLECTION = os.getenv("QUERY_CACHE_COLLECTION_NAME", "query_cache")

# Embedding dimension
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

# Initialize Milvus client for Zilliz Cloud
milvus_client = MilvusClient(
    uri=ZILLIZ_CLOUD_URI,
    token=ZILLIZ_CLOUD_TOKEN
)

# Test connection
try:
    collections = milvus_client.list_collections()
    print(f"✓ Zilliz Cloud connection successful. Collections: {collections}")
except Exception as e:
    print(f"✗ Zilliz Cloud connection failed: {e}")
    raise

# Create collections schema's
def create_milvus_collection():
    """Create Milvus collection only if it does NOT already exist.
    Includes explicit file_name field for filtering.
    """
    try:
        existing = milvus_client.list_collections()
        if MILVUS_COLLECTION_NAME in existing:
            print(f"Milvus collection already exists: {MILVUS_COLLECTION_NAME}")
            return  # <-- STOP. Do NOT recreate or modify.

        print(f"Creating new collection: {MILVUS_COLLECTION_NAME}")

        # Define schema
        schema = milvus_client.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
            description="RAG document embeddings collection"
        )

        schema.add_field(
            field_name="pk",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True,
            description="Primary key"
        )

        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=EMBEDDING_DIM,
            description="Document embedding vector"
        )

        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=65535,
            description="Document text content"
        )

        schema.add_field(
            field_name="file_name",
            datatype=DataType.VARCHAR,
            max_length=512,
            description="Original PDF filename / source"
        )

        # Index
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )

        # Create collection
        milvus_client.create_collection(
            collection_name=MILVUS_COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong"
        )

        print(f"✓ Created collection '{MILVUS_COLLECTION_NAME}' with file_name field")

        # Load collection
        milvus_client.load_collection(MILVUS_COLLECTION_NAME)
        print(f"✓ Loaded collection '{MILVUS_COLLECTION_NAME}'")

    except Exception as e:
        print(f"Error creating Milvus collection: {e}")
        raise

def create_pdf_metadata_collection():
    if PDF_COLLECTION in milvus_client.list_collections():
        return

    schema = milvus_client.create_schema(auto_id=True, description="PDF metadata")
    schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("file_name", DataType.VARCHAR, max_length=512)
    schema.add_field("file_path", DataType.VARCHAR, max_length=1024)
    schema.add_field("timestamp", DataType.INT64)
    
    # Add vector field
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=384)

    # Create collection
    milvus_client.create_collection(
        collection_name=PDF_COLLECTION,
        schema=schema,
        consistency_level="Strong"
    )

    # Create index - MUST be done BEFORE loading
    index_params = milvus_client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )
    milvus_client.create_index(
        collection_name=PDF_COLLECTION,
        index_params=index_params
    )
    
    # Wait for index to be created
    time.sleep(2)
    
    # Now load the collection
    milvus_client.load_collection(PDF_COLLECTION)

    print("✓ Created pdf_metadata collection")

def create_query_cache_collection():
    if QUERY_CACHE_COLLECTION in milvus_client.list_collections():
        return

    schema = milvus_client.create_schema(auto_id=True, description="Query cache")
    schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("query", DataType.VARCHAR, max_length=2048)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    schema.add_field("answer", DataType.VARCHAR, max_length=65535)
    schema.add_field("context", DataType.VARCHAR, max_length=65535)
    schema.add_field("timestamp", DataType.INT64)

    # STEP 1: Create collection with no index first
    milvus_client.create_collection(
        collection_name=QUERY_CACHE_COLLECTION,
        schema=schema,
        consistency_level="Strong"
    )

    # STEP 2: Now create index
    index_params = milvus_client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )
    milvus_client.create_index(
        collection_name=QUERY_CACHE_COLLECTION,
        index_params=index_params
    )

    # STEP 3: Load collection
    milvus_client.load_collection(QUERY_CACHE_COLLECTION)

    print("✓ Created query_cache collection")

# Ensure collections exist
create_milvus_collection()
create_pdf_metadata_collection()
create_query_cache_collection()

# PDF Metadata operations
def insert_pdf_metadata(file_name, file_path, embedding=None):
    """Store PDF metadata into Milvus."""
    
    # If no embedding provided, create a dummy one (or generate a real one)
    if embedding is None:
        embedding = [0.0] * 384  # Dummy embedding - replace with actual embeddings!
    
    milvus_client.insert(
        collection_name=PDF_COLLECTION,
        data=[{
            "file_name": file_name,
            "file_path": file_path,
            "timestamp": int(time.time()),
            "embedding": embedding  # MUST include embedding!
        }]
    )
    print(f"✓ Saved PDF metadata for {file_name}")

def get_pdf_metadata():
    """Return all PDF metadata records."""
    results = milvus_client.query(
        collection_name=PDF_COLLECTION,
        filter="pk >= 0",
        output_fields=["file_name", "file_path", "timestamp"]
    )
    return results

def clear_all_pdfs():
    """Delete all PDF metadata."""
    if PDF_COLLECTION in milvus_client.list_collections():
        milvus_client.drop_collection(PDF_COLLECTION)
        create_pdf_metadata_collection()
        print("✓ Cleared all PDF metadata")
    
# Query cache operations 
def store_query_result(query_text, embedding, answer, context):
    """Insert query cache entry into Milvus."""
    if isinstance(embedding, np.ndarray):
        embedding = embedding.tolist()

    milvus_client.insert(
        collection_name=QUERY_CACHE_COLLECTION,
        data=[{
            "query": query_text,
            "embedding": embedding,
            "answer": answer,
            "context": context,
            "timestamp": int(time.time())
        }]
    )
    print("✓ Cached query")
    
def find_similar_cached_query(query_embedding, threshold=0.85):
    """Vector search inside Milvus query_cache collection."""
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()

    results = milvus_client.search(
        collection_name=QUERY_CACHE_COLLECTION,
        data=[query_embedding],
        anns_field="embedding",
        limit=3,
        output_fields=["query", "answer", "context", "timestamp"],
    )

    top = results[0]
    if not top:
        return None

    hit = top[0]
    if hit["distance"] >= threshold:  # cosine similarity
        return {
            "query": hit["entity"]["query"],
            "answer": hit["entity"]["answer"],
            "context": hit["entity"]["context"],
            "timestamp": hit["entity"]["timestamp"]
        }

    return None

def clear_cache_entries():
    """Reset entire query cache."""
    milvus_client.drop_collection(QUERY_CACHE_COLLECTION)
    create_query_cache_collection()
    print("✓ Cleared query cache")


def get_cache_stats():
    stats = milvus_client.get_collection_stats(QUERY_CACHE_COLLECTION)
    return stats

# Clear data functions
def clear_all_embeddings():
    """Clear all embeddings from Milvus."""
    try:
        # Check if collection exists before dropping
        if MILVUS_COLLECTION_NAME in milvus_client.list_collections():
            milvus_client.drop_collection(MILVUS_COLLECTION_NAME)
            print(f"✓ Dropped Milvus collection '{MILVUS_COLLECTION_NAME}' successfully.")
        else:
            print(f"Collection '{MILVUS_COLLECTION_NAME}' does not exist.")
        
        # Recreate the collection
        create_milvus_collection()
        return True
    except Exception as e:
        print(f"Error clearing all embeddings: {str(e)}")
        raise

# Milvus statistics
def get_milvus_collection_stats():
    """Get statistics about the Milvus collection"""
    try:
        stats = milvus_client.get_collection_stats(MILVUS_COLLECTION_NAME)
        return {
            "collection_name": MILVUS_COLLECTION_NAME,
            "stats": stats
        }
    except Exception as e:
        print(f"Error getting Milvus stats: {str(e)}")
        return {"error": str(e)}