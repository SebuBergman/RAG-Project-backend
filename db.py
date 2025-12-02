import os
import time
import datetime
import numpy as np
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from pymilvus import MilvusClient, DataType

load_dotenv()

# Set the environment paths
DATA_PATH = os.getenv("DATA_PATH")

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")

# Create a new client and connect to the server
client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# Connect to MongoDB
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[DATABASE_NAME]

# Milvus config
MILVUS_DB_PATH = os.getenv("MILVUS_DB_PATH", "milvus_local.db")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "embeddings")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

milvus_client = MilvusClient(MILVUS_DB_PATH)

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)

try:
    client.server_info()  # forces connection
    print("MongoDB connection successful")
except Exception as e:
    print("MongoDB connection failed:", e)

# Create collections schema
def create_milvus_collection():
    if MILVUS_COLLECTION_NAME not in milvus_client.list_collections():
        schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="sentence", datatype=DataType.VARCHAR, max_length=65535)

        index_params = milvus_client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="FLAT", metric_type="COSINE")

        milvus_client.create_collection(collection_name=MILVUS_COLLECTION_NAME, schema=schema, index_params=index_params)
        print(f"Created Milvus collection: {MILVUS_COLLECTION_NAME}")
    else:
        print(f"Milvus collection already exists: {MILVUS_COLLECTION_NAME}")

create_milvus_collection()

QUERY_CACHE_COLLECTION = mongo_db[os.getenv("QUERY_CACHE_COLLECTION_NAME", "query_cache")]
PDFS_COLLECTION = mongo_db[os.getenv("PDFS_COLLECTION_NAME", "pdfs")]

# Get cached collection from MongoDB
def get_cache_collection():
    """Get the cache collection instance"""
    try:
        print("Fetching cache collection...")
        return QUERY_CACHE_COLLECTION
    except Exception as e:
        print(f"Error fetching cache collection: {str(e)}")
        raise

# Fetch pdfs metadata into MongoDB
def insert_pdf_metadata(metadata):
    """Insert PDF metadata into MongoDB."""
    try:
        PDFS_COLLECTION.insert_one(metadata)
        print("PDF metadata inserted successfully.")
    except Exception as e:
        print(f"Error inserting PDF metadata: {e}")

# Fetch PDF metadata from MongoDB
def get_pdf_metadata():
    """Retrieve all PDF metadata from MongoDB."""
    try:
        return list(PDFS_COLLECTION.find({}, {"_id": 0, "file_name": 1, "file_path": 1}))
    except Exception as e:
        print(f"Error retrieving PDF metadata: {e}")
        return []
    
def store_query_result(query_text, query_embedding, answer, context, threshold=0.9):
    """Store query result in cache."""
    try:
        print(f"Storing query result: {query_text[:50]}...")
        collection = get_cache_collection()

        # Ensure embedding is stored as a flat list
        if isinstance(query_embedding, np.ndarray):
            flattened_embedding = query_embedding.flatten().tolist()
        else:
            flattened_embedding = query_embedding

        # Insert the new query result
        collection.insert_one({
            "query": query_text,
            "embedding": flattened_embedding,
            "answer": answer,
            "context": context,
            "timestamp": datetime.datetime.utcnow()
        })
        print(f"Query result stored successfully.")
    except Exception as e:
        print(f"Error storing query in cache: {str(e)}")
    
def find_similar_cached_query(query_embedding, threshold=0.85):
    """Find semantically similar cached queries."""
    try:
        print("Searching for similar cached queries...")
        query_collection = get_cache_collection()

        # Fetch all cached queries
        cached_queries = list(query_collection.find({}, {
            "query": 1,
            "embedding": 1,
            "answer": 1,
            "context": 1,
            "_id": 0
        }))

        if not cached_queries:
            print("No cached queries found.")
            return None

        print(f"Total cached queries found: {len(cached_queries)}")
        valid_embeddings = []
        valid_queries = []

        # Collect valid embeddings and queries
        for query in cached_queries:
            if "embedding" in query and query["embedding"]:
                valid_embeddings.append(np.array(query["embedding"]))
                valid_queries.append(query)

        if not valid_embeddings:
            print("No valid embeddings found in cached queries.")
            return None

        # Stack embeddings into a 2D array
        cache_embeddings = np.vstack(valid_embeddings)

        # Ensure query_embedding is properly shaped
        if isinstance(query_embedding, np.ndarray):
            query_emb = query_embedding.reshape(1, -1)
        else:
            query_emb = np.array(query_embedding).reshape(1, -1)

        # Calculate cosine similarity
        similarities = cosine_similarity(query_emb, cache_embeddings).flatten()

        # Find the best match
        max_idx = np.argmax(similarities)
        if similarities[max_idx] >= threshold:
            print(f"Similar query found with similarity {similarities[max_idx]:.3f}: {valid_queries[max_idx]['query'][:50]}...")
            return valid_queries[max_idx]

        print(f"No similar queries exceed threshold {threshold}. Best match: {similarities[max_idx]:.3f}")
        return None
    except Exception as e:
        print(f"Error finding similar cached query: {str(e)}")
        return None
    
def clear_cache_entries(days=None):
    """Clear cache entries, optionally filtered by age."""
    try:
        collection = get_cache_collection()

        if days is not None:
            # Clear entries older than the specified number of days
            cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
            result = collection.delete_many({"timestamp": {"$lt": cutoff}})
            print(f"Deleted {result.deleted_count} cache entries older than {days} days")
        else:
            # Clear all entries
            result = collection.delete_many({})
            print(f"Deleted ALL {result.deleted_count} cache entries")
            
        return result.deleted_count
    except Exception as e:
        print(f"Error clearing cache entries: {str(e)}")
        return 0
    
def get_cache_stats():
    """Get statistics about the cache."""
    try:
        collection = get_cache_collection()

        stats = {
            "total_entries": collection.count_documents({}),
            "oldest_entry": None,
            "newest_entry": None
        }

        # Get the oldest entry
        oldest = collection.find_one(
            {},
            {"_id": 0, "query": 1, "timestamp": 1},
            sort=[("timestamp", 1)]
        )
        if oldest and "timestamp" in oldest:
            oldest["timestamp"] = oldest["timestamp"].isoformat()
            stats["oldest_entry"] = oldest

        # Get the newest entry
        newest = collection.find_one(
            {},
            {"_id": 0, "query": 1, "timestamp": 1},
            sort=[("timestamp", -1)]
        )
        if newest and "timestamp" in newest:
            newest["timestamp"] = newest["timestamp"].isoformat()
            stats["newest_entry"] = newest

        return stats
    except Exception as e:
        print(f"Error getting cache stats: {str(e)}")
        return {"error": str(e)}
    
def clear_all_embeddings():
    """Clear all embeddings from Milvus."""
    try:
        milvus_client.drop_collection(MILVUS_COLLECTION_NAME)
        print(f"Dropped Milvus collection successfully.")
        # Recreate the collection
        create_milvus_collection()
        return True
    except Exception as e:
        print(f"Error clearing all embeddings: {str(e)}")
        raise

def clear_all_pdfs():
    """Clear all PDFs from the database."""
    try:
        result = PDFS_COLLECTION.delete_many({})
        print(f"Deleted {result.deleted_count} PDF metadata entries from MongoDB.")
        return result.deleted_count
    except Exception as e:
        print(f"Error clearing PDF metadata: {e}")
        raise