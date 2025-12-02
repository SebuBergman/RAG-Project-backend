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
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "3072"))

milvus_client = MilvusClient(MILVUS_DB_PATH)

uri = os.getenv("MONGO_URI")
client = MongoClient(uri, serverSelectionTimeoutMS=5000)

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

def insert_embeddings(data):
    """
    Insert embeddings into Milvus.
    data: list of dicts with keys: embedding (list[float]), file_name, sentence
    """
    try:
        if not data:
            print("No data to insert.")
            return {"inserted": 0}
        # Validate dims & prepare entities list expected by client
        entities = []
        for d in data:
            emb = d.get("embedding")
            if not emb or len(emb) != EMBEDDING_DIM:
                raise ValueError(f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {len(emb) if emb else 'None'}")
            entities.append({
                "embedding": emb,
                "file_name": d.get("file_name", ""),
                "sentence": d.get("sentence", "")
            })
        result = milvus_client.insert(collection_name=MILVUS_COLLECTION_NAME, data=entities)
        insert_count = result.get("insert_count", len(entities)) if isinstance(result, dict) else len(entities)
        print(f"Inserted {insert_count} embeddings into Milvus.")
        return {"inserted": insert_count}
    except Exception as e:
        print(f"Error inserting embeddings into Milvus: {e}")
        raise

# Retrieve all embeddings from Milvus Lite
def get_all_embeddings():
    """Retrieve all embeddings from Milvus Lite."""
    try:
        stats = milvus_client.get_collection_stats(collection_name=MILVUS_COLLECTION_NAME)
        row_count = stats.get('row_count', 0)
        print(f"Milvus collection has {row_count} embeddings.")
        return row_count
    except Exception as e:
        print(f"Error retrieving embeddings from Milvus: {e}")
        return 0

# Get cached collection from MongoDB
def get_cache_collection():
    """Get the cache collection instance"""
    try:
        print("Fetching cache collection...")
        return QUERY_CACHE_COLLECTION
    except Exception as e:
        print(f"Error fetching cache collection: {str(e)}")
        raise

def vector_search(query_embedding, limit=7):
    """Perform vector search in Milvus Lite.
    
    Args:
        query_embedding: numpy array of shape (1, 384) or (384,)
        limit: number of results to return
    """
    try:
        print("Performing vector search...")

        # Ensure query_embedding is a flat list
        if isinstance(query_embedding, np.ndarray):
            query_vector = query_embedding.flatten().tolist()
        else:
            query_vector = query_embedding

        results = milvus_client.search(
            collection_name=MILVUS_COLLECTION_NAME,
            data=[query_vector],
            limit=limit,
            output_fields=["file_name", "sentence"],
        )

        formatted_results = []
        for hit in results[0]:
            formatted_results.append({
                "file_name": hit.get("entity", {}).get("file_name", hit.get("file_name", "")),
                "sentence": hit.get("entity", {}).get("sentence", hit.get("sentence", "")),
                "score": hit.get("distance", 0),
            })
        print(f"Vector search found {len(formatted_results)} results.")
        return formatted_results
    except Exception as e:
        print(f"Error during vector search: {e}")
        return []
    
# Keyword search
def keyword_search_local(query, file_name=None, limit=7):
    """Simple keyword search using Milvus query functionality.
    
    Args:
        query: search keyword/phrase
        file_name: optional file name filter
        limit: max results to return
    """
    try:
        print(f"Keyword search - Query: '{query}', File: '{file_name}'")

        # Build filter expression
        filter_expr = None
        if file_name:
            filter_expr = f'file_name == "{file_name}"'

        # Query all document (or filtered by file_name)
        all_results = milvus_client.query(
            collection_name=MILVUS_COLLECTION_NAME,
            filter=filter_expr,
            output_fields=["file_name", "sentence"],
            limit=10000,
        )

        # Filter by keyword match
        query_lower = query.lower()
        results = []
        for item in all_results:
            sentence = item.get("sentence", "")
            if query_lower in sentence.lower():
                results.append({
                    "file_name": item.get("file_name", ""),
                    "sentence": sentence,
                    "score": 1.0
                })
                if len(results) >= limit:
                    break
        
        print(f"Keyword search found {len(results)} results.")
        return results
    except Exception as e:
        print(f"Error in local keyword search: {str(e)}")
        return []

# Hybrid search
def hybrid_search(vector_results, keyword_results, alpha=0.7, limit=7):
    """
    Combine vector and keyword results into a single ranked list.
    alpha = weighting factor (semantic importance)
    (1 - alpha) = keyword importance
    """
    
    # Create a map of keyword scores
    keyword_map = {}
    for result in keyword_results:
        sent = result.get("sentence", "")
        keyword_map[sent] = 1.0 # base score
        # Optionally, fuzzy match to give proportional weight
        # keyword_map[sent] = fuzz.partial_ratio(query.lower(), sent.lower()) / 100.0

    # Merge and compute hybrid scores
    merged = []
    for vec in vector_results:
        sentence = vec.get("sentence", "")
        file_name = vec.get("file_name", "")
        vector_score = vec.get("score", 0)
        keyword_score = keyword_map.get(sentence, 0)

        hybrid_score = (alpha * (vector_score)) + ((1 - alpha) * keyword_score)
        merged.append({
            "file_name": file_name,
            "sentence": sentence,
            "vector_score": vector_score,
            "keyword_score": keyword_score,
            "hybrid_score": hybrid_score
        })
    
    # Add any keyword-only results not in vector results
    vector_sentences = {m["sentence"] for m in merged}
    for kw in keyword_results:
        sent = kw.get("sentence", "")
        if sent not in vector_sentences:
            merged.append({
                "file_name": kw.get("file_name", ""),
                "sentence": sent,
                "vector_score": 0,
                "keyword_score": 1.0,
                "hybrid_score": (1 - alpha) * 1.0
            })
    
    # Soft by hybrid score descending
    merged_sorted = sorted(merged, key=lambda x: x["hybrid_score"], reverse=True)
    return merged_sorted[:limit]

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