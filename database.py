import os
from pymongo import MongoClient
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import datetime
from pymilvus import MilvusClient


load_dotenv()

# Set the environment paths
DATA_PATH = os.getenv("DATA_PATH")

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")

# Connect to MongoDB
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[DATABASE_NAME]

# Milvus Lite Configuration
MILVUS_DB_PATH = os.getenv("MILVUS_DB_PATH", "milvus_local.db")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "embeddings")

# Initialize Milvus Lite client (stores in local file)
milvus_client = MilvusClient(MILVUS_DB_PATH)

# Create collections if they don't exist
if MILVUS_COLLECTION_NAME not in milvus_client.list_collections():
    milvus_client.create_collection(
        collection_name=MILVUS_COLLECTION_NAME,
        dimension=384  # use the embedding dimension you're generating
    )

# MongoDB Collections
QUERY_CACHE_COLLECTION = mongo_db[os.getenv("QUERY_CACHE_COLLECTION_NAME", "query_cache")]
PDFS_COLLECTION = mongo_db[os.getenv("PDFS_COLLECTION_NAME")]

# Store embeddings in Milvus Lite
def insert_embeddings(data):
    """Insert embeddings into Milvus Lite."""
    try:
        vectors = [d["embedding"] for d in data]
        metadata = [{"file_name": d["file_name"], "sentence": d["sentence"]} for d in data]

        milvus_client.insert(
            collection_name=MILVUS_COLLECTION_NAME,
            data=vectors,
            metadata=metadata
        )
        print(f"Inserted {len(vectors)} embeddings into Milvus Lite.")
    except Exception as e:
        print(f"Error inserting embeddings into Milvus: {e}")

# Retrieve all embeddings from Milvus Lite
def get_all_embeddings():
    """Retrieve all embeddings from Milvus Lite."""
    try:
        count = milvus_client.get_collection_stats(collection_name=MILVUS_COLLECTION_NAME)
        print(f"Milvus collection has {count} embeddings.")
        return count
    except Exception as e:
        print(f"Error retrieving embeddings from Milvus: (e)")
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
    """Perform vector search in Milvus Lite."""
    try:
        print("Performing vector search...")
        results = milvus_client.search(
            collection_name=MILVUS_COLLECTION_NAME,
            data=[query_embedding.flatten().tolist()],
            limit=limit,
        )

        formatted_results = []
        for hit in results[0]:
            formatted_results.append({
                "file_name": hit["entity"]["file_name"],
                "sentence": hit["entity"]["sentence"],
                "score": hit["distance"],
            })
        return formatted_results
    except Exception as e:
        print(f"Error during vector search: {(e)}")
        return []
    
# Keyword search
def keyword_search_local(query, file_name=None, limit=7):
    """Simple local keyword search using text metadata stored in Milvus."""
    print(f"User Query: {query}, File Name: {file_name}")
    try:
        all_data = milvus_client.query(
            collection_name=MILVUS_COLLECTION_NAME,
            filter_expression=None,
            output_fields=["file_name", "sentence"]
        )

        # Basic filtering
        results = []
        for item in all_data:
            if query.lower() in item["sentence"].lower():
                if not file_name or item["file_name"] == file_name:
                    results.append(item)
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
        sent = result.get("sentence") or result.get("sentences", "")
        keyword_map[sent] = 1.0 # base score
        # Optionally, fuzzy match to give proportional weight
        # keyword_map[sent] = fuzz.partial_ratio(query.lower(), sent.lower()) / 100.0

    # Merge and compute hybrid scores
    merged = []
    for vec in vector_results:
        sentence = vec.get("sentence") or vec.get("sentences", "")
        file_name = vec.get("file_name")
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
    for kw in keyword_results:
        sent = kw.get("sentence") or kw.get("sentences", "")
        if sent not in [m["sentence"] for m in merged]:
            merged.append({
                "file_name": kw.get("file_name"),
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

# Fetch pdf metadata from MongoDB
"""def get_pdf_metadata():
    try:
        return list(PDFS_COLLECTION.find({}, {"_id": 0, "file_name": 1, "file_path": 1}))
    except Exception as e:
        print(f"Error retrieving PDF metadata: {e}")
        return []"""
    
def store_query_result(query_text, query_embedding, answer, context, threshold=0.9):
    """Store query result in cache."""
    try:
        print(f"Storing query result:\nQuery: {query_text}\nAnswer: {answer}\nThreshold: {threshold}")
        collection = get_cache_collection()

        # Ensure embedding is stored as a flat list
        flattened_embedding = query_embedding.flatten().tolist()

        # Insert the new query result
        collection.insert_one({
            "query": query_text,
            "embedding": flattened_embedding,
            "answer": answer,
            "context": context,
            "timestamp": datetime.datetime.utcnow()
        })
        print(f"Query result stored successfully: {query_text}")
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
            if "embedding" in query:
                # Convert embedding back to a NumPy array and ensure correct shape
                valid_embeddings.append(np.array(query["embedding"]))
                valid_queries.append(query)
            else:
                print(f"Skipping cached query without 'embedding': {query}")

        if not valid_embeddings:
            print("No valid embeddings found in cached queries.")
            return None

        # Stack embeddings into a 2D array
        cache_embeddings = np.vstack(valid_embeddings)
        print(f"Cache embeddings shape: {cache_embeddings.shape}")

        # Calculate cosine similarity
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),  # Ensure query embedding is 2D
            cache_embeddings
        ).flatten()

        print(f"Similarity scores: {similarities}")

        # Find the best match
        max_idx = np.argmax(similarities)
        if similarities[max_idx] >= threshold:
            print(f"Similar query found with similarity {similarities[max_idx]}: {valid_queries[max_idx]['query']}")
            return valid_queries[max_idx]

        print("No similar queries exceed the threshold.")
        return None
    except Exception as e:
        print(f"Error finding similar cached query: {str(e)}")
        return None
    
def clear_cache_entries(days=None):
    """Clear cache entries, optionally filtered by age."""
    try:
        print(f"Clearing cache entries. Days filter: {days}")
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
        print("Fetching cache statistics...")
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
        if oldest:
            oldest["timestamp"] = oldest["timestamp"].isoformat()
            stats["oldest_entry"] = oldest

        # Get the newest entry
        newest = collection.find_one(
            {},
            {"_id": 0, "query": 1, "timestamp": 1},
            sort=[("timestamp", -1)]
        )
        if newest:
            newest["timestamp"] = newest["timestamp"].isoformat()
            stats["newest_entry"] = newest

        print(f"Cache stats: {stats}")
        return stats
    except Exception as e:
        print(f"Error getting cache stats: {str(e)}")
        return {"error": str(e)}
    
def clear_all_embeddings():
    """Clear all embeddings from the database."""
    try:
        milvus_client.drop_collection(MILVUS_COLLECTION_NAME)
        print(f"Dropped Milvus collection succesfully.")
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