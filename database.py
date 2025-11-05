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
def keyword_search(query, file_name, limit=7):
    """Search for a keyword in the sentences array with proper sentence matching."""
    print(f"User Query: {query}, File Name: {file_name}")
    try:
        search_query = [
            {
                '$search': {
                    'index': 'default',
                    'compound': {
                        'must': [
                            {
                                'text': {
                                    'query': query,
                                    'path': 'sentences',
                                    'fuzzy': {}  # Add fuzzy search for better matching
                                }
                            }
                        ],
                        'filter': [
                            {
                                'text': {
                                    'query': file_name,
                                    'path': 'file_name'
                                }
                            }
                        ]
                    }
                }
            },
            {
                '$addFields': {
                    'score': {'$meta': 'searchScore'},
                    'matched_sentences': {
                        '$filter': {
                            'input': '$sentences',
                            'as': 'sentence',
                            'cond': {
                                '$gt': [
                                    {'$indexOfCP': [
                                        {'$toLower': '$$sentence'}, 
                                        {'$toLower': query}
                                    ]},
                                    -1
                                ]
                            }
                        }
                    }
                }
            },
            {
                '$project': {
                    'file_name': 1,
                    'sentences': 1,
                    'score': 1,
                    'matched_sentences': 1,
                    '_id': 0
                }
            },
            {
                '$limit': limit
            }
        ]
        
        results = EMBEDDINGS_COLLECTION.aggregate(search_query)
        results_list = list(results)
        
        """print(f"Found {len(results_list)} results")
        for result in results_list:
            print(f"Match score: {result['score']}")
            print(f"Matched sentences count: {len(result.get('matched_sentences', []))}")
            if result.get('matched_sentences'):
                print("Sample matched sentence:", result['matched_sentences'][0][:100] + "...")"""
        
        return results_list
    except Exception as e:
        print(f"Error in keyword search: {str(e)}")
        return []

# Hybrid search
def hybrid_search(vector_results, keyword_results, limit=7):
    """Combine results while preserving all matched sentences and vector results"""
    # Collect all unique matched sentences from keyword search
    matched_sentences = []
    for result in keyword_results:
        matched_sentences.extend(result.get('matched_sentences', []))
    
    # Combine with vector results (keeping original structure)
    combined_results = {
        'vector_results': vector_results[:limit],
        'matched_sentences': matched_sentences[:limit]
    }
    
    return combined_results

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