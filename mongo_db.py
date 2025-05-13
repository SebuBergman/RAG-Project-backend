import os
from pymongo import MongoClient
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import datetime

load_dotenv()

# Set the environment paths
DATA_PATH = os.getenv("DATA_PATH")

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]

# Collections
QUERY_CACHE_COLLECTION = db[os.getenv("QUERY_CACHE_COLLECTION_NAME", "query_cache")]
EMBEDDINGS_COLLECTION = db[os.getenv("EMBEDDINGS_COLLECTION_NAME")]
PDFS_COLLECTION = db[os.getenv("PDFS_COLLECTION_NAME")]

# Fetch all embeddings from MongoDB
def get_all_embeddings():
    """Retrieve all embeddings from MongoDB."""
    try:
        # Get the collection object using the name from .env
        collection = db[os.getenv("EMBEDDINGS_COLLECTION_NAME")]
        
        # Use the collection object to perform find()
        embeddings = list(collection.find({}))
        print(f"Successfully retrieved {len(embeddings)} embeddings.")
        return embeddings
    except Exception as e:
        print(f"Error retrieving embeddings: {str(e)}")
        return []
    
def get_cache_collection():
    """Get the cache collection instance"""
    try:
        print("Fetching cache collection...")
        return QUERY_CACHE_COLLECTION
    except Exception as e:
        print(f"Error fetching cache collection: {str(e)}")
        raise
    
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

# Store embeddings in MongoDB
def insert_embeddings(data):
    """Insert embeddings into MongoDB."""
    try:
        EMBEDDINGS_COLLECTION.insert_many(data)
        print("Embeddings inserted successfully.")
    except Exception as e:
        print(f"Error inserting embeddings: {e}")

# Fetch pdfs metadata into MongoDB
def insert_pdf_metadata(metadata):
    """Insert PDF metadata into MongoDB."""
    try:
        PDFS_COLLECTION.insert_one(metadata)
        print("PDF metadata inserted successfully.")
    except Exception as e:
        print(f"Error inserting PDF metadata: {e}")

# Fetch pdf metadata from MongoDB
def get_pdf_metadata():
    """Retrieve PDF metadata from MongoDB."""
    try:
        return list(PDFS_COLLECTION.find({}, {"_id": 0, "file_name": 1, "file_path": 1}))
    except Exception as e:
        print(f"Error retrieving PDF metadata: {e}")
        return []
    
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
        result = EMBEDDINGS_COLLECTION.delete_many({})
        print(f"Deleted {result.deleted_count} embeddings.")
        return result.deleted_count
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