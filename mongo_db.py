import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Set the environment paths
DATA_PATH = os.getenv("DATA_PATH")

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
EMBEDDINGS_COLLECTION_NAME = os.getenv("EMBEDDINGS_COLLECTION_NAME")
PDFS_COLLECTION_NAME = os.getenv("PDFS_COLLECTION_NAME")

# Collections
EMBEDDINGS_COLLECTION = db[EMBEDDINGS_COLLECTION_NAME]
PDFS_COLLECTION = db[PDFS_COLLECTION_NAME]

# Fetch all embeddings from MongoDB
def get_all_embeddings():
    """Retrieve all embeddings from MongoDB."""
    try:
        print("Embeddings retrieved successfully.")
        return EMBEDDINGS_COLLECTION.find({})
    except Exception as e:
        print(f"Error retrieving embeddings: {e}")
        return []
    
# Keyword search
def keyword_search(query, file_name, limit=5):
    """Search for a keyword in the embeddings."""
    print(f"User Query: {query}")
    try:
        print(f"Executing keyword search for query: {query}")
        search_query = [
            {
                '$search': {
                    'index': 'default',
                    'compound': {
                        'must': [
                            {
                                'text': {
                                    'query': query,
                                    'path': 'text'
                                }
                            }
                        ],
                        'filter': [
                            {
                                'text': {
                                    'query': file_name,
                                    'path': 'fileName'
                                }
                            }
                        ]
                    }
                }
            }, {
                '$match': {
                    'fileName': file_name
                }
            }, {
                '$addFields': {
                    'score': {
                        '$meta': 'searchScore'
                    }
                }
            }, {
                '$project': {
                    'embedding': 0
                }
            }, {
                '$limit': limit
            }
        ]
        results = db[EMBEDDINGS_COLLECTION].aggregate(search_query)
        results_list = list(results)
        print(f"Keyword search results: {results_list}")
        return results_list
    except Exception as e:
        print(f"Error searching for keyword: {e}")
        return []
    
# Hybrid search
def hybrid_search(vector_results, keyword_results, limit=5):
    """Combine vector and keyword search results."""
    # Convert results to dictionaries keyed by file_name for easier merging
    vector_dict = {result.get("file_name", "unknown"): result for result in vector_results}
    keyword_dict = {result.get("file_name", "unknown"): result for result in keyword_results}

    # Merge results, prioritize documents that appear in both searches
    combined_results = []
    for file_name in keyword_dict.keys():
        if file_name in vector_dict:
            combined_results.append({
                **keyword_dict[file_name],
                "vector_score": vector_dict[file_name].get("score", 0),
                "keyword_score": keyword_dict[file_name].get("score", 0),
            })

    for file_name in vector_dict.keys():
        if file_name not in keyword_dict:
            combined_results.append(vector_dict[file_name])

    # Sort by combined score
    combined_results.sort(key=lambda x: x.get("vector_score", 0) + x.get("keyword_score", 0), reverse=True)

    # If no combined results exist, return vector results only
    if not combined_results:
        combined_results = vector_results

    print(f"Combined results: {combined_results}")

    # Collect up to 'limit' top sentences
    top_sentences = []
    for result in combined_results:
        if "sentences" in result:
            # If 'sentences' is a list, extend the top_sentences list
            if isinstance(result["sentences"], list):
                top_sentences.extend(result["sentences"])
            else:
                # Otherwise, append the single sentence
                top_sentences.append(result["sentences"])
        # Stop collecting if we reach the limit
        if len(top_sentences) >= limit:
            break

    # Return only the top 'limit' sentences
    return top_sentences[:limit]

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