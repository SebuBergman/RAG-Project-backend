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

# Collections
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
        
        print(f"Found {len(results_list)} results")
        for result in results_list:
            print(f"Match score: {result['score']}")
            print(f"Matched sentences count: {len(result.get('matched_sentences', []))}")
            if result.get('matched_sentences'):
                print("Sample matched sentence:", result['matched_sentences'][0][:100] + "...")
        
        return results_list
    except Exception as e:
        print(f"Error in keyword search: {str(e)}")
        return []
    
# Simple test function
def test_search():
    test_query = [
        {
            '$search': {
                'index': 'default',
                'text': {
                    'query': "composting",  # Try different terms here
                    'path': 'sentences'
                }
            }
        },
        {'$limit': 7}
    ]
    results = EMBEDDINGS_COLLECTION.aggregate(test_query)
    results_list = list(results)  # Convert cursor to list
    print("Test search results:", results_list)  # Add this line
    return results_list

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