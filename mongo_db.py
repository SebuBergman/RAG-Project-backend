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
EMBEDDINGS_COLLECTION = db[EMBEDDINGS_COLLECTION_NAME]  # Initialize as collection object
PDFS_COLLECTION = db[PDFS_COLLECTION_NAME]  # Initialize as collection object

# Store embeddings in MongoDB
def insert_embeddings(data):
    """Insert embeddings into MongoDB."""
    try:
        EMBEDDINGS_COLLECTION.insert_many(data)
        print("Embeddings inserted successfully.")
    except Exception as e:
        print(f"Error inserting embeddings: {e}")

# Fetch all embeddings from MongoDB
def get_all_embeddings():
    """Retrieve all embeddings from MongoDB."""
    try:
        return EMBEDDINGS_COLLECTION.find({})
    except Exception as e:
        print(f"Error retrieving embeddings: {e}")
        return []

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