from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# Zilliz Cloud connection configuration
ZILLIZ_CLOUD_URI = os.getenv("ZILLIZ_CLOUD_URI")
ZILLIZ_CLOUD_TOKEN = os.getenv("ZILLIZ_CLOUD_TOKEN")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "embeddings")

# Validate required environment variables
if not ZILLIZ_CLOUD_URI or not ZILLIZ_CLOUD_TOKEN:
    raise ValueError(
        "Missing required environment variables: ZILLIZ_CLOUD_URI and ZILLIZ_CLOUD_TOKEN must be set"
    )

# Connection args for LangChain Milvus
MILVUS_CONNECTION = {
    "uri": ZILLIZ_CLOUD_URI,
    "token": ZILLIZ_CLOUD_TOKEN,
    "secure": True
}

# Initialize embeddings (text-embedding-3-small is 1536 dimensions)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Global vectorstore instance
vectorstore = None

def get_vectorstore():
    """
    Get or initialize the vectorstore for Zilliz Cloud.
    This function ensures the vectorstore is properly connected to the cloud instance.
    
    Returns:
        Milvus: The vectorstore instance, or None if initialization fails
    """
    global vectorstore
    
    if vectorstore is None:
        try:
            print(f"Initializing vectorstore connection to Zilliz Cloud...")
            print(f"Collection: {COLLECTION_NAME}")
            print(f"URI: {ZILLIZ_CLOUD_URI[:50]}...")
            
            # Initialize the Milvus vectorstore
            # This will connect to existing collection or create if needed
            vectorstore = Milvus(
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
                connection_args=MILVUS_CONNECTION,
                # Key parameters for cloud
                consistency_level="Strong",
                drop_old=False,  # Never drop existing collection
                auto_id=True,  # Use auto-generated IDs
            )
            
            print(f"✓ Vectorstore initialized successfully")
            
        except Exception as e:
            print(f"✗ Error initializing vectorstore: {e}")
            print(f"Connection details:")
            print(f"  - URI: {ZILLIZ_CLOUD_URI[:50]}...")
            print(f"  - Collection: {COLLECTION_NAME}")
            print(f"  - Token: {'*' * 20}")
            
            # Don't set vectorstore if initialization fails
            vectorstore = None
            
    return vectorstore

def reset_vectorstore():
    """
    Reset the vectorstore connection.
    Useful after clearing collections or reconnecting.
    Call this after operations that modify the collection structure.
    """
    global vectorstore
    vectorstore = None
    print("Vectorstore connection reset - will reinitialize on next get_vectorstore() call")

def get_connection_info():
    """
    Get connection information for debugging.
    
    Returns:
        dict: Connection configuration (with masked token)
    """
    return {
        "uri": ZILLIZ_CLOUD_URI,
        "collection": COLLECTION_NAME,
        "token": f"{ZILLIZ_CLOUD_TOKEN[:8]}..." if ZILLIZ_CLOUD_TOKEN else None,
        "initialized": vectorstore is not None
    }