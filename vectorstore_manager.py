from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

MILVUS_CONNECTION = {
    "uri": os.getenv("MILVUS_DB_PATH", "milvus_local.db"),
}
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "embeddings")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = None

def get_vectorstore():
    global vectorstore
    if vectorstore is None:
        try:
            vectorstore = Milvus(
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
                connection_args=MILVUS_CONNECTION
            )
        except Exception as e:
            print(f"Error initializing vectorstore: {e}")
    return vectorstore
