from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from bson.objectid import ObjectId

# setting the environment
DATA_PATH = r"data"
MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "RAG"
COLLECTION_NAME = "documents"

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# loading the document
loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()

# splitting the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(raw_documents)

# preparing to be added into mongodb
documents = []
metadata = []
ids = []

i = 0

for i, chunk in enumerate(chunks):
    document = {
        "_id": ObjectId(), # unique id for the document
        "content": chunk.page_content, # content of the document
        "metadata": chunk.metadata, # metadata of the document
    }
    documents.append(document)

# adding to mongodb
collection.insert_many(documents)