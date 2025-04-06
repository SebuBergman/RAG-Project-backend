import os
from pymongo import MongoClient
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Setting the environment
# Mongodb configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DATABASE_NAME = "RAG"
COLLECTION_NAME = "documents"

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Getting the user query
user_query = input("What do you want to know about growing vegetables?\n\n")

# Searching in MongoDB
results = collection.find(
    {"$text": {"$search": user_query}},
    {"score": {"$meta": "textScore"}}
).sort([("score", {"$meta": "textScore"})]).limit(3)

# Preparing data for OpenAI
documents = [result["content"] for result in results]
metadatas = [result["metadata"] for result in results]

# Printing the results for debugging
print(results)

system_prompt = """
You are a helpful assistant. You answer questions about growing vegetables in Florida. 
But you only answer based on knowledge I'm providing you. You don't use your internal 
knowledge and you don't make things up.
If you don't know the answer, just say: I don't know
--------------------
The data:
""" + "\n".join(documents) + """
"""

# Printing the results for debugging
print(system_prompt)

# Getting response from OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
)

print("\n\n---------------------\n\n")

print(response.choices[0].message.content)