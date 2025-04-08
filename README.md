# RAG-Project-backend

RAG agent built with python & FastAPI
The agent splits a pdf into sentences and creates embeddings of them using a huggingface model (BAAI/bge-small-en-v1.5)
The agent then asks the user for input and based on that input fetches relevant data from the database (MongoDB) and then sends it to openAI's model which then returns it's answer based on the data and question the agent sent.

Here is the Front-end for this project
<a href="https://github.com/SebuBergman/RAG-Project-frontend">https://github.com/SebuBergman/RAG-Project-frontend</a>
