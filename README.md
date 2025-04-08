# RAG-Project-backend

Developed a Python-based RAG agent using FastAPI, designed to process PDF documents by splitting them into individual sentences. These sentences are then embedded using a Hugging Face model (BAAI/bge-small-en-v1.5) to enable semantic search and context-aware retrieval.

The system takes user input, performs a similarity search over the embedded sentence database (stored in MongoDB), and retrieves the most relevant information. This data is then sent to an OpenAI model, which generates a response tailored to both the user's query and the retrieved context.

Additionally, I built a separate front-end interface allowing users to interact seamlessly with the system. User questions submitted via the front-end are routed through the API to the backend RAG agent, ensuring fast and contextually accurate responses from the OpenAI model.

Here is the Front-end for this project
<a href="https://github.com/SebuBergman/RAG-Project-frontend">https://github.com/SebuBergman/RAG-Project-frontend</a>
