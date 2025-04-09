import os
import json
import requests
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv
from mongo_db import insert_embeddings

load_dotenv()

# Hugging Face API Configuration
HF_API_URL = os.getenv("HF_API_URL")
HF_TOKEN = os.getenv("HF_TOKEN")

# Set the environment paths
DATA_PATH = os.getenv("DATA_PATH", "./data")
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "./embeddings")

os.makedirs(EMBEDDINGS_PATH, exist_ok=True)

def query_hf_api(payload):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=360)
    return response.json()

def process_pdf(file_path):
    """Process PDF file and extract text."""
    try:
        # Extract the name of the PDF
        file_name = os.path.basename(file_path)
        unique_id = os.path.splitext(file_name)[0]  # Use file name (without extension) as unique ID

        # Log file processing
        print(f"Processing file: {file_name}, Path: {file_path}")

        # Read PDF files and extract sentencess
        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()
        print(f"Loaded {len(documents)} documents from {file_name}")

        # Perform sentence embedding using the API
        embeddings = []
        for doc in documents:
            sentences = doc.text.split('. ')  # Split content into sentences
            payload = {"inputs": sentences}
            response = query_hf_api(payload)

            # Log API response
            if "error" in response:
                print(f"Error embedding document {doc.doc_id}: {response['error']}")
            else:
                doc_embeddings = response
                embeddings.append({
                    "unique_id": unique_id,
                    "file_name": file_name,
                    "file_path": file_path,
                    "sentences": sentences,
                    "embeddings": doc_embeddings
                })

        # Save embeddings to a file
        output_file = os.path.join(EMBEDDINGS_PATH, f"{unique_id}_embeddings.json")
        with open(output_file, "w") as f:
            json.dump(embeddings, f, indent=4)
        print(f"Embeddings saved to {output_file}")

        # Insert embeddings into MongoDB
        try:
            insert_embeddings(embeddings)
            print(f"Embeddings for {file_name} inserted into MongoDB.")
        except Exception as e:
            print(f"Error inserting embeddings into MongoDB: {e}")

    except Exception as e:
        print(f"Error processing PDF {file_path}: {e}")
        raise