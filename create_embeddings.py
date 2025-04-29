import os
import json
import requests
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv
from mongo_db import insert_embeddings
import boto3
from botocore.exceptions import NoCredentialsError

load_dotenv()

# Hugging Face API Configuration
HF_API_URL = os.getenv("HF_API_URL")
HF_TOKEN = os.getenv("HF_TOKEN")

# Set the environment paths
DATA_PATH = os.getenv("DATA_PATH", "./data")
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "./embeddings")

os.makedirs(EMBEDDINGS_PATH, exist_ok=True)

s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)

def upload_to_s3(file_path, bucket_name, s3_key):
    """Uploads a file to S3 and returns the file URL."""
    try:
        s3_client.upload_file(file_path, bucket_name, s3_key)
        s3_url = f"https://{bucket_name}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{s3_key}"
        print(f"File uploaded to S3: {s3_url}")
        return s3_url
    except NoCredentialsError:
        print("AWS credentials not found.")
        raise
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        raise

def query_hf_api(payload):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=360)
    return response.json()

def process_pdf(file_path):
    """Process PDF file and extract text."""
    print(f"Processing PDF file: {file_path}")
    try:
        # Extract the name of the PDF
        file_name = os.path.basename(file_path)
        unique_id = os.path.splitext(file_name)[0]  # Use file name (without extension) as unique ID

        # Upload the file to S3
        bucket_name = os.getenv("S3_BUCKET_NAME")
        s3_key = f"pdfs/{file_name}"
        s3_url = upload_to_s3(file_path, bucket_name, s3_key)

        # Log file processing
        print(f"Processing file: {file_name}, S3 Path: {s3_url}")

        # Read PDF files and extract sentencess
        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()
        #print(f"Loaded {len(documents)} documents from {file_name}")

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
                    "file_path": s3_url,
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