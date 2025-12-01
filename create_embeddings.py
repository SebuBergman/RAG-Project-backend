import os
import json
import requests
from langchain_experimental.text_splitter import SemanticChunker
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv
from database import insert_embeddings
import boto3
from botocore.exceptions import NoCredentialsError
from langchain_openai.embeddings import OpenAIEmbeddings

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

def process_pdf(file_path):
    """Process a PDF file, extract text, create embeddings with OpenAI, and insert into Milvus."""
    print(f"Processing PDF file: {file_path}")
    try:
        file_name = os.path.basename(file_path)
        unique_id = os.path.splitext(file_name)[0]

        # Upload PDF to S3
        bucket_name = os.getenv("S3_BUCKET_NAME")
        s3_key = f"pdfs/{file_name}"
        s3_url = upload_to_s3(file_path, bucket_name, s3_key)
        print(f"Processing file: {file_name}, S3 Path: {s3_url}")

        # Read PDF
        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()

        if not documents:
            print(f"No documents extracted from {file_name}")
            return

        # Initialize OpenAI embeddings and SemanticChunker
        embed_model = OpenAIEmbeddings(model="text-embedding-3-large")
        text_splitter = SemanticChunker(
            embed_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=0.8
        )

        embeddings_to_insert = []

        for doc in documents:
            # Split into semantic chunks
            chunks = text_splitter.create_documents([doc.text])

            # Extract clean sentences
            sentences = []
            for chunk in chunks:
                for s in chunk.page_content.split(". "):
                    s = s.strip()
                    if s:
                        sentences.append(s)

            if not sentences:
                print(f"No valid sentences extracted for {file_name} document {getattr(doc,'doc_id','')}")
                continue

            # Create embeddings in batches to avoid very large requests
            batch_size = 128
            all_embeddings = []
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                try:
                    batch_embeddings = embed_model.embed_documents(batch)
                    all_embeddings.extend(batch_embeddings)
                except Exception as e:
                    print(f"Error generating embeddings for batch: {e}")
                    continue

            if len(all_embeddings) != len(sentences):
                print(f"Warning: embeddings count {len(all_embeddings)} != sentences {len(sentences)} for {file_name}")

            # Prepare data for Milvus
            for sent, emb in zip(sentences, all_embeddings):
                try:
                    embeddings_to_insert.append({
                        "embedding": [float(x) for x in emb],
                        "file_name": file_name,
                        "sentence": sent
                    })
                except Exception:
                    print(f"Skipping sentence due to bad embedding: {sent[:60]}")

        # Save embeddings for backup
        os.makedirs(EMBEDDINGS_PATH, exist_ok=True)
        output_file = os.path.join(EMBEDDINGS_PATH, f"{unique_id}_embeddings.json")
        with open(output_file, "w") as f:
            json.dump(embeddings_to_insert, f, indent=2)
        print(f"Embeddings saved to {output_file}")

        # Insert into Milvus
        if embeddings_to_insert:
            try:
                insert_embeddings(embeddings_to_insert)
                print(f"Inserted {len(embeddings_to_insert)} embeddings for {file_name}")
            except Exception as e:
                print(f"Error inserting embeddings into Milvus: {e}")
        else:
            print(f"No embeddings to insert for {file_name}")

    except Exception as e:
        print(f"Error processing PDF {file_path}: {e}")
        raise

def delete_all_s3_files():
    """Delete all files in the S3 bucket."""
    try:
        bucket_name = os.getenv("S3_BUCKET_NAME")
        deleted_count = 0

        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name):
            if "Contents" in page:
                keys = [{"Key": obj["Key"]} for obj in page["Contents"]]
                if keys:
                    response = s3_client.delete_objects(
                        Bucket=bucket_name,
                        Delete={"Objects": keys}
                    )
                    deleted_count += len(keys)
                    print(f"Deleted {len(keys)} objects from S3")
        print(f"Total deleted files: {deleted_count}")
        return deleted_count

    except Exception as e:
        print(f"Error deleting files from S3: {e}")
        raise
