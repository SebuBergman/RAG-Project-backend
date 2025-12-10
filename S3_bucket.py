import os
from dotenv import load_dotenv
import boto3
from botocore.exceptions import NoCredentialsError

load_dotenv()

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
