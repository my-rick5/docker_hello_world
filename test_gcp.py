from google.cloud import storage
import os

def verify_gcp_connection():
    bucket_name = os.getenv("GCP_BUCKET_NAME")
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    print(f"Testing connection to bucket: {bucket_name}")
    print(f"Using key at: {key_path}")
    
    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        
        # List the first 5 blobs in the bucket
        blobs = list(bucket.list_blobs(max_results=5))
        print("Successfully connected! Files found in bucket:")
        for blob in blobs:
            print(f" - {blob.name}")
            
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    verify_gcp_connection()
