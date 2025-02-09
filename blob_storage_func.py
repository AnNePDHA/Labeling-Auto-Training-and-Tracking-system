import os
import time
import json

from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

# Load environment variables from .env file
load_dotenv()

# Define your connection string and container name
CONNECTION_STRING = os.environ.get("BLOB_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.environ.get("CONTAINER_NAME")

# Create a BlobServiceClient object which will be used to create a ContainerClient
blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)

# Create the container if it does not exist
container_client = blob_service_client.get_container_client(CONTAINER_NAME)
try:
    container_client.create_container()
except Exception as e:
    pass
    # print(f"Container already exists: {e}")


def list_blobs_in_folder(folder_path):
    """
    List all blobs in a specified folder within a blob storage container.

    Inputs:
        folder_path (str): The path to the folder in the blob storage container.

    Outputs:
        list: A list of blob names present in the specified folder.
    """
    blobs = container_client.list_blobs(folder_path)
    blob_lst = []
    for blob in blobs:
        blob_lst.append(blob.name)
    return blob_lst


def read_file_from_blob(blob_name):
    """
    Read the content of a specified blob from blob storage.

    Inputs:
        blob_name (str): The name of the blob to read.

    Outputs:
        str: The content of the specified blob.
    """
    blob_client = container_client.get_blob_client(blob=blob_name)
    blob_data = blob_client.download_blob().readall()
    return blob_data


def rewrite_blob_content(blob_name, new_content):
    """
    Rewrite the content of a specified blob in blob storage.

    Inputs:
        blob_name (str): The name of the blob to rewrite.
        new_content (str): The new content to write to the blob.
    """
    blob_client = container_client.get_blob_client(blob=blob_name)

    # Upload new content
    blob_client.upload_blob(new_content, overwrite=True)
    print(f"Blob {blob_name} content rewritten")


def move_blob(source_blob_name, destination_blob_name):
    """
    Move a blob from one location to another within blob storage.

    Inputs:
        source_blob_name (str): The name of the source blob.
        destination_blob_name (str): The name of the destination blob.
    """
    source_blob_client = container_client.get_blob_client(blob=source_blob_name)
    destination_blob_client = container_client.get_blob_client(blob=destination_blob_name)

    # Copy the blob to the new location
    destination_blob_client.start_copy_from_url(source_blob_client.url)

    # Wait for the copy to complete
    while destination_blob_client.get_blob_properties().copy.status != 'success':
        print("Waiting for copy to complete...")
        time.sleep(1)

    # Delete the source blob
    source_blob_client.delete_blob()
    print(f"Blob {source_blob_name} moved to {destination_blob_name}")


def upload_json_to_blob(blob_name, json_data):
    """
    Upload result json of progress tracking to blob storage

    Inputs:
        blob_name (str) : specific path to blob storage
        json_data (dictionary): json file result want to upload
    """
    blob_client = container_client.get_blob_client(blob=blob_name)
    # Convert the JSON data to a string
    json_string = json.dumps(json_data)

    # Upload the JSON string to the blob
    blob_client.upload_blob(json_string, overwrite=True)
    print(f"JSON file uploaded to blob storage at {blob_name}")


def delete_webm_files_in_folder(folder_path):
    """
    Delete all .webm files in a specified folder on Azure Blob Storage.

    Inputs:
        folder_path (str): Path to the folder containing the files.
    """
    # List blobs in the specified folder
    blobs = container_client.list_blobs(name_starts_with=folder_path)

    # Iterate over blobs and delete .mp4 files
    for blob in blobs:
        if blob.name.endswith(".webm"):
            try:
                # print(f"Deleting {blob.name}")
                container_client.delete_blob(blob.name)
            except Exception as e:
                print(f"Error deleting {blob.name}: {str(e)}")

    print("Finished deleting all .webm files")
