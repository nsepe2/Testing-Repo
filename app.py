import sys
import os

# Add the utils directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))

# Standard imports
import streamlit as st
from dotenv import load_dotenv
from utils.b2 import B2  # Import the B2 class after updating the Python path

# Load environment variables
load_dotenv()

# Set Backblaze connection
b2 = B2(
    endpoint=os.getenv('B2_ENDPOINT', 'https://s3.us-east-005.backblazeb2.com'),
    key_id=os.getenv('B2_KEYID'),
    secret_key=os.getenv('B2_APPKEY')
)

# Example function to retrieve data from Backblaze
def fetch_data(bucket_name, file_name):
    try:
        # Set the bucket first
        b2.set_bucket(bucket_name)
        # Fetch the object from Backblaze
        obj = b2.get_object(file_name)
        return obj.read()
    except Exception as e:
        st.error(f"Error fetching data from Backblaze: {e}")
        return None

# Function to list files in the bucket
def list_files_in_bucket(bucket_name):
    try:
        b2.set_bucket(bucket_name)
        files = b2.list_files()
        return files
    except Exception as e:
        st.error(f"Error listing files in Backblaze: {e}")
        return []

# Streamlit UI
st.title('Backblaze Data Fetcher')
bucket = st.text_input('Enter Bucket Name', value='AirBnB-CSV')  # Default bucket name
file = st.text_input('Enter File Name')

# Fetch Data Button
if st.button('Fetch Data'):
    data = fetch_data(bucket, file)
    if data:
        st.write('Data:', data)  # Display data if successfully fetched

# List Files Button
if st.button('List Files'):
    files = list_files_in_bucket(bucket)
    if files:
        st.write('Files in bucket:', files)
    else:
        st.write('No files found or error listing files.')
