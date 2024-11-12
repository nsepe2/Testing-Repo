import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))

import streamlit as st
from dotenv import load_dotenv
from .utils.b2 import B2  # Assuming you have created this class for Backblaze connection.

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
        obj = b2.get_object(bucket_name, file_name)
        return obj.read()
    except Exception as e:
        st.error(f"Error fetching data from Backblaze: {e}")
        return None

# Streamlit UI
st.title('Backblaze Data Fetcher')
bucket = st.text_input('Enter Bucket Name', value='AirBnB-CSV')  # Default bucket name
file = st.text_input('Enter File Name')

if st.button('Fetch Data'):
    data = fetch_data(bucket, file)
    if data:
        st.write('Data:', data)  # Display data if successfully fetched


