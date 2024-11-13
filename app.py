import sys
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from utils.b2 import B2

# Add the utils directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))

# Load environment variables
load_dotenv()

# Set Backblaze connection
b2 = B2(
    endpoint=os.getenv('B2_ENDPOINT', 's3.us-east-005.backblazeb2.com'),
    key_id=os.getenv('B2_KEYID'),
    secret_key=os.getenv('B2_APPKEY')
)

def fetch_data():
    try:
        b2.set_bucket('AirBnB-Bucket')  # Set the bucket
        obj = b2.get_object('Airbnb Dataset_Long.csv')  # Use the EXACT file name
        return pd.read_csv(obj)
    except Exception as e:
        st.error(f"Error fetching data from Backblaze: {e}")
        return None

# APPLICATION
st.title("Airbnb Data Viewer")

# Main Page with Buyer and Seller buttons
if "page" not in st.session_state:
    st.session_state.page = "main"

if st.session_state.page == "main":
    st.header("Welcome to the Airbnb Explorer!")
    buyer = st.button("Buyer")
    seller = st.button("Seller")

    if buyer:
        st.session_state.page = "buyer"
    if seller:
        st.session_state.page = "seller"

# Fetch data from Backblaze
data = fetch_data()
if data is not None:
    st.write("Data loaded successfully.")

# Placeholder for Buyer and Seller code to be added 
if st.session_state.page == "buyer":
    st.write("Buyer window placeholder. Replace with  implementation.")

elif st.session_state.page == "seller":
    st.write("Seller window placeholder. Replace with actual implementation.")


