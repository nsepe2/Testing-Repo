import sys
import os
import streamlit as st
import pandas as pd
import random
import pydeck as pdk
from dotenv import load_dotenv
from utils.b2 import B2

# Add the utils directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))

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
        b2.set_bucket(bucket_name)  # Set the bucket before fetching the file
        # Fetch the object from Backblaze
        obj = b2.get_object(file_name)  # Use only the file name (remote_path)
        return pd.read_csv(obj)
    except Exception as e:
        st.error(f"Error fetching data from Backblaze: {e}")
        return None

# Streamlit app
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

# Buyer Page
elif st.session_state.page == "buyer":
    st.header("Explore Listings in Austin, Texas")
    bucket = st.text_input('Enter Bucket Name', value='AirBnB-CSV')
    file = st.text_input('Enter File Name', value='AirBnB-CSV.csv')
    if st.button('Load Data'):
        data = fetch_data(bucket, file)
        if data is not None:
            st.map(data[['latitude', 'longitude']])

            # Add interactivity with Pydeck
            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/streets-v11',
                initial_view_state=pdk.ViewState(
                    latitude=data['latitude'].mean(),
                    longitude=data['longitude'].mean(),
                    zoom=10,
                    pitch=50,
                ),
                layers=[
                    pdk.Layer(
                        'ScatterplotLayer',
                        data=data,
                        get_position='[longitude, latitude]',
                        get_color='[200, 30, 0, 160]',
                        get_radius=200,
                        pickable=True
                    )
                ],
                tooltip={
                    "html": "<b>Listing:</b> {name}<br/><b>Bedrooms:</b> {bedrooms}<br/><b>Bathrooms:</b> {bathrooms}<br/><b>Amenities:</b> {amenities}",
                    "style": {
                        "backgroundColor": "steelblue",
                        "color": "white"
                    }
                }
            ))

    # Back button to go back to main page
    if st.button("Back"):
        st.session_state.page = "main"

# Seller Page
elif st.session_state.page == "seller":
    st.header("Estimate Your Airbnb Listing Review Score")

    # Drop-down inputs for seller
    property_type = st.selectbox("Property Type", ["House", "Apartment", "Condo", "Townhouse"])
    bedrooms = st.selectbox("Number of Bedrooms", [1, 2, 3, 4, 5])
    bathrooms = st.selectbox("Number of Bathrooms", [1, 2, 3, 4, 5])
    amenities = st.text_input("List of Amenities (comma separated)")
    description = st.text_area("Description of Your Listing")

    if st.button("Generate Review Score"):
        # Generate a random score out of 5
        review_score = round(random.uniform(1, 5), 2)
        st.success(f"Estimated Review Score: {review_score} out of 5")

    # Back button to go back to main page
    if st.button("Back"):
        st.session_state.page = "main"
