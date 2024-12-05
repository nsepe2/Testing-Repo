import os
import pickle
import pandas as pd
import streamlit as st
import pydeck as pdk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from utils.modeling_sentiment import encode_property_type, load_model
from utils.b2 import B2

@st.cache_data
def fetch_data():
    try:
        b2.set_bucket('B@_BUCKETNAME')  # Set the bucket
        obj = b2.get_object('Final_PROJ.xlsx')  # Use the EXACT file name
        return pd.read_csv(obj)
    except Exception as e:
        st.error(f"Error fetching data from Backblaze: {e}")
        return None

def get_sentiment_score(text, analyzer):
    """Utility function to get sentiment score using SentimentIntensityAnalyzer."""
    if text:
        sentiment = analyzer.polarity_scores(text)
        return sentiment['compound']
    return 0  # Default sentiment score if text is missing

# Load trained model and scaler from pickle file
try:
    model, scaler, expected_features = load_model()
except FileNotFoundError:
    st.error("Model file not found. Please add the trained model.pickle.")
    st.stop()

# Streamlit UI
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
    st.dataframe(data.head())

# Buyer Page
if st.session_state.page == "buyer":
    st.header("Explore Listings on Map")

    # Filter rows with valid latitude, longitude, name, and price
    data_clean = data.dropna(subset=["latitude", "longitude", "NAME", "Price"])

    # Create a list of properties to be used in the map
    properties = []
    for index, row in data_clean.iterrows():
        if row["NAME"] and row["Price"]:  # Ensure both name and price are present
            properties.append({
                "name": row["NAME"],
                "price": row["Price"],
                "neighborhood": row["Host Neighbourhood"],
                "latitude": row["latitude"],
                "longitude": row["longitude"]
            })

    # Create Pydeck Map with property listings and markers
    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/streets-v11",
        initial_view_state=pdk.ViewState(
            latitude=data_clean["latitude"].mean(),
            longitude=data_clean["longitude"].mean(),
            zoom=10,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=properties,
                get_position=["longitude", "latitude"],
                get_color="[200, 30, 0, 160]",
                get_radius=180,
                pickable=True
            )
        ],
        tooltip={
            "html": "<b>Listing Name:</b> {name}<br/><b>Price:</b>{price}<br/><b>Neighborhood:</b> {neighborhood}",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }
    )

    st.pydeck_chart(deck)

# Seller Page
elif st.session_state.page == "seller":
    # Sidebar for Seller Input Form
    st.sidebar.title("Seller's Property Details")

    # User inputs for prediction
    accommodates = st.sidebar.number_input("Accommodates", min_value=1, step=1)
    bathrooms = st.sidebar.number_input("Bathrooms", min_value=0.5, step=0.5)
    bedrooms = st.sidebar.number_input("Bedrooms", min_value=1, step=1)
    beds = st.sidebar.number_input("Beds", min_value=1, step=1)
    price = st.sidebar.number_input("Price (USD)", min_value=10, step=1)
    neighborhood_overview = st.sidebar.text_area("Neighborhood Overview")
    host_neighborhood = st.sidebar.text_area("Host Neighborhood Description")
    amenities = st.sidebar.text_area("Amenities")
    property_type = st.sidebar.selectbox("Property Type", ["Apartment", "House", "Condo", "unknown"])

    # Sentiment Analysis
    analyzer = SentimentIntensityAnalyzer()
    neighborhood_sentiment = get_sentiment_score(neighborhood_overview, analyzer)
    host_neighborhood_sentiment = get_sentiment_score(host_neighborhood, analyzer)
    amenities_sentiment = get_sentiment_score(amenities, analyzer)

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'accommodates': [accommodates],
        'bathrooms': [bathrooms],
        'bedrooms': [bedrooms],
        'beds': [beds],
        'price': [price],
        'neighborhood_sentiment': [neighborhood_sentiment],
        'host_neighbourhood_sentiment': [host_neighborhood_sentiment],
        'amenities_sentiment': [amenities_sentiment],
        'property_type': [property_type]
    })

    # One-hot encode 'property_type'
    input_data_encoded = encode_property_type(input_data)

    # Ensure the input data has all columns expected by the model
    for missing_feature in expected_features:
        if missing_feature not in input_data_encoded.columns:
            if 'property_type' in missing_feature:
                # Add missing property type columns with a default value of 0
                input_data_encoded[missing_feature] = 0
            else:
                # Add missing numerical features with a default value of 0
                input_data_encoded[missing_feature] = 0

    # Reorder columns to match the expected features
    input_data_encoded = input_data_encoded[expected_features]

    # Add button to submit input data
    if st.sidebar.button("Predict Review Score"):
        # Standardize features
        try:
            input_data_scaled = scaler.transform(input_data_encoded)
        except ValueError as e:
            st.error(f"Error during feature scaling: {e}")
            st.stop()

        # Make prediction
        predicted_score = model.predict(input_data_scaled)[0]

        st.subheader("Predicted Review Score")
        st.write(f"The predicted review score for your listing is: {predicted_score:.2f}")

# Back button to go back to main page
if st.button("Back"):
    st.session_state.page = "main"




















