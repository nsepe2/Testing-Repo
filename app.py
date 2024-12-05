import os
import sys
import streamlit as st
import pandas as pd
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.modeling_sentiment import load_or_train_model, encode_property_type

# Load environment variables
load_dotenv()

# Load trained model and scaler from pickle file
model, scaler = load_or_train_model()

# Set up Streamlit app interface
st.title('Airbnb Review Score Prediction')

# Get user inputs
st.header('Input Listing Details')
accommodates = st.number_input('Number of Guests Accommodates', min_value=1, step=1)
bathrooms = st.number_input('Number of Bathrooms', min_value=0.5, step=0.5)
bedrooms = st.number_input('Number of Bedrooms', min_value=1, step=1)
beds = st.number_input('Number of Beds', min_value=1, step=1)
price = st.number_input('Price per Night (USD)', min_value=1.0, step=0.5)

neighborhood_overview = st.text_area('Neighborhood Overview')
host_neighbourhood = st.text_area('Host Neighborhood Description')
amenities = st.text_area('Amenities (list)')

property_type = st.selectbox('Property Type', ['Apartment', 'House', 'Condo', 'Loft', 'Other'])

if st.button('Predict Review Score'):
    try:
        # Sentiment Analysis for the user input
        analyzer = SentimentIntensityAnalyzer()
        neighborhood_sentiment = analyzer.polarity_scores(neighborhood_overview)['compound']
        host_neighbourhood_sentiment = analyzer.polarity_scores(host_neighbourhood)['compound']
        amenities_sentiment = analyzer.polarity_scores(amenities)['compound']

        # Create DataFrame for model prediction
        input_data = pd.DataFrame({
            'accommodates': [accommodates],
            'bathrooms': [bathrooms],
            'bedrooms': [bedrooms],
            'beds': [beds],
            'price': [price],
            'neighborhood_sentiment': [neighborhood_sentiment],
            'host_neighbourhood_sentiment': [host_neighbourhood_sentiment],
            'amenities_sentiment': [amenities_sentiment],
            'property_type': [property_type]
        })

        # One-hot encode 'property_type'
        input_data_encoded = encode_property_type(input_data)

        # Align columns with the model features
        all_columns = scaler.mean_.shape[0]
        missing_cols = all_columns - input_data_encoded.shape[1]
        if missing_cols > 0:
            for i in range(missing_cols):
                input_data_encoded[f'missing_{i}']=













