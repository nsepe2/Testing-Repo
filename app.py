import os
import pickle
import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from utils.modeling_sentiment import encode_property_type, load_model

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
def main():
    st.title("Airbnb Review Score Prediction")
    st.write("Provide details of your potential listing to predict the review score.")

    # User inputs for prediction
    accommodates = st.number_input("Accommodates", min_value=1, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=0.5, step=0.5)
    bedrooms = st.number_input("Bedrooms", min_value=1, step=1)
    beds = st.number_input("Beds", min_value=1, step=1)
    price = st.number_input("Price (USD)", min_value=10, step=1)
    neighborhood_overview = st.text_area("Neighborhood Overview")
    host_neighborhood = st.text_area("Host Neighborhood Description")
    amenities = st.text_area("Amenities")
    property_type = st.selectbox("Property Type", ["Apartment", "House", "Condo", "unknown"])

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
                # Add missing numerical features with a default value of the mean or 0
                input_data_encoded[missing_feature] = 0

    # Reorder columns to match the expected features
    input_data_encoded = input_data_encoded[expected_features]

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

if __name__ == "__main__":
    main()

















