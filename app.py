import streamlit as st
import numpy as np
import pickle
from utils.modeling_sentiment import (
    get_sentiment_score,
    encode_property_type
)

# Load trained model, scaler, and analyzer from pickle files
with open('model.pickle', 'rb') as model_file:
    model_data = pickle.load(model_file)
    model = model_data['model']
    scaler = model_data['scaler']

with open('analyzer.pickle', 'rb') as analyzer_file:
    analyzer = pickle.load(analyzer_file)

# Streamlit Interface
if 'page' not in st.session_state:
    st.session_state.page = 'main'

# Main Page Content
if st.session_state.page == 'main':
    st.title("Welcome to the Property Review Score Predictor")
    st.write("Please select if you are a buyer or a seller to proceed.")
    if st.button("I'm a Seller"):
        st.session_state.page = 'seller'
    if st.button("I'm a Buyer"):
        st.write("Buyer page is under construction.")

if st.session_state.page == "seller":
    st.sidebar.title("Seller's Property Details")

    # Sidebar for Seller Input Form
    property_types = ["Entire home", "Private room", "Shared room", "Hotel room"]
    price = st.sidebar.number_input("Price", min_value=10, max_value=50000, value=150)

    # Dropdown for Property Type
    property_type = st.sidebar.selectbox("Property Type", property_types)

    # Number inputs for Bedrooms, Bathrooms, Beds, etc.
    bedrooms = st.sidebar.number_input("Number of Bedrooms", min_value=1, max_value=10, value=1)
    bathrooms = st.sidebar.number_input("Number of Bathrooms", min_value=1, max_value=10, value=1)
    accommodates = st.sidebar.number_input("Accommodates", min_value=1, max_value=20, value=1)
    beds = st.sidebar.number_input("Number of Beds", min_value=1, max_value=20, value=1)

    # Text inputs for Neighborhood Overview, Host Neighborhood, and Amenities
    neighborhood_overview = st.sidebar.text_area("Neighborhood Overview", "Enter details about the neighborhood...")
    host_neighborhood = st.sidebar.text_area("Host Neighborhood", "Enter details about the host's neighborhood...")
    amenities = st.sidebar.text_area("Amenities", "Enter details about the amenities available...")

    # Calculate sentiment scores using the loaded analyzer
    def get_sentiment_score(text):
        return analyzer.polarity_scores(text)['compound']

    neighborhood_sentiment = get_sentiment_score(neighborhood_overview)
    host_neighborhood_sentiment = get_sentiment_score(host_neighborhood)
    amenities_sentiment = get_sentiment_score(amenities)

    # Flag to check if the submit button has been clicked
    submitted = st.sidebar.button("Submit Property")

    # Seller Page Content
    if not submitted:
        st.title("Seller's Property Submission")
        st.write("Fill in the property details on the sidebar to submit your listing.")
    else:
        st.markdown("### Property Details Submitted")
        st.write(f"**Property Type:** {property_type}")
        st.write(f"**Price:** ${price}")
        st.write(f"**Bedrooms:** {bedrooms}")
        st.write(f"**Bathrooms:** {bathrooms}")
        st.write(f"**Accommodates:** {accommodates}")
        st.write(f"**Beds:** {beds}")
        st.write(f"**Neighborhood Overview Sentiment:** {neighborhood_sentiment}")
        st.write(f"**Host Neighborhood Sentiment:** {host_neighborhood_sentiment}")
        st.write(f"**Amenities Sentiment:** {amenities_sentiment}")

        # Generate and display the predicted review score
        input_features = np.array([[
            accommodates, bathrooms, bedrooms, beds, price,
            neighborhood_sentiment, host_neighborhood_sentiment, amenities_sentiment
        ] + encode_property_type(property_type)])
        input_features = scaler.transform(input_features)  # Standardize the input features
        predicted_score = model.predict(input_features)[0]
        predicted_score = round(min(max(predicted_score, 0), 5), 2)
        st.markdown(f"## 🔥 **Predicted Review Score Rating: {predicted_score:.2f}** 🔥")

# Back button to go back to main page
if st.button("Back"):
    st.session_state.page = "main"








