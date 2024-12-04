import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
import nltk
from utils.modeling_sentiment import get_sentiment_score, encode_property_type, train_model, initialize_analyzer

# Ensure VADER lexicon is downloaded
nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
initialize_analyzer()

# Train the Linear Regression model
model = train_model()

# Streamlit Interface
if 'page' not in st.session_state:
    st.session_state.page = 'main'

if st.session_state.page == "seller":
    # Sidebar for Seller Input Form
    st.sidebar.title("Seller's Property Details")
    property_types = ["Entire home", "Private room", "Shared room", "Hotel room"]
    price = st.sidebar.number_input("Price", min_value=10, max_value=50000, value=150)

    # Dropdown for Property Type
    property_type = st.sidebar.selectbox("Property Type", property_types)

    # Number inputs for Bedrooms, Bathrooms, Beds, etc.
    bedrooms = st.sidebar.number_input("Number of Bedrooms", min_value=1, max_value=10, value=1)
    bathrooms = st.sidebar.number_input("Number of Bathrooms", min_value=1, max_value=10, value=1)
    accommodates = st.sidebar.number_input("Accommodates", min_value=1, max_value=20, value=1)

    # Text inputs for Neighborhood Overview, Host Neighborhood, and Amenities
    neighborhood_overview = st.sidebar.text_area("Neighborhood Overview", "Enter details about the neighborhood...")
    host_neighborhood = st.sidebar.text_area("Host Neighborhood", "Enter details about the host's neighborhood...")
    amenities = st.sidebar.text_area("Amenities", "Enter details about the amenities available...")

    # Calculate sentiment scores using VADER
    neighborhood_sentiment = get_sentiment_score(neighborhood_overview)
    host_neighborhood_sentiment = get_sentiment_score(host_neighborhood)
    amenities_sentiment = get_sentiment_score(amenities)

    # Flag to check if the submit button has been clicked
    submitted = st.sidebar.button("Submit Property")

    # Main Page Content
    if not submitted:
        # Display introductory text only if not submitted
        st.title("Seller's Property Submission")
        st.write("Fill in the property details on the sidebar to submit your listing.")
    else:
        # Display submitted property details
        st.markdown("### Property Details Submitted")
        st.write(f"**Property Type:** {property_type}")
        st.write(f"**Price:** ${price}")
        st.write(f"**Bedrooms:** {bedrooms}")
        st.write(f"**Bathrooms:** {bathrooms}")
        st.write(f"**Accommodates:** {accommodates}")
        st.write(f"**Neighborhood Overview Sentiment:** {neighborhood_sentiment}")
        st.write(f"**Host Neighborhood Sentiment:** {host_neighborhood_sentiment}")
        st.write(f"**Amenities Sentiment:** {amenities_sentiment}")

        # Generate and display the predicted review score using the linear regression model
        predicted_score = model.predict(np.array([[
            accommodates, bathrooms, bedrooms, price,
            neighborhood_sentiment, host_neighborhood_sentiment, amenities_sentiment
        ] + encode_property_type(property_type)]))[0]
        st.markdown(f"## 🔥 **Predicted Review Score Rating: {predicted_score:.2f}** 🔥")

# Back button to go back to main page
if st.button("Back"):
    st.session_state.page = "main"

