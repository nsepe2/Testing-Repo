import numpy as np
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
import ast

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Generate sample training data (for demonstration purposes)
np.random.seed(42)
X_train = np.random.rand(100, 7)  # 100 samples, 7 features (accommodates, bathrooms, bedrooms, price, sentiments)
y_train = np.random.rand(100) * 5  # Random review scores between 0 and 5

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

def predict_review_score(accommodates, bathrooms, bedrooms, price, neighborhood_sentiment, host_neighborhood_sentiment, amenities_sentiment, property_type):
    # Encoding property type (assuming one-hot encoding and pre-determined columns for property types)
    property_type_encoded = encode_property_type(property_type)
    
    # Creating input data based on the features identified as slightly positive and sentiment analysis
    input_data = np.array([[
        accommodates, bathrooms, bedrooms, price,
        neighborhood_sentiment, host_neighborhood_sentiment, amenities_sentiment
    ] + property_type_encoded])

    # Predicting the review score using the linear regression model
    predicted_score = model.predict(input_data)
    
    return predicted_score[0]

def encode_property_type(property_type):
    # Dummy encoding for property types (example for property types: 'Entire home', 'Private room', 'Shared room')
    property_types = ['Entire home', 'Private room', 'Shared room', 'Hotel room']
    property_type_encoded = [1 if property == property_type else 0 for property in property_types]
    return property_type_encoded

def get_sentiment_score(text):
    # Analyze sentiment using VADER and return the compound score
    return analyzer.polarity_scores(text)['compound']

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
        predicted_score = predict_review_score(
            accommodates, bathrooms, bedrooms, price,
            neighborhood_sentiment, host_neighborhood_sentiment, amenities_sentiment, property_type
        )
        st.markdown(f"## 🔥 **Predicted Review Score Rating: {predicted_score:.2f}** 🔥")

# Back button to go back to main page
if st.button("Back"):
    st.session_state.page = "main"
