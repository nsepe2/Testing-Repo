import os
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
from utils.b2 import B2

# Initialize VADER sentiment analyzer
analyzer = None

def initialize_analyzer():
    global analyzer
    analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    # Analyze sentiment using VADER and return the compound score
    return analyzer.polarity_scores(text)['compound']

def encode_property_type(property_type):
    # Dummy encoding for property types
    property_types = ['Entire home', 'Private room', 'Shared room', 'Hotel room']
    return [1 if property == property_type else 0 for property in property_types]

def fetch_data():
    load_dotenv()
    try:
        b2 = B2(
            endpoint=os.getenv('B2_ENDPOINT'),
            key_id=os.getenv('B2_KEYID'),
            secret_key=os.getenv('B2_APPKEY')
        )
        bucket_name = os.getenv('B2_BUCKETNAME')
        
        if not bucket_name:
            raise ValueError("Bucket name not found in environment variables")

        b2.set_bucket(bucket_name)
        st.write("Connection to Backblaze successful!")
        return None
    except Exception as e:
        st.error(f"Error connecting to Backblaze: {e}")
        return None

def load_or_train_model():
    if os.path.exists('model.pickle'):
        # Load existing model and scaler from pickle
        with open('model.pickle', 'rb') as model_file:
            model_data = pickle.load(model_file)
            return model_data['model'], model_data['scaler']
    else:
        # Train a new model and save it
        data = fetch_data()
        if data is not None:
            return train_model(data)
        else:
            raise ValueError("Failed to load data for training")

def train_model(data):
    # Preprocess the data and train the model
    features = data[['accommodates', 'bathrooms', 'bedrooms', 'beds', 'price', 'neighborhood_sentiment', 'host_neighborhood_sentiment', 'amenities_sentiment']]
    target = data['review_score_rating']

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    model = LinearRegression()
    model.fit(features_scaled, target)

    # Save the model and scaler to a pickle file
    model_data = {
        'model': model,
        'scaler': scaler
    }
    with open('model.pickle', 'wb') as model_file:
        pickle.dump(model_data, model_file)

    return model, scaler






