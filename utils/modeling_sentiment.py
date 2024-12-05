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
        b2.set_bucket(os.getenv('B2_BUCKETNAME'))
        obj = b2.get_object('Final_PROJ.xlsx')
        data = pd.read_excel(obj)
        print("Data loaded successfully. Preview:")
        print(data.head())
        return data
    except Exception as e:
        print(f"Error fetching data from Backblaze: {e}")
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






