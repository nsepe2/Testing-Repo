import os
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dotenv import load_dotenv
from utils.b2 import B2

# Initialize VADER sentiment analyzer
analyzer = None

# Load environment variables
load_dotenv()

# Initialize Backblaze B2 connection
b2 = B2(
    endpoint=os.getenv('B2_ENDPOINT'),
    key_id=os.getenv('B2_KEYID'),
    secret_key=os.getenv('B2_APPKEY')
)
b2.set_bucket(os.getenv('B2_BUCKETNAME'))

# Load and preprocess dataset
def load_and_preprocess_data(remote_file_name):
    # Load the dataset from Backblaze B2
    data = b2.get_df(remote_file_name)

    # Select relevant columns and drop missing values
    columns_to_use = ['property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'price', 'review_score_rating']
    data = data[columns_to_use].dropna()

    # Encode categorical features
    property_types = data['property_type'].unique().tolist()

    def encode_property_type(property_type):
        return [1 if property == property_type else 0 for property in property_types]

    data['property_type_encoded'] = data['property_type'].apply(encode_property_type)
    return data

# Train the model
def train_model(data):
    # Prepare feature matrix X and target vector y
    other_features = data[['accommodates', 'bathrooms', 'bedrooms', 'beds', 'price']].values
    encoded_property_types = np.array(data['property_type_encoded'].tolist())

    X_train = np.hstack((other_features, encoded_property_types))
    y_train = data['review_score_rating'].values

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the trained model and scaler to a pickle file
    with open('model.pickle', 'wb') as model_file:
        pickle.dump({'model': model, 'scaler': scaler}, model_file)

    return model, scaler

# Initialize sentiment analyzer
def initialize_analyzer():
    global analyzer
    nltk.download('vader_lexicon')
    analyzer = SentimentIntensityAnalyzer()

# Get sentiment score
def get_sentiment_score(text):
    # Analyze sentiment using VADER and return the compound score
    return analyzer.polarity_scores(text)['compound']

# Encode property type
def encode_property_type(property_type):
    # Dummy encoding for property types
    property_types = ['Entire home', 'Private room', 'Shared room', 'Hotel room']
    return [1 if property == property_type else 0 for property in property_types]

# Load the dataset, train the model, and initialize scaler
REMOTE_DATA = 'Final_PROJ.xlsx'
data = load_and_preprocess_data(REMOTE_DATA)
model, scaler = train_model(data)

# Ensure VADER lexicon is downloaded and initialize analyzer
initialize_analyzer()



