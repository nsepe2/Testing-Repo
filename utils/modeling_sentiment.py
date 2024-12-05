import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize VADER sentiment analyzer
analyzer = None

# Load and preprocess dataset
def load_and_preprocess_data(file_path):
    # Load the dataset using pandas
    data = pd.read_excel(file_path)

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
file_path = '/Users/nasase/Downloads/Final_PROJ.xlsx'
data = load_and_preprocess_data(file_path)
model, scaler = train_model(data)

# Ensure VADER lexicon is downloaded and initialize analyzer
initialize_analyzer()


