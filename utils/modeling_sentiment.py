import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
from utils.b2 import B2

# Load environment variables
load_dotenv()

# Set Backblaze connection
b2 = B2(
    endpoint=os.getenv('B2_ENDPOINT', 's3.us-east-005.backblaze.com'),
    key_id=os.getenv('B2_KEYID'),
    secret_key=os.getenv('B2_APPKEY')
)

# Set bucket name directly
bucket_name = 'AirBnB-CSV'
b2.set_bucket(bucket_name)

# Load and preprocess dataset
def load_and_preprocess_data(remote_file_name):
    try:
        # Load the dataset from Backblaze B2
        obj = b2.get_object(remote_file_name)
        data = pd.read_csv(obj)

        # Select relevant columns and drop missing values
        columns_to_use = ['property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'price', 'review_score_rating']
        data = data[columns_to_use].dropna()

        # Encode categorical features
        property_types = data['property_type'].unique().tolist()

        def encode_property_type(property_type):
            return [1 if property == property_type else 0 for property in property_types]

        data['property_type_encoded'] = data['property_type'].apply(encode_property_type)
        return data
    except Exception as e:
        print(f"Error fetching data from Backblaze: {e}")
        return None

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
    nltk.download('vader_lexicon')
    analyzer = SentimentIntensityAnalyzer()
    # Save the analyzer to a pickle file for future reuse
    with open('analyzer.pickle', 'wb') as f:
        pickle.dump(analyzer, f)

# Load or train the model
def load_or_train_model(remote_file_name):
    if os.path.exists('model.pickle'):
        # Load existing model and scaler from pickle
        with open('model.pickle', 'rb') as model_file:
            model_data = pickle.load(model_file)
            return model_data['model'], model_data['scaler']
    else:
        # Train a new model and save it
        data = load_and_preprocess_data(remote_file_name)
        if data is not None:
            return train_model(data)
        else:
            raise ValueError("Failed to load data for training")

# Load or initialize the sentiment analyzer
def load_or_initialize_analyzer():
    if os.path.exists('analyzer.pickle'):
        # Load existing analyzer from pickle
        with open('analyzer.pickle', 'rb') as f:
            return pickle.load(f)
    else:
        # Initialize a new analyzer and save it
        initialize_analyzer()
        with open('analyzer.pickle', 'rb') as f:
            return pickle.load(f)

# Load the dataset, train the model, and initialize scaler
REMOTE_DATA = 'Airbnb Dataset_Long.csv'
model, scaler = load_or_train_model(REMOTE_DATA)
analyzer = load_or_initialize_analyzer()





