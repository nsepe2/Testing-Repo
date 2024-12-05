import os
import sys
from io import BytesIO

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from utils.b2 import B2
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Function to load data from Backblaze and perform sentiment analysis
def load_and_preprocess_data():
    load_dotenv()  # Load environment variables
    try:
        # Set up Backblaze connection
        b2 = B2(
            endpoint=os.getenv('B2_ENDPOINT'),
            key_id=os.getenv('B2_KEYID'),
            secret_key=os.getenv('B2_APPKEY')
        )
        
        # Set the bucket
        bucket_name = os.getenv('B2_BUCKETNAME')
        if not bucket_name:
            raise ValueError("Bucket name not found in environment variables")
        
        b2.set_bucket(bucket_name)
        
        # Retrieve the file from Backblaze
        obj = b2.get_object('Final_PROJ.xlsx')
        if obj is None:
            raise ValueError("Failed to get the object from Backblaze bucket")

        # Debug: Check what the 'obj' is
        print(f"Retrieved object type: {type(obj)}")
        
        # Convert StreamingBody to BytesIO and load data into DataFrame
        try:
            file_content = obj.read()  # Ensure this is being read as bytes
            print(f"Content type: {type(file_content)}")  # Debug: Check type of content
        except AttributeError:
            raise ValueError("The object retrieved does not have a 'read' method. The data type might be incorrect.")
        
        if not isinstance(file_content, bytes):
            raise ValueError("Error fetching data from Backblaze: Retrieved object is not in bytes format.")
        
        # Try reading the Excel file content
        try:
            data = pd.read_excel(BytesIO(file_content), engine='openpyxl')  # Adding engine parameter
        except Exception as e:
            raise ValueError(f"Failed to read Excel file from Backblaze: {e}")
        
        # Data preprocessing
        data.dropna(inplace=True)  # Remove rows with missing values

        # Perform sentiment analysis
        analyzer = SentimentIntensityAnalyzer()

        def get_sentiment_score(text):
            if pd.isna(text):
                return 0  # Default score if there's no text
            sentiment = analyzer.polarity_scores(text)
            return sentiment['compound']

        # Add sentiment scores to the DataFrame
        data['neighborhood_sentiment'] = data['neighborhood_overview'].apply(get_sentiment_score)
        data['host_neighborhood_sentiment'] = data['host_neighborhood'].apply(get_sentiment_score)
        data['amenities_sentiment'] = data['amenities'].apply(get_sentiment_score)

        print("Data loaded, preprocessed, and sentiment analysis performed successfully.")
        return data

    except Exception as e:
        raise ValueError(f"Error fetching data from Backblaze: {e}")



# Function to encode property type
def encode_property_type(X):
    if 'property_type' in X.columns:
        X = pd.get_dummies(X, columns=['property_type'])
    return X

# Train the model and save it as model.pickle
def train_and_save_model():
    try:
        # Load and preprocess data
        data = load_and_preprocess_data()

        # Extract features and target
        feature_columns = [
            'accommodates', 'bathrooms', 'bedrooms', 'beds', 'price',
            'neighborhood_sentiment', 'host_neighborhood_sentiment',
            'amenities_sentiment', 'property_type'
        ]
        
        if 'review_scores_rating' not in data.columns:
            raise ValueError("Target column 'review_scores_rating' not found in dataset")

        X = data[feature_columns]
        y = data['review_scores_rating']

        # One-hot encode 'property_type'
        X = encode_property_type(X)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train the model
        model = LinearRegression()
        model.fit(X_scaled, y)

        # Save the model, scaler, and expected features to a pickle file
        save_path = os.path.join(os.path.dirname(__file__), 'utils', 'model.pickle')
        expected_features = list(X.columns)  # Capture the expected features

        with open(save_path, 'wb') as model_file:
            pickle.dump({'model': model, 'scaler': scaler, 'expected_features': expected_features}, model_file)

        print("Model, scaler, and expected features saved successfully as model.pickle")

    except Exception as e:
        print(f"Error during model training and saving: {e}")

# Function to load the trained model
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'utils', 'model.pickle')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as model_file:
            model_data = pickle.load(model_file)
            print("Loaded existing model from model.pickle")
            return model_data['model'], model_data['scaler'], model_data['expected_features']
    else:
        raise FileNotFoundError("model.pickle not found in the expected location")

if __name__ == "__main__":
    train_and_save_model()














