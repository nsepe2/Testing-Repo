import os
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from utils.b2 import B2

# Function to load data from Backblaze
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
        
        # Read data from Excel file
        data = pd.read_excel(obj)
        
        # Data preprocessing
        data.dropna(inplace=True)  # Remove rows with missing values
        print("Data loaded and preprocessed successfully.")
        return data
    except Exception as e:
        raise ValueError(f"Error fetching data from Backblaze: {e}")

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
        
        if 'review_score_rating' not in data.columns:
            raise ValueError("Target column 'review_score_rating' not found in dataset")

        X = data[feature_columns]
        y = data['review_score_rating']

        # One-hot encode 'property_type'
        X = pd.get_dummies(X, columns=['property_type'])

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train the model
        model = LinearRegression()
        model.fit(X_scaled, y)

        # Save the model and scaler to a pickle file
        save_path = os.path.join(os.path.dirname(__file__), 'model.pickle')
        with open(save_path, 'wb') as model_file:
            pickle.dump({'model': model, 'scaler': scaler}, model_file)

        print("Model and scaler saved successfully as model.pickle")

    except Exception as e:
        print(f"Error during model training and saving: {e}")

if __name__ == "__main__":
    train_and_save_model()







