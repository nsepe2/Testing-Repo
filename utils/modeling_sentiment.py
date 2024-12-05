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
        b2 = B2(
            endpoint=os.getenv('B2_ENDPOINT'),
            key_id=os.getenv('B2_KEYID'),
            secret_key=os.getenv('B2_APPKEY')
        )
        bucket_name = os.getenv('B2_BUCKETNAME')
        if not bucket_name:
            raise ValueError("Bucket name not found in environment variables")
        
        b2.set_bucket(bucket_name)
        obj = b2.get_object('Final_PROJ.xlsx')
        if obj is None:
            raise ValueError("Failed to get the object from Backblaze bucket")
        
        data = pd.read_excel(obj)
        data.dropna(inplace=True)  # Remove rows with missing values
        return data
    except Exception as e:
        raise ValueError(f"Error fetching data from Backblaze: {e}")

# Train the model and save it as model.pickle
def train_and_save_model():
    # Load and preprocess data
    data = load_and_preprocess_data()

    # Extract features and target
    X = data[['accommodates', 'bathrooms', 'bedrooms', 'beds', 'price', 'neighborhood_sentiment', 'host_neighborhood_sentiment', 'amenities_sentiment', 'property_type']]
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
    with open('model.pickle', 'wb') as model_file:
        pickle.dump({'model': model, 'scaler': scaler}, model_file)

    print("Model and scaler saved successfully as model.pickle")

if __name__ == "__main__":
    train_and_save_model()






