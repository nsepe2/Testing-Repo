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

# Function to encode property type
def encode_property_type(X):
    if 'property_type' in X.columns:
        X = pd.get_dummies(X, columns=['property_type'])
    return X

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
    try:
        # Load the existing model
        model, scaler, expected_features = load_model()
        print("Model loaded successfully, ready for use in predictions.")
    except FileNotFoundError as e:
        print(e)












