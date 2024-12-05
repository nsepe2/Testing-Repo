import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Function to encode property type
def encode_property_type(X):
    if 'property_type' in X.columns:
        X = pd.get_dummies(X, columns=['property_type'])
    return X

# Load the trained model
def load_model():
    try:
        # Load model and scaler from pickle file
        model_path = os.path.join(os.path.dirname(__file__), 'model.pickle')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as model_file:
                model_data = pickle.load(model_file)
                print("Loaded existing model from model.pickle")
                return model_data['model'], model_data['scaler'], model_data['expected_features']
        else:
            raise FileNotFoundError("model.pickle not found in the expected location.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise











