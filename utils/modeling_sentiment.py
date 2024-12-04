import numpy as np
from sklearn.linear_model import LinearRegression
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    # Analyze sentiment using VADER and return the compound score
    return analyzer.polarity_scores(text)['compound']

def encode_property_type(property_type):
    # Dummy encoding for property types
    property_types = ['Entire home', 'Private room', 'Shared room', 'Hotel room']
    return [1 if property == property_type else 0 for property in property_types]

def train_model():
    # Generate sample training data (for demonstration purposes)
    np.random.seed(42)
    X_train = np.random.rand(100, 7)
    y_train = np.random.rand(100) * 5

    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
