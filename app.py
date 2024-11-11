import os
import pickle
import streamlit as st
from dotenv import load_dotenv

# Import B2 from utils/b2.py
from utils.b2 import B2

# ------------------------------------------------------
#                      APP CONSTANTS
# ------------------------------------------------------
REMOTE_DATA = 'Airbnb Dataset_Final.xlsx'  # Update if your file name is different

# ------------------------------------------------------
#                        CONFIG
# ------------------------------------------------------
# Load environment variables
load_dotenv()

# Set Backblaze connection with new credentials
b2 = B2(
    endpoint=os.getenv('B2_ENDPOINT', 's3.us-east-005.backblazeb2.com'),  # Update if endpoint changes
    key_id=os.getenv('B2_KEYID', '005491ab29352f00000000003'),
    secret_key=os.getenv('B2_APPKEY', 'K005urBSkXoICdWzCf8QtT/CPxQCMy8')
)
# ------------------------------------------------------
#                         APP
# ------------------------------------------------------
# ------------------------------
# PART 0 : Overview
# ------------------------------
st.write(
'''
# Airbnb Listings Application
Welcome to the Airbnb Listings Web App. Please select whether you are a buyer or a seller.
''')

b2.set_bucket(os.getenv('B2_BUCKETNAME', 'AirBnB-Data'))
df_coffee = b2.get_df(REMOTE_DATA)

# Load the sentiment analysis model
with open('./model.pickle', 'rb') as f:
    analyzer = pickle.load(f)

# Average sentiment scores for the whole dataset
benchmarks = df_coffee[['neg', 'neu', 'pos', 'compound']].agg(['mean', 'median'])

# ------------------------------
# PART 1 : User Type Selection
# ------------------------------
user_type = st.radio("Are you a buyer or a seller?", ("Buyer", "Seller"))

# ------------------------------
# PART 2 : Buyer Section
# ------------------------------
if user_type == "Buyer":
    st.write("## View Listings on Map")
    m = folium.Map(location=[df_coffee['latitude'].mean(), df_coffee['longitude'].mean()], zoom_start=12)

    for _, row in df_coffee.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Price: ${row['price']}, Bedrooms: {row['bedrooms']}, Bathrooms: {row['bathrooms']}",
        ).add_to(m)

    st_folium(m, width=700, height=500)

# ------------------------------
# PART 3 : Seller Section
# ------------------------------
elif user_type == "Seller":
    st.write("## Enter Your Property Details")

    price = st.number_input("Enter the price ($):", min_value=0)
    bedrooms = st.number_input("Number of bedrooms:", min_value=0, step=1)
    bathrooms = st.number_input("Number of bathrooms:", min_value=0, step=1)
    amenities = st.text_area("List the amenities (comma separated):")
    neighborhood_description = st.text_area("Describe the neighborhood:")
    location = st.text_input("Enter the location (latitude, longitude):")

    if st.button("Get Predicted Score"):
        predicted_score = random.uniform(0, 5)
        st.write(f"### Predicted Score for Your Listing: {predicted_score:.2f} out of 5")
