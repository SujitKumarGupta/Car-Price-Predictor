import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from utils.preprocessing import validate_input, prepare_input_data, format_price
from utils.database import save_prediction, get_predictions
from datetime import datetime
import os

# Page config
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    with open('model/car_price_model.pkl', 'rb') as f:
        return pickle.load(f)

def main():
    st.title("🚗 Car Price Predictor")

    try:
        model_data = load_model()
        model = model_data['model']
        scaler = model_data['scaler']
        feature_columns = model_data['features']
    except FileNotFoundError:
        st.error("Model not found. Please train the model first.")
        return

    # Sidebar for inputs
    st.sidebar.header("Car Details")

    year = st.sidebar.number_input("Year", min_value=1900, max_value=datetime.now().year, value=2020)
    mileage = st.sidebar.number_input("Mileage", min_value=0, max_value=1000000, value=50000)
    brand = st.sidebar.selectbox("Brand", ["Toyota", "Honda", "Ford", "BMW", "Mercedes"])
    car_model = st.sidebar.selectbox("Model", ["Sedan", "SUV", "Truck", "Coupe"])

    if st.sidebar.button("Predict Price"):
        # Validate input
        is_valid, message = validate_input(year, mileage, brand, car_model)

        if not is_valid:
            st.error(message)
        else:
            # Prepare input data
            input_data = prepare_input_data(year, mileage, brand, car_model, feature_columns)

            # Scale the input
            input_scaled = scaler.transform(input_data)

            # Make prediction
            prediction = model.predict(input_scaled)[0]

            # Create prediction record
            prediction_data = {
                'year': year,
                'mileage': mileage,
                'brand': brand,
                'model': car_model,
                'predicted_price': float(prediction)
            }

            # Save to MongoDB
            save_prediction(prediction_data)

            # Display prediction
            st.success(f"Predicted Price: {format_price(prediction)}")

    # Display previous predictions from MongoDB
    st.header("Previous Predictions")

    # Get predictions from MongoDB
    predictions = get_predictions()
    if predictions:
        # Convert to DataFrame
        df_predictions = pd.DataFrame(predictions)

        # Drop MongoDB _id and format timestamp
        df_predictions = df_predictions.drop('_id', axis=1)
        df_predictions['timestamp'] = pd.to_datetime(df_predictions['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

        # Display as table
        st.dataframe(df_predictions)

        # Create visualization
        fig = px.scatter(df_predictions, x='year', y='predicted_price',
                        color='brand', size='mileage',
                        hover_data=['model', 'timestamp'],
                        title='Predictions Visualization')
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()