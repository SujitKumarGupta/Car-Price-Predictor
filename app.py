import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from utils.preprocessing import validate_input, prepare_input_data, format_price
from utils.database import save_prediction, get_predictions, get_prediction_stats
from datetime import datetime
import os

# Page config
st.set_page_config(
    page_title="Indian Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    with open('model/car_price_model.pkl', 'rb') as f:
        return pickle.load(f)

def main():
    st.title("🚗 Indian Car Price Predictor")
    st.markdown("Predict used car prices in Indian Rupees (₹)")

    try:
        model_data = load_model()
        model = model_data['model']
        scaler = model_data['scaler']
        feature_columns = model_data['features']
    except FileNotFoundError:
        st.error("Model not found. Please train the model first.")
        return

    # Display prediction stats
    stats = get_prediction_stats()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Predictions", stats['total_predictions'])
    with col2:
        st.metric("Average Predicted Price", format_price(stats['avg_price']))

    # Sidebar for inputs
    st.sidebar.header("Car Details")

    year = st.sidebar.number_input("Year", min_value=1900, max_value=datetime.now().year, value=2020)
    mileage = st.sidebar.number_input("Mileage (km)", min_value=0, max_value=500000, value=50000)
    brand = st.sidebar.selectbox(
        "Brand",
        ["Maruti Suzuki", "Hyundai", "Tata", "Mahindra", "Honda", "Toyota", "Kia", "MG"]
    )
    car_model = st.sidebar.selectbox(
        "Model Type",
        ["Hatchback", "Sedan", "SUV", "Compact SUV", "MPV", "Luxury"]
    )

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

            # Save prediction to MongoDB
            prediction_data = {
                'year': year,
                'mileage': mileage,
                'brand': brand,
                'model': car_model,
                'predicted_price': float(prediction)
            }
            save_prediction(prediction_data)

            # Display prediction
            st.success(f"Predicted Price: {format_price(prediction)}")

    # Display previous predictions
    st.header("Previous Predictions")
    predictions = get_predictions()

    if predictions:
        # Convert to DataFrame and handle MongoDB specific fields
        df_predictions = pd.DataFrame(predictions)

        # Format display columns
        df_predictions['predicted_price_formatted'] = df_predictions['predicted_price'].apply(format_price)
        df_predictions['timestamp'] = pd.to_datetime(df_predictions['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

        # Remove MongoDB _id column and display
        if '_id' in df_predictions.columns:
            df_predictions = df_predictions.drop('_id', axis=1)

        # Display as table
        st.dataframe(
            df_predictions[['timestamp', 'year', 'mileage', 'brand', 'model', 'predicted_price_formatted']],
            hide_index=True
        )

        # Create visualization
        fig = px.scatter(df_predictions, x='year', y='predicted_price',
                        color='brand', size='mileage',
                        hover_data=['model', 'timestamp'],
                        title='Predictions Visualization (in ₹)',
                        labels={'predicted_price': 'Predicted Price (₹)'})
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()