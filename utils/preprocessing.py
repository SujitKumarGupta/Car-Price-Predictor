import pandas as pd
import numpy as np
from datetime import datetime

def validate_input(year, mileage, brand, model):
    """Validate user inputs"""
    current_year = datetime.now().year
    
    if not (1900 <= year <= current_year):
        return False, f"Year must be between 1900 and {current_year}"
    
    if not (0 <= mileage <= 1000000):
        return False, "Mileage must be between 0 and 1,000,000"
    
    if not brand or not model:
        return False, "Brand and model must not be empty"
    
    return True, "Valid input"

def prepare_input_data(year, mileage, brand, model, feature_columns):
    """Prepare input data for prediction"""
    input_data = pd.DataFrame({
        'year': [year],
        'mileage': [mileage],
        'brand': [brand],
        'model': [model]
    })
    
    # Create dummy variables
    input_encoded = pd.get_dummies(input_data)
    
    # Ensure all features from training are present
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Reorder columns to match training data
    input_encoded = input_encoded[feature_columns]
    
    return input_encoded

def format_price(price):
    """Format price prediction"""
    return f"${price:,.2f}"
