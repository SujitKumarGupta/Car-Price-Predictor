from pymongo import MongoClient
from datetime import datetime
import os

def get_database():
    """Get MongoDB database connection"""
    client = MongoClient('mongodb://localhost:27017/')
    db = client['car_price_predictor']
    return db

def save_prediction(prediction_data):
    """Save prediction to MongoDB"""
    db = get_database()
    predictions = db.predictions
    
    # Add timestamp
    prediction_data['timestamp'] = datetime.now()
    
    # Insert prediction
    result = predictions.insert_one(prediction_data)
    return result.inserted_id

def get_predictions(limit=100):
    """Get recent predictions from MongoDB"""
    db = get_database()
    predictions = db.predictions
    
    # Get recent predictions, sorted by timestamp
    cursor = predictions.find().sort('timestamp', -1).limit(limit)
    return list(cursor)
