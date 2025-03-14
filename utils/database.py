import pymongo
from datetime import datetime
from typing import Dict, List, Any
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB connection settings
MONGO_HOST = "0.0.0.0"
MONGO_PORT = 27017
MONGO_DB = "car_price_predictor"
MONGO_COLLECTION = "predictions"

def get_database():
    """Get MongoDB database connection"""
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017/")

        # Test the connection
        client.server_info()
        logger.info("Successfully connected to MongoDB")
        db = client[MONGO_DB]

        # Create collection if it doesn't exist
        if MONGO_COLLECTION not in db.list_collection_names():
            db.create_collection(MONGO_COLLECTION)
            logger.info(f"Created collection: {MONGO_COLLECTION}")

        return db
    except Exception as e:
        logger.error(f"MongoDB Connection Error: {str(e)}")
        raise

def save_prediction(prediction_data: Dict[str, Any]) -> bool:
    """Save a prediction to MongoDB"""
    try:
        db = get_database()
        predictions = db[MONGO_COLLECTION]
        prediction_data['timestamp'] = datetime.now()
        result = predictions.insert_one(prediction_data)
        logger.info(f"Saved prediction with ID: {result.inserted_id}")
        return bool(result.inserted_id)
    except Exception as e:
        logger.error(f"Error saving prediction: {str(e)}")
        return False

def get_predictions(limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieve predictions from MongoDB"""
    try:
        db = get_database()
        predictions = db[MONGO_COLLECTION]
        return list(predictions.find().sort('timestamp', -1).limit(limit))
    except Exception as e:
        logger.error(f"Error retrieving predictions: {str(e)}")
        return []

def get_prediction_stats() -> Dict[str, Any]:
    """Get basic statistics about predictions"""
    try:
        db = get_database()
        predictions = db[MONGO_COLLECTION]
        total = predictions.count_documents({})

        if total > 0:
            avg_price = list(predictions.aggregate([
                {'$group': {'_id': None, 'avg': {'$avg': '$predicted_price'}}}
            ]))[0]['avg']
        else:
            avg_price = 0

        return {
            'total_predictions': total,
            'avg_price': avg_price
        }
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return {'total_predictions': 0, 'avg_price': 0}