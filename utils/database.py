import pymongo
from datetime import datetime
from typing import Dict, List, Any

# MongoDB connection
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["car_price_predictor"]
predictions = db["predictions"]

def save_prediction(prediction_data: Dict[str, Any]) -> None:
    """Save a prediction to MongoDB"""
    prediction_data['timestamp'] = datetime.now()
    predictions.insert_one(prediction_data)

def get_predictions(limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieve predictions from MongoDB"""
    return list(predictions.find().sort('timestamp', -1).limit(limit))

def get_prediction_stats() -> Dict[str, Any]:
    """Get basic statistics about predictions"""
    return {
        'total_predictions': predictions.count_documents({}),
        'avg_price': list(predictions.aggregate([
            {'$group': {'_id': None, 'avg': {'$avg': '$predicted_price'}}}
        ]))[0]['avg'] if predictions.count_documents({}) > 0 else 0
    }
