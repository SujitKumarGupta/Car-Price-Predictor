import pymongo
from datetime import datetime

# MongoDB connection settings
MONGO_HOST = "0.0.0.0"
MONGO_PORT = 27017
MONGO_DB = "car_price_predictor"
MONGO_COLLECTION = "predictions"

def init_database():
    try:
        # Connect to MongoDB
        client = pymongo.MongoClient(f"mongodb://{MONGO_HOST}:{MONGO_PORT}/")
        
        # Create database and collection
        db = client[MONGO_DB]
        if MONGO_COLLECTION not in db.list_collection_names():
            db.create_collection(MONGO_COLLECTION)
        
        # Insert a test document to ensure the database is visible
        predictions = db[MONGO_COLLECTION]
        test_prediction = {
            'timestamp': datetime.now(),
            'year': 2020,
            'mileage': 50000,
            'brand': 'Maruti Suzuki',
            'model': 'Hatchback',
            'predicted_price': 500000
        }
        predictions.insert_one(test_prediction)
        
        print(f"Successfully initialized database '{MONGO_DB}' with collection '{MONGO_COLLECTION}'")
        print("You can now use 'show dbs' in mongo shell to see the database")
        
    except Exception as e:
        print(f"Error initializing database: {str(e)}")

if __name__ == "__main__":
    init_database()
