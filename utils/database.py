import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from typing import Dict, List, Any

def get_database_connection():
    """Get PostgreSQL database connection"""
    try:
        return psycopg2.connect(os.environ['DATABASE_URL'])
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        raise

def save_prediction(prediction_data: Dict[str, Any]) -> bool:
    """Save a prediction to PostgreSQL"""
    try:
        with get_database_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO predictions (year, mileage, brand, model, predicted_price)
                    VALUES (%s, %s, %s, %s, %s)
                    """, (
                        prediction_data['year'],
                        prediction_data['mileage'],
                        prediction_data['brand'],
                        prediction_data['model'],
                        prediction_data['predicted_price']
                    ))
                conn.commit()
                return True
    except Exception as e:
        print(f"Error saving prediction: {str(e)}")
        return False

def get_predictions(limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieve predictions from PostgreSQL"""
    try:
        with get_database_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT timestamp, year, mileage, brand, model, predicted_price
                    FROM predictions
                    ORDER BY timestamp DESC
                    LIMIT %s
                    """, (limit,))
                return cur.fetchall()
    except Exception as e:
        print(f"Error retrieving predictions: {str(e)}")
        return []

def get_prediction_stats() -> Dict[str, Any]:
    """Get basic statistics about predictions"""
    try:
        with get_database_connection() as conn:
            with conn.cursor() as cur:
                # Get total count
                cur.execute("SELECT COUNT(*) FROM predictions")
                total_predictions = cur.fetchone()[0]

                # Get average price
                cur.execute("SELECT AVG(predicted_price) FROM predictions")
                avg_price = cur.fetchone()[0] or 0

                return {
                    'total_predictions': total_predictions,
                    'avg_price': float(avg_price)
                }
    except Exception as e:
        print(f"Error getting stats: {str(e)}")
        return {'total_predictions': 0, 'avg_price': 0}