import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import os

def load_and_preprocess_data():
    # Sample car data (you can replace this with your actual dataset)
    data = pd.DataFrame({
        'year': np.random.randint(2000, 2024, 1000),
        'mileage': np.random.randint(0, 200000, 1000),
        'brand': np.random.choice(['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes'], 1000),
        'model': np.random.choice(['Sedan', 'SUV', 'Truck', 'Coupe'], 1000),
        'price': np.random.randint(5000, 100000, 1000)
    })
    
    # Save the data
    data.to_csv('data/car_data.csv', index=False)
    
    # Prepare features
    X = pd.get_dummies(data.drop('price', axis=1))
    y = data['price']
    
    return X, y

def train_model():
    X, y = load_and_preprocess_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Save the model and scaler
    with open('model/car_price_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'features': X.columns}, f)
    
    # Calculate metrics
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    return train_score, test_score

if __name__ == "__main__":
    os.makedirs('model', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    train_score, test_score = train_model()
    print(f"Training Score: {train_score:.4f}")
    print(f"Testing Score: {test_score:.4f}")
