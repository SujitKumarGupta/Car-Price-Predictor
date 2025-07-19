import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

def load_and_preprocess_data(generate_new=False):
    filepath = 'data/car_data.csv'
    
    if os.path.exists(filepath) and not generate_new:
        print("✅ Loading existing data...")
        data = pd.read_csv(filepath)
    else:
        print("🔄 Generating new synthetic data...")
        np.random.seed(42)
        brands = ['Maruti Suzuki', 'Hyundai', 'Tata', 'Mahindra', 'Honda', 'Toyota', 'Kia', 'MG']
        models = ['Hatchback', 'Sedan', 'SUV', 'Compact SUV', 'MPV', 'Luxury']
        data = pd.DataFrame({
            'year': np.random.randint(2000, 2024, 1000),
            'mileage': np.random.randint(0, 200000, 1000),
            'brand': np.random.choice(brands, 1000),
            'model': np.random.choice(models, 1000),
            'price': np.random.randint(300000, 5000000, 1000)
        })

        data['age'] = 2024 - data['year']
        data['price_per_year'] = data['price'] / data['age']
        data['mileage_per_year'] = data['mileage'] / data['age']

        data.to_csv("./car_data.csv", index=False)
        print(f"📁 Saved new data to {"./car_data.csv"}")

    # Prepare features
    X = pd.get_dummies(data.drop('price', axis=1))
    y = data['price']
    return X, y

def train_model(generate_new_data=False):
    X, y = load_and_preprocess_data(generate_new=generate_new_data)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameter tuning with RandomizedSearchCV
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    model = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=20, cv=5, scoring='r2', n_jobs=-1, random_state=42)
    random_search.fit(X_train_scaled, y_train)

    best_model = random_search.best_estimator_

    # Save the model and scaler
    os.makedirs('model', exist_ok=True)
    with open('model/car_price_model.pkl', 'wb') as f:
        pickle.dump({'model': best_model, 'scaler': scaler, 'features': X.columns}, f)
    print("💾 Model and scaler saved to model/car_price_model.pkl")

    # Metrics
    y_train_pred = best_model.predict(X_train_scaled)
    y_test_pred = best_model.predict(X_test_scaled)
    train_score = r2_score(y_train, y_train_pred)
    test_score = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    return train_score, test_score, train_mae, test_mae

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)

    # Set to True if you want to regenerate data
    regenerate_data = False

    train_score, test_score, train_mae, test_mae = train_model(generate_new_data=regenerate_data)

    print(f"\n📊 Training R² Score: {train_score:.4f}")
    print(f"📊 Testing R² Score: {test_score:.4f}")
    print(f"📉 Training MAE: ₹{train_mae:,.2f}")
    print(f"📉 Testing MAE: ₹{test_mae:,.2f}")
