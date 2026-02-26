import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def train_model():
    print("Loading data...")
    df = pd.read_csv('urban_safety.csv')
    
    # 1. Drop lat/lng from features
    df = df.drop(columns=['lat', 'lng'])
    
    # 2. X = all columns except safety_score, y = safety_score
    X = df.drop(columns=['safety_score'])
    y = df['safety_score']
    
    feature_names = X.columns.tolist()
    
    # 3. Train/test split 80/20, random_state=42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Fit StandardScaler on X_train only, transform both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Train RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # 6. Print MAE, RMSE, R^2 on test set
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Model Evaluation ---")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2:  {r2:.4f}")
    
    # 7. Save model as urban_sight_model.pkl using joblib
    joblib.dump(model, 'urban_sight_model.pkl')
    print("\nModel saved to 'urban_sight_model.pkl'")
    
    # 8. Save scaler as scaler.pkl using joblib
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved to 'scaler.pkl'")
    
    # 9. Plot feature importance bar chart, save as feature_importance.png
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.bar(range(X.shape[1]), importances[sorted_indices], align='center')
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in sorted_indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Feature importance plot saved to 'feature_importance.png'")


def predict(raw_dict):
    """
    Accepts dict with all feature keys
    Loads scaler.pkl and urban_sight_model.pkl
    Returns: { "safety_score": float, "safety_pct": int, "category": "Low/Medium/High" }
    Thresholds: score < 0.4 = Low, 0.4-0.7 = Medium, > 0.7 = High
    """
    # Load model and scaler
    model = joblib.load('urban_sight_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Convert dict to DataFrame to maintain column order/names
    df_input = pd.DataFrame([raw_dict])
    
    expected_cols = ['hour', 'day_of_week', 'lighting_score', 'crowd_density', 
                     'historical_crime_index', 'police_dist_km', 'is_isolated', 'near_transit']
                     
    # Ensure columns exist and order matches
    X_input = df_input[expected_cols]
    
    # Transform
    X_scaled = scaler.transform(X_input)
    
    # Predict
    score = float(model.predict(X_scaled)[0])
    
    # Thresholds
    if score < 0.4:
        category = "Low"
    elif score <= 0.7:
        category = "Medium"
    else:
        category = "High"
        
    safety_pct = int(round(score * 100))
    
    return {
        "safety_score": round(score, 4),
        "safety_pct": safety_pct,
        "category": category
    }

if __name__ == "__main__":
    if not os.path.exists('urban_safety.csv'):
        print("urban_safety.csv not found. Please run data_factory.py first.")
    else:
        train_model()
        
        # Test predict() with a sample input
        print("\n--- Testing predict() ---")
        sample_input = {
            'hour': 23,
            'day_of_week': 5,
            'lighting_score': 3.5,
            'crowd_density': 0.1,
            'historical_crime_index': 0.8,
            'police_dist_km': 4.0,
            'is_isolated': 1,
            'near_transit': 0
        }
        
        print("Sample Input:", sample_input)
        result = predict(sample_input)
        print("Prediction Result:", result)
