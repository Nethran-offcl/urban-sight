import joblib
import pandas as pd
import numpy as np
import shap

try:
    model = joblib.load('urban_sight_model.pkl')
    scaler = joblib.load('scaler.pkl')
    explainer = shap.TreeExplainer(model)
except Exception as e:
    print(f"Warning: Failed to load models. Ensure urban_sight_model.pkl and scaler.pkl exist. Error: {e}")
    model, scaler, explainer = None, None, None

def get_shap_explanation(feature_dict, safety_score):
    if explainer is None:
        return {"explanation": "Model not loaded.", "top_features": []}
        
    expected_cols = ['hour', 'day_of_week', 'lighting_score', 'crowd_density', 
                     'historical_crime_index', 'police_dist_km', 'is_isolated', 'near_transit']
    
    df_input = pd.DataFrame([feature_dict])[expected_cols]
    X_scaled = scaler.transform(df_input)
    
    shap_values = explainer.shap_values(X_scaled)[0]
    
    feature_map = {
        'lighting_score': 'street lighting',
        'historical_crime_index': 'historical crime rate',
        'crowd_density': 'crowd density',
        'police_dist_km': 'distance from nearest police station',
        'is_isolated': 'area isolation',
        'hour': 'time of day',
        'near_transit': 'proximity to transit hub',
        'day_of_week': 'day of week'
    }
    
    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap)[-2:][::-1]
    
    f1_key = expected_cols[top_indices[0]]
    f2_key = expected_cols[top_indices[1]]
    
    f1 = feature_map.get(f1_key, f1_key)
    f2 = feature_map.get(f2_key, f2_key)
    
    if safety_score < 0.4:
        explanation = f"Safety concern: {f1} and {f2} are the main risk factors."
    elif safety_score <= 0.7:
        explanation = f"Moderate safety: Caution advised due to {f1}."
    else:
        explanation = f"This area is relatively safe. {f1} contributes positively."
        
    return {
        "explanation": explanation,
        "top_features": [f1, f2]
    }

def get_recommendations(category, feature_dict):
    if category == "Low":
        return [
            "Avoid poorly lit streets when walking.",
            "Consider traveling with a larger group.",
            "Use transit options with safer hubs nearby."
        ]
    elif category == "Medium":
        return [
            "Stay alert in less crowded areas.",
            "Keep emergency contacts readily accessible."
        ]
    else:
        return [
            "Standard positive precautions are sufficient."
        ]

def predict(feature_dict):
    """Returns base safety_score and category."""
    if model is None:
        return 0.5, "Medium"
        
    expected_cols = ['hour', 'day_of_week', 'lighting_score', 'crowd_density', 
                     'historical_crime_index', 'police_dist_km', 'is_isolated', 'near_transit']
    df_input = pd.DataFrame([feature_dict])[expected_cols]
    X_scaled = scaler.transform(df_input)
    
    score = float(model.predict(X_scaled)[0])
    
    if score < 0.4:
        category = "Low"
    elif score <= 0.7:
        category = "Medium"
    else:
        category = "High"
        
    return score, category
