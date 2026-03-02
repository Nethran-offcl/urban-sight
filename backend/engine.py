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
    score = float(np.clip(score, 0.05, 0.98))
    
    if score < 0.4:
        category = "Low"
    elif score <= 0.7:
        category = "Medium"
    else:
        category = "High"
        
    return score, category

def get_location_features(lat, lng, hour):
    import hashlib
    
    # Create deterministic seed from location
    seed = int(hashlib.md5(
      f"{round(lat, 3)}{round(lng, 3)}".encode()
    ).hexdigest()[:8], 16) % (2**31)
    
    rng = np.random.RandomState(seed)
    
    # Distance from Bengaluru city center
    dist = ((lat - 12.9716)**2 + (lng - 77.5946)**2)**0.5
    
    # Derive features deterministically
    lighting_score = float(np.clip(8.0 - dist * 40, 2.0, 10.0))
    crowd_density = float(np.clip(0.8 - dist * 3, 0.1, 0.95))
    historical_crime_index = float(np.clip(dist * 4, 0.05, 0.90))
    police_dist_km = float(np.clip(0.5 + dist * 20, 0.5, 5.0))
    is_isolated = 1 if (crowd_density < 0.2 and lighting_score < 4) else 0
    near_transit = 1 if dist < 0.05 else 0
    
    # Night adjustment
    if hour >= 22 or hour <= 4:
      lighting_score = float(max(1.0, lighting_score - 3.0))
    
    return {
      "lighting_score": lighting_score,
      "crowd_density": crowd_density,
      "historical_crime_index": historical_crime_index,
      "police_dist_km": police_dist_km,
      "is_isolated": is_isolated,
      "near_transit": near_transit,
      "hour": hour,
      "day_of_week": 0
    }

def get_area_adjustment(lat, lng):
    # Known safer areas — higher score
    safe_zones = [
      (12.9352, 77.6245, 0.12),  # Koramangala
      (12.9784, 77.6408, 0.12),  # Indiranagar
      (12.9767, 77.5713, 0.10),  # MG Road
      (12.9719, 77.5937, 0.10),  # Brigade Road
      (13.0358, 77.5970, 0.08),  # Hebbal
      (12.9698, 77.7499, 0.08),  # Whitefield
      (12.9121, 77.6446, 0.08),  # BTM Layout
      (12.9279, 77.6271, 0.08),  # HSR Layout
    ]
    
    # Known higher risk areas — lower score
    risk_zones = [
      (12.9542, 77.4908, -0.10), # Rajarajeshwari Nagar
      (12.9902, 77.5509, -0.08), # Yeshwanthpur
      (13.0549, 77.5939, -0.07), # Yelahanka
    ]
    
    adjustment = 0.0
    for zone_lat, zone_lng, adj in safe_zones + risk_zones:
      dist = ((lat - zone_lat)**2 + (lng - zone_lng)**2)**0.5
      if dist < 0.02:
        adjustment += adj * (1 - dist / 0.02)
    
    return float(np.clip(adjustment, -0.15, 0.15))
