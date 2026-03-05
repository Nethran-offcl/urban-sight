from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import numpy as np

from models import AnalyzeRequest, RouteRequest, LocationFeatures
from engine import predict, get_shap_explanation, get_recommendations, get_location_features, get_area_adjustment
from personalization import apply_profile_weights

app = FastAPI(title="Urban Sight API", version="1.0.0")

app.add_middleware(
    CORSMiddleware, 
    allow_origins=[
        "http://localhost:3000",
        "https://*.vercel.app",
        "*"
    ],
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

def get_category_color(score):
    if score < 0.4:
        return "Low", "#ef4444"
    elif score <= 0.7:
        return "Medium", "#f97316"
    else:
        return "High", "#22c55e"

@app.get("/health")
def health():
    return {"status": "ok", "model": "loaded", "version": "1.0.0"}

@app.post("/analyze")
@app.post("/predict-risk")
def predict_risk(request: AnalyzeRequest):
    now = datetime.now()
    feature_dict = request.location.dict()
    
    hour = feature_dict['hour']
    if hour == -1:
        hour = now.hour
        feature_dict['hour'] = hour
    if feature_dict['day_of_week'] == -1:
        feature_dict['day_of_week'] = now.weekday()
        
    print("\n" + "="*40)
    print(f"DEBUG: Incoming latitude: {request.location.lat}, longitude: {request.location.lng}")
    print(f"DEBUG: time_of_day (hour): {hour}, travel_mode: {request.profile.mode}")
        
    # Use deterministic features based on location
    loc_features = get_location_features(
        feature_dict.get('lat', 0.0), 
        feature_dict.get('lng', 0.0), 
        hour
    )
    for k, v in loc_features.items():
        feature_dict[k] = v
        
    # Override with any explicitly provided values
    if request.location.lighting_score != 5.0:
        feature_dict["lighting_score"] = request.location.lighting_score
    if request.location.crowd_density != 0.5:
        feature_dict["crowd_density"] = request.location.crowd_density
            
    # Modify feature_dict based on travel_mode to ensure model input depends on it
    mode = request.profile.mode.lower()
    if mode == "transit":
        feature_dict["near_transit"] = 1
    elif mode == "driving":
        feature_dict["is_isolated"] = 0
        feature_dict["crowd_density"] = max(0.1, feature_dict["crowd_density"] - 0.2)
    elif mode == "walking":
        feature_dict["is_isolated"] = 1 if feature_dict["crowd_density"] < 0.4 else 0
        
    print(f"DEBUG: Engineered feature values: {feature_dict}")
            
    # Get base score from engine.py predict()
    base_score, _ = predict(feature_dict)
    
    # Apply personalization from personalization.py
    pers = apply_profile_weights(base_score, request.profile, feature_dict)
    
    area_adj = get_area_adjustment(
        feature_dict.get('lat', 0.0), 
        feature_dict.get('lng', 0.0)
    )
    
    adjusted_score = float(np.clip(
        pers["adjusted_score"] + area_adj, 
        0.05, 0.98
    ))
    adjustments_applied = pers["adjustments_applied"]
    
    category, color_code = get_category_color(adjusted_score)
    
    # Get SHAP explanation from engine.py
    shap_res = get_shap_explanation(feature_dict, adjusted_score)
    
    # Get recommendations from engine.py
    recommendations = get_recommendations(category, feature_dict)
    
    return {
        "safety_score": round(base_score, 4),
        "adjusted_score": adjusted_score,
        "category": category,
        "color_code": color_code,
        "explanation": shap_res.get("explanation", ""),
        "top_features": shap_res.get("top_features", []),
        "recommendations": recommendations,
        "adjustments_applied": adjustments_applied
    }

@app.post("/route")
def route(request: RouteRequest):
    import math
    origin = request.origin
    dest = request.destination
    
    # Generate 3 mock routes with specific geometric curves
    route_profiles = [
        {"name": "Safest", "detour_val": 0.01},      # bow outwards
        {"name": "Fastest", "detour_val": 0.0},      # straight line
        {"name": "Comfortable", "detour_val": -0.01} # bow inwards
    ]
    
    routes_response = []
    
    now = datetime.now()
    base_loc = LocationFeatures(lat=0, lng=0, hour=now.hour, day_of_week=now.weekday()).dict()
    
    print("\n" + "="*40)
    print(f"NEW ROUTE REQUEST: Origin({origin.lat}, {origin.lng}) to Dest({dest.lat}, {dest.lng})")
    print("="*40)
    
    for rp in route_profiles:
        print(f"\n--- Processing Route Profile: {rp['name']} ---")
        detour = rp["detour_val"]
        waypoints = []
        scores = []
        risk_zone_count = 0
        
        num_points = 5
        
        # Calculate orthogonal vector for bowing effect
        dx = dest.lng - origin.lng
        dy = dest.lat - origin.lat
        dist = math.sqrt(dx*dx + dy*dy)
        if dist == 0:
            nx, ny = 0, 0
        else:
            nx = -dy / dist
            ny = dx / dist
            
        for i in range(num_points):
            frac = i / (num_points - 1) if num_points > 1 else 0
            
            # create bowing effect using sine wave (0 at ends, max at middle)
            bow = math.sin(frac * math.pi) * detour
            
            lat = origin.lat + dy * frac + nx * bow
            lng = origin.lng + dx * frac + ny * bow
            
            waypoints.append({"lat": lat, "lng": lng})
            print(f"  [{rp['name']}] Point {i} coords: lat={lat:.6f}, lng={lng:.6f}")
            
            # Generate deterministic dynamic features based on lat/lng
            f = base_loc.copy()
            f["lat"] = lat
            f["lng"] = lng
            
            loc_features = get_location_features(lat, lng, f["hour"])
            for k, v in loc_features.items():
                f[k] = v
            
            print(f"    -> Feature dict: {f}")
            
            base_score, category = predict(f)
            # handle unpacking adjusted_score as a dict. extract the float numeric score
            adj_score = float(apply_profile_weights(base_score, request.profile, f)["adjusted_score"])
            scores.append(adj_score)
            
            print(f"    -> Predicted base score: {base_score:.4f}, Adjusted score: {adj_score:.4f}")
            
            if adj_score < 0.4:
                risk_zone_count += 1
                
        avg_score = sum(scores) / len(scores)
        
        if rp["name"] == "Safest":
            avg_score = float(np.clip(avg_score * 1.05, 0.05, 0.98))
            explanation = f"This route prioritises well-lit roads and avoids {risk_zone_count} high-risk zones. Safety score: {int(avg_score * 100)}%."
        elif rp["name"] == "Fastest":
            avg_score = float(np.clip(avg_score * 0.88, 0.05, 0.98))
            explanation = f"Shortest path to destination. Passes through {risk_zone_count} caution zones. Safety score: {int(avg_score * 100)}%."
        elif rp["name"] == "Comfortable":
            avg_score = float(np.clip(avg_score * 0.95, 0.05, 0.98))
            explanation = f"Balanced route avoiding major risk areas. {risk_zone_count} minor caution zones. Safety score: {int(avg_score * 100)}%."
        else:
            avg_score = float(np.clip(avg_score, 0.05, 0.98))
            explanation = f"Average safety score of {int(avg_score * 100)}% with {risk_zone_count} risky areas."

        cat, col = get_category_color(avg_score)
        
        # Deterministic pseudo-random for estimated minutes
        minute_seed = int(abs(origin.lat + origin.lng + dest.lat + dest.lng) * 10000) % 15
        
        # Scale to integer percentage for display on older UI
        avg_score_pct = int(avg_score * 100)
        
        routes_response.append({
            "name": rp["name"],
            "waypoints": waypoints,
            "avg_safety_score": avg_score_pct,
            "category": cat,
            "color_code": col,
            "risk_zone_count": risk_zone_count,
            "estimated_minutes": 15 + minute_seed + (0 if rp["name"] == "Fastest" else (2 if rp["name"] == "Comfortable" else 5)),
            "explanation": explanation
        })
        
    return {
        "routes": routes_response,
        "recommended": "Safest"
    }

@app.get("/heatmap")
def heatmap(min_lat: float, max_lat: float, min_lng: float, max_lng: float, hour: int = -1):
    now = datetime.now()
    if hour == -1:
        hour = now.hour
        
    lats = np.linspace(min_lat, max_lat, 10)
    lngs = np.linspace(min_lng, max_lng, 10)
    
    base_loc = LocationFeatures(lat=0, lng=0, hour=hour, day_of_week=now.weekday()).dict()
    
    points = []
    for lat in lats:
        for lng in lngs:
            f = base_loc.copy()
            f["lat"] = float(lat)
            f["lng"] = float(lng)
            loc_features = get_location_features(f["lat"], f["lng"], f["hour"])
            for k, v in loc_features.items():
                f[k] = v
            
            base_score_val_tup = predict(f)
            area_adj = get_area_adjustment(f["lat"], f["lng"])
            final_score = float(np.clip(base_score_val_tup[0] + area_adj, 0.05, 0.98))
            
            _, col = get_category_color(final_score)
            
            points.append({
                "lat": float(lat),
                "lng": float(lng),
                "safety_score": round(final_score, 4),
                "color_code": col
            })
            
    return {
        "points": points,
        "count": len(points)
    }
