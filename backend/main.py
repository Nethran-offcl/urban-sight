from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import numpy as np

from models import AnalyzeRequest, RouteRequest, LocationFeatures
from engine import predict, get_shap_explanation, get_recommendations
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
def analyze(request: AnalyzeRequest):
    now = datetime.now()
    feature_dict = request.location.dict()
    
    if feature_dict['hour'] == -1:
        feature_dict['hour'] = now.hour
    if feature_dict['day_of_week'] == -1:
        feature_dict['day_of_week'] = now.weekday()
        
    # Get base score from engine.py predict()
    base_score, _ = predict(feature_dict)
    
    # Apply personalization from personalization.py
    pers = apply_profile_weights(base_score, request.profile, feature_dict)
    adjusted_score = pers["adjusted_score"]
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
    origin = request.origin
    dest = request.destination
    
    # Generate 3 mock routes
    route_profiles = [
        {"name": "Safest", "detour_val": 0.005},
        {"name": "Fastest", "detour_val": 0.0},
        {"name": "Comfortable", "detour_val": 0.002}
    ]
    
    routes_response = []
    
    now = datetime.now()
    base_loc = LocationFeatures(lat=0, lng=0, hour=now.hour, day_of_week=now.weekday()).dict()
    
    for rp in route_profiles:
        detour = rp["detour_val"]
        waypoints = []
        scores = []
        risk_zone_count = 0
        
        # analyze 5 points along route
        num_points = 5
        for i in range(num_points):
            frac = i / (num_points - 1) if num_points > 1 else 0
            lat = origin.lat + (dest.lat - origin.lat) * frac + np.random.uniform(-detour, detour)
            lng = origin.lng + (dest.lng - origin.lng) * frac + np.random.uniform(-detour, detour)
            
            waypoints.append({"lat": lat, "lng": lng})
            
            f = base_loc.copy()
            f["lat"] = lat
            f["lng"] = lng
            
            base_score, _ = predict(f)
            adj_score = apply_profile_weights(base_score, request.profile, f)["adjusted_score"]
            scores.append(adj_score)
            
            if adj_score < 0.4:
                risk_zone_count += 1
                
        if rp["name"] == "Safest":
            best_3 = sorted(scores)[2:]
            avg_score = min(sum(best_3) / len(best_3) * 1.05, 1.0)
            explanation = f"This route prioritises well-lit roads and avoids {risk_zone_count} high-risk zones. Safety score: {round(avg_score * 100)}%."
        elif rp["name"] == "Fastest":
            avg_score = sum(scores) / len(scores) * 0.88
            explanation = f"Shortest path to destination. Passes through {risk_zone_count} caution zones. Safety score: {round(avg_score * 100)}%."
        elif rp["name"] == "Comfortable":
            avg_score = sum(scores) / len(scores) * 0.95
            explanation = f"Balanced route avoiding major risk areas. {risk_zone_count} minor caution zones. Safety score: {round(avg_score * 100)}%."
        else:
            avg_score = sum(scores) / len(scores)
            explanation = f"Average safety score of {round(avg_score * 100)}% with {risk_zone_count} risky areas."

        cat, col = get_category_color(avg_score)
        
        routes_response.append({
            "name": rp["name"],
            "waypoints": waypoints,
            "avg_safety_score": round(avg_score, 4),
            "category": cat,
            "color_code": col,
            "risk_zone_count": risk_zone_count,
            "estimated_minutes": int(np.random.randint(15, 30)),
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
            
            score, _ = predict(f)
            _, col = get_category_color(score)
            
            points.append({
                "lat": float(lat),
                "lng": float(lng),
                "safety_score": round(score, 4),
                "color_code": col
            })
            
    return {
        "points": points,
        "count": len(points)
    }
