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
            
            seed1 = math.sin(lat * 1000 + lng * 1000)
            seed2 = math.cos(lat * 800 - lng * 1200)
            seed3 = math.sin(lat * 500) * math.cos(lng * 500)
            
            f["lighting_score"] = float(np.clip(5.0 + 5.0 * seed1, 0.0, 10.0))
            f["crowd_density"] = float(np.clip(0.5 + 0.5 * seed2, 0.0, 1.0))
            f["historical_crime_index"] = float(np.clip(0.5 + 0.5 * seed3, 0.0, 1.0))
            f["police_dist_km"] = float(np.clip(2.5 + 2.5 * seed1 * seed2, 0.0, 5.0))
            f["is_isolated"] = 1 if seed2 > 0.5 else 0
            f["near_transit"] = 1 if seed3 > 0.5 else 0
            
            print(f"    -> Feature dict: {f}")
            
            base_score, _ = predict(f)
            adj_score = apply_profile_weights(base_score, request.profile, f)["adjusted_score"]
            scores.append(adj_score)
            
            print(f"    -> Predicted base score: {base_score:.4f}, Adjusted score: {adj_score:.4f}")
            
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
        
        # Deterministic pseudo-random for estimated minutes
        minute_seed = int((origin.lat + origin.lng + dest.lat + dest.lng) * 10000) % 15
        
        routes_response.append({
            "name": rp["name"],
            "waypoints": waypoints,
            "avg_safety_score": round(avg_score, 4),
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
