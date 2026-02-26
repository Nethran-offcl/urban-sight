def apply_profile_weights(base_score, profile, features):
    score = base_score
    adjustments_applied = []
    
    # Unpack profile
    mode = profile.mode
    group_size = profile.group_size
    is_night = profile.is_night
    gender_sensitive = profile.gender_sensitive
    
    # Unpack features
    lighting = features.get('lighting_score', 5.0)
    is_isolated = features.get('is_isolated', 0)
    
    # Apply in order, compound them
    if mode == "walking" and is_night:
        score *= 0.75
        adjustments_applied.append("Walking at night reduces safety score.")
        
    if mode == "cycling" and is_night:
        score *= 0.85
        adjustments_applied.append("Cycling at night reduces safety score.")
        
    if mode == "driving":
        score = min(score * 1.05, 1.0)
        adjustments_applied.append("Driving generally increases safety.")
        
    if group_size >= 4:
        score = min(score + 0.08, 1.0)
        adjustments_applied.append("Large group size enhances safety.")
        
    if group_size == 1 and is_night:
        score *= 0.90
        adjustments_applied.append("Traveling alone at night reduces safety.")
        
    if gender_sensitive and lighting < 4:
        score *= 0.88
        adjustments_applied.append("Gender sensitive profile penalty for poor lighting.")
        
    if gender_sensitive and is_isolated == 1:
        score *= 0.85
        adjustments_applied.append("Gender sensitive profile penalty for isolated areas.")
        
    # Clip final score to [0.0, 1.0]
    score = max(0.0, min(score, 1.0))
    
    return {
        "adjusted_score": round(score, 4),
        "adjustments_applied": adjustments_applied
    }
