import numpy as np
import pandas as pd

def generate_data(num_rows=5000):
    np.random.seed(42)
    
    # Bengaluru Coordinates: lat 12.83-13.14, lng 77.46-77.78
    lat = np.random.uniform(12.83, 13.14, num_rows)
    lng = np.random.uniform(77.46, 77.78, num_rows)
    
    # hour (0-23, uniform distribution for simplicity, night hours will have specific features modified)
    hour = np.random.randint(0, 24, num_rows)
    
    # day_of_week (0-6)
    day_of_week = np.random.randint(0, 7, num_rows)
    
    # lighting_score (float 1-10, lower during night hours)
    lighting_score = []
    for h in hour:
        if h >= 22 or h <= 4:
            lighting_score.append(np.clip(np.random.normal(3, 1.5), 1, 10))
        else:
            lighting_score.append(np.clip(np.random.normal(7, 2), 1, 10))
    lighting_score = np.array(lighting_score)
    
    # crowd_density (float 0-1, higher during 8-10am and 5-8pm)
    crowd_density = []
    for h in hour:
        if (8 <= h <= 10) or (17 <= h <= 20):
            crowd_density.append(np.clip(np.random.normal(0.8, 0.15), 0, 1))
        else:
            crowd_density.append(np.clip(np.random.normal(0.3, 0.2), 0, 1))
    crowd_density = np.array(crowd_density)
    
    # historical_crime_index (float 0-1, spatially clustered via a simple generic function)
    historical_crime_index = np.clip(np.random.normal(0.5, 0.2, num_rows) + np.sin(lat*100)*0.1 + np.cos(lng*100)*0.1, 0, 1)
    
    # police_dist_km (float 0.5-5.0)
    police_dist_km = np.random.uniform(0.5, 5.0, num_rows)
    
    # is_isolated (int 0/1: 1 if crowd_density<0.2 AND lighting<4)
    is_isolated = ((crowd_density < 0.2) & (lighting_score < 4)).astype(int)
    
    # near_transit (int 0/1: randomly assign 30% of rows as 1)
    near_transit = np.random.choice([0, 1], size=num_rows, p=[0.7, 0.3])
    
    # Target: safety_score (float 0.0-1.0)
    base = 1.0 - (historical_crime_index * 0.35) \
               - ((1 - lighting_score / 10) * 0.25) \
               - (is_isolated * 0.15) \
               - ((police_dist_km / 5.0) * 0.10) \
               - (crowd_density * 0.05)
               
    night_mask = (hour >= 22) | (hour <= 4)
    base[night_mask] -= 0.10
    
    transit_mask = (near_transit == 1)
    base[transit_mask] += 0.08
    
    base += np.random.normal(0, 0.04, num_rows)
    safety_score = np.clip(base, 0.05, 0.98)
    
    df = pd.DataFrame({
        'lat': lat,
        'lng': lng,
        'hour': hour,
        'day_of_week': day_of_week,
        'lighting_score': lighting_score,
        'crowd_density': crowd_density,
        'historical_crime_index': historical_crime_index,
        'police_dist_km': police_dist_km,
        'is_isolated': is_isolated,
        'near_transit': near_transit,
        'safety_score': safety_score
    })
    
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print(f"\nSafety Score Mean: {safety_score.mean():.4f}")
    print("Safety Score Distribution (quantiles):")
    print(df['safety_score'].quantile([0, 0.25, 0.5, 0.75, 1.0]))
    
    df.to_csv('urban_safety.csv', index=False)
    print("\nDataset saved to 'urban_safety.csv'")

if __name__ == "__main__":
    generate_data()
