from pydantic import BaseModel, Field

class LocationFeatures(BaseModel):
    lat: float
    lng: float
    hour: int = Field(default=-1)
    day_of_week: int = Field(default=-1)
    lighting_score: float = Field(default=5.0)
    crowd_density: float = Field(default=0.5)
    historical_crime_index: float = Field(default=0.3)
    police_dist_km: float = Field(default=1.5)
    is_isolated: int = Field(default=0)
    near_transit: int = Field(default=0)

class UserProfile(BaseModel):
    mode: str = Field(default="walking")
    group_size: int = Field(default=1)
    is_night: bool = Field(default=False)
    gender_sensitive: bool = Field(default=False)

class AnalyzeRequest(BaseModel):
    location: LocationFeatures
    profile: UserProfile

class RoutePoint(BaseModel):
    lat: float
    lng: float

class RouteRequest(BaseModel):
    origin: RoutePoint
    destination: RoutePoint
    profile: UserProfile
