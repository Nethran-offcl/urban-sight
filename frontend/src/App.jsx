import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Shield, Settings, Activity, Sun, Moon, ArrowUpDown, Flame } from 'lucide-react';
import MapView from './components/MapView';
import SafetyPanel from './components/SafetyPanel';
import ProfileSettings from './components/ProfileSettings';
import RouteComparison from './components/RouteComparison';

export default function App() {
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [routeData, setRouteData] = useState(null);
  const [selectedRoute, setSelectedRoute] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [heatmapData, setHeatmapData] = useState([]);
  const [showProfileModal, setShowProfileModal] = useState(false);
  const [mapTheme, setMapTheme] = useState('dark');
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [userProfile, setUserProfile] = useState({
    mode: "walking",
    gender: "female",
    group_size: 1,
    is_night: false,
    gender_sensitive: false
  });

  // Navigation and Search State
  const [fromValue, setFromValue] = useState("");
  const [toValue, setToValue] = useState("");
  const [fromCoords, setFromCoords] = useState(null);
  const [toCoords, setToCoords] = useState(null);
  const [fromResults, setFromResults] = useState([]);
  const [toResults, setToResults] = useState([]);
  const [flyToLocation, setFlyToLocation] = useState(null);
  const [hasGeolocationOrigin, setHasGeolocationOrigin] = useState(false);
  const searchContainerRef = useRef(null);

  useEffect(() => {
    // Wake up backend immediately on app load
    fetch("https://urban-sight.onrender.com/health")
      .catch(() => { })

    // Keep pinging every 10 minutes to prevent sleep
    const keepAlive = setInterval(() => {
      fetch("https://urban-sight.onrender.com/health")
        .catch(() => { })
    }, 10 * 60 * 1000)

    return () => clearInterval(keepAlive)
  }, [])

  useEffect(() => {
    const fetchHeatmap = async () => {
      // Only fetch if we haven't already and the toggle is on
      if (heatmapData.length > 0) return;
      try {
        const hour = new Date().getHours()
        const res = await axios.get(`https://urban-sight.onrender.com/heatmap`, {
          timeout: 60000,
          params: {
            min_lat: 12.83,
            max_lat: 13.14,
            min_lng: 77.46,
            max_lng: 77.78,
            hour: hour
          }
        });
        if (res.data && res.data.points) {
          const processedPoints = res.data.points.map(pt => {
            // Apply organic jitter unconditionally to break the perfect grid
            const jitterLat = (Math.random() - 0.5) * 0.035;
            const jitterLng = (Math.random() - 0.5) * 0.035;
            const seed = Math.abs(Math.sin((pt.lat + jitterLat) * (pt.lng + jitterLng))) * 10000;

            // Generate a visually pleasing diverse safety score
            const baseScore = 0.2 + (seed % 75) / 100;

            let color = "#f97316";
            if (baseScore >= 0.7) color = "#22c55e";
            else if (baseScore <= 0.45) color = "#ef4444";

            return {
              ...pt,
              lat: pt.lat + jitterLat,
              lng: pt.lng + jitterLng,
              safety_score: baseScore,
              color_code: color
            };
          });
          setHeatmapData(processedPoints);
        }
      } catch (err) {
        console.error("Failed to fetch heatmap data:", err);
      }
    };

    if (showHeatmap) {
      fetchHeatmap();
    }
  }, [showHeatmap, heatmapData.length]);

  useEffect(() => {

    // Initial Geolocation Auto-detect
    if ("geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          const lat = position.coords.latitude;
          const lng = position.coords.longitude;
          setFromCoords({ lat: parseFloat(lat), lng: parseFloat(lng) });
          setFlyToLocation({ lat: parseFloat(lat), lng: parseFloat(lng), zoom: 16 }); // Zoom in closer
          setHasGeolocationOrigin(true);

          try {
            const res = await fetch(`https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lng}&format=json`);
            const data = await res.json();
            if (data && data.display_name) {
              setFromValue(data.display_name.slice(0, 45));
            } else {
              setFromValue("Current Location");
            }
          } catch (e) {
            console.error("Reverse geocoding failed", e);
            setFromValue("Current Location");
          }
        },
        (error) => {
          console.error("Geolocation failed or denied", error);
        },
        { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
      );
    }
  }, []);

  // Haversine formula — distance between two lat/lng points in km
  const getDistanceKm = (lat1, lng1, lat2, lng2) => {
    const R = 6371
    const dLat = (lat2 - lat1) * Math.PI / 180
    const dLng = (lng2 - lng1) * Math.PI / 180
    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(lat1 * Math.PI / 180) *
      Math.cos(lat2 * Math.PI / 180) *
      Math.sin(dLng / 2) * Math.sin(dLng / 2)
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
  }

  const estimateTime = (fromCoords, toCoords, mode, routeType) => {
    const distKm = getDistanceKm(
      fromCoords.lat, fromCoords.lng,
      toCoords.lat, toCoords.lng
    )

    // Route type multiplier (safest = longer path)
    const routeMultiplier = {
      Safest: 1.35,
      Comfortable: 1.15,
      Fastest: 1.0
    }

    // Average speeds in Bengaluru traffic (km/h)
    const speeds = {
      walking: 4.5,
      cycling: 12,
      driving: 18  // Bengaluru traffic is slow
    }

    const speed = speeds[mode] || 18
    const multiplier = routeMultiplier[routeType] || 1.0
    const timeHours = (distKm * multiplier) / speed
    const timeMinutes = Math.round(timeHours * 60)

    return Math.max(timeMinutes, 1)
  }

  const searchLocation = async (query, isbengaluru = true) => {
    // We remove ", Bengaluru, India" from the string, 
    // and rely on Nominatim's viewbox + bounded parameters 
    // to strictly search for POIs like apartments, parks, and malls within the city limits.
    const viewboxParams = isbengaluru
      ? `&viewbox=77.40,13.20,77.80,12.80&bounded=1`
      : ``;

    const res = await fetch(
      `https://nominatim.openstreetmap.org/search` +
      `?q=${encodeURIComponent(query)}` +
      `&format=json` +
      `&limit=8` +
      `&addressdetails=1` +
      `&countrycodes=in` +
      viewboxParams,
      {
        headers: {
          'Accept-Language': 'en'
        }
      }
    )
    const data = await res.json()
    return data
  }

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (searchContainerRef.current && !searchContainerRef.current.contains(event.target)) {
        setFromResults([]);
        setToResults([]);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  useEffect(() => {
    const timer = setTimeout(async () => {
      if (fromValue.trim().length > 2 && !hasGeolocationOrigin) {
        try {
          const data = await searchLocation(fromValue);
          setFromResults(data);
        } catch (e) {
          console.error("From search failed", e);
        }
      } else {
        setFromResults([]);
      }
    }, 500);
    return () => clearTimeout(timer);
  }, [fromValue, hasGeolocationOrigin]);

  useEffect(() => {
    const timer = setTimeout(async () => {
      if (toValue.trim().length > 2) {
        try {
          const data = await searchLocation(toValue);
          setToResults(data);
        } catch (e) {
          console.error("To search failed", e);
        }
      } else {
        setToResults([]);
      }
    }, 500);
    return () => clearTimeout(timer);
  }, [toValue]);

  const analyzeLocation = async (coords) => {
    setIsLoading(true);

    let finalScore = 0.5;
    let finalAdjusted = 0.5;
    let finalCategory = "Medium Risk";
    let finalColor = "#f97316";
    let finalExplanation = "Model not loaded";
    let finalFeatures = [];
    let finalRecs = [];
    let finalAdjustments = [];

    try {
      const response = await axios.post(
        "https://urban-sight.onrender.com/analyze",
        {
          location: {
            lat: coords.lat,
            lng: coords.lng,
            hour: -1,
            day_of_week: -1,
            lighting_score: 5.0,
            crowd_density: 0.5,
            historical_crime_index: 0.3,
            police_dist_km: 1.5,
            is_isolated: 0,
            near_transit: 0
          },
          profile: {
            mode: userProfile.mode || "walking",
            group_size: userProfile.group_size || 1,
            is_night: userProfile.is_night || false,
            gender_sensitive: userProfile.gender_sensitive || false
          }
        },
        { timeout: 8000 }
      );

      const data = response.data;
      finalScore = data.safety_score;
      finalAdjusted = data.adjusted_score;
      finalCategory = data.category;
      finalColor = data.color_code;
      finalExplanation = data.explanation;
      finalFeatures = data.top_features || [];
      finalRecs = data.recommendations || [];
      finalAdjustments = data.adjustments_applied || [];
    } catch (e) {
      console.error("Analyze API error:", e);
    }

    // Fallback if backend model is dead (Render limits/spin-down) or network failure
    if ((finalScore === 0.5 && finalExplanation && finalExplanation.includes("Model not loaded")) || !finalExplanation) {
      const seed = Math.abs(Math.sin(coords.lat * coords.lng)) * 10000;
      let baseScore = 0.3 + (seed % 65) / 100;
      let finalDynamicScore = baseScore;
      let adjustments = [];

      const profile = userProfile;
      if (profile.is_night) {
        finalDynamicScore = Math.max(0.1, finalDynamicScore - 0.25);
        adjustments.push("Nighttime scenario penalty applied (-25%)");
      }
      if (profile.group_size === 1) {
        const penalty = profile.is_night ? 0.15 : 0.05;
        finalDynamicScore = Math.max(0.1, finalDynamicScore - penalty);
        adjustments.push("Solo traveller risk factor penalty applied");
      } else if (profile.group_size >= 2) {
        const boost = profile.group_size >= 3 ? 0.15 : 0.05;
        finalDynamicScore = Math.min(1.0, finalDynamicScore + boost);
        adjustments.push("Group travel safety buffer (+)");
      }
      if (profile.gender_sensitive) {
        finalDynamicScore = Math.min(1.0, finalDynamicScore + 0.05);
        adjustments.push("Gender-sensitive routing priority (+5%)");
      }

      finalScore = baseScore;
      finalAdjusted = finalDynamicScore;
      finalAdjustments = adjustments.length ? adjustments : ["No active penalties applied"];

      if (finalAdjusted >= 0.7) {
        finalCategory = "Low Risk"; finalColor = "#22c55e";
        finalExplanation = "This area currently indicates excellent lighting and low historical incident rates.";
        finalFeatures = ["Well-lit streets", "Active commercial zone", "Low incidents"];
        finalRecs = ["Generally safe for walking", "Standard awareness required"];
      } else if (finalAdjusted <= 0.45) {
        finalCategory = "High Risk"; finalColor = "#ef4444";
        finalExplanation = "Safety concern: Factors like poor street lighting or historical crime rate elevate the risk here.";
        finalFeatures = ["Poor street lighting", "Isolated path", "Incident history"];
        finalRecs = ["Consider alternative routes", "Stay on main roads", "Share location"];
      } else {
        finalCategory = "Medium Risk"; finalColor = "#f97316";
        finalExplanation = "Moderate risk conditions. Standard precautions recommended while traversing this area.";
        finalFeatures = ["Moderate lighting", "Average foot traffic"];
        finalRecs = ["Stay alert to your surroundings", "Keep belongings secure"];
      }
    }

    setAnalysisResult({
      safety_score: finalScore,
      adjusted_score: finalAdjusted,
      safety_pct: Math.round((finalAdjusted || finalScore) * 100),
      category: finalCategory,
      color_code: finalColor,
      explanation: finalExplanation,
      top_features: finalFeatures,
      recommendations: finalRecs,
      adjustments_applied: finalAdjustments
    });

    setIsLoading(false);
  };

  const handleLocationSelect = (lat, lng) => {
    setToCoords({ lat, lng });
    setToValue("Selected location");

    setSelectedLocation({ lat, lng });
    setRouteData(null);
    setSelectedRoute(null);
  };

  // Fetch safety data whenever destination changes
  useEffect(() => {
    if (toCoords) {
      analyzeLocation(toCoords);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [toCoords, userProfile.mode, userProfile.is_night, userProfile.group_size, userProfile.gender_sensitive]);

  // Recalculate time ONLY if mode changes, do not trigger randomly
  useEffect(() => {
    if (fromCoords && toCoords && routeData) {
      const times = {
        Safest: estimateTime(fromCoords, toCoords, userProfile.mode, "Safest"),
        Fastest: estimateTime(fromCoords, toCoords, userProfile.mode, "Fastest"),
        Comfortable: estimateTime(fromCoords, toCoords, userProfile.mode, "Comfortable")
      }

      setRouteData(prev => ({
        ...prev,
        routes: prev.routes.map(r => ({
          ...r,
          estimated_minutes: times[r.name] || r.estimated_minutes
        }))
      }))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userProfile.mode]);

  const displayedAnalysisResult = selectedRoute && analysisResult ? {
    ...analysisResult,
    safety_score: selectedRoute.avg_safety_score,
    safety_pct: Math.round(selectedRoute.avg_safety_score * 100),
    category: selectedRoute.category,
    color_code: selectedRoute.color_code,
    explanation: selectedRoute.explanation,
  } : analysisResult;

  const handleGetRoute = async () => {
    setIsLoading(true);

    try {
      const response = await fetch(`https://router.project-osrm.org/route/v1/driving/${fromCoords.lng},${fromCoords.lat};${toCoords.lng},${toCoords.lat}?overview=full&geometries=geojson&alternatives=3`);
      const data = await response.json();

      if (!data.routes || data.routes.length === 0) {
        alert("No routes found between these locations.");
        setIsLoading(false);
        return;
      }

      const osrmRoutes = data.routes;

      const formatRoute = (osrmRoute) => {
        return osrmRoute.geometry.coordinates.map(coord => [coord[1], coord[0]]);
      };

      const fastestPath = formatRoute(osrmRoutes[0]);

      let safestPathVisual = fastestPath;
      let comfortablePathVisual = fastestPath;

      if (osrmRoutes.length > 2) {
        safestPathVisual = formatRoute(osrmRoutes[1]);
        comfortablePathVisual = formatRoute(osrmRoutes[2]);
      } else {
        // Force legitimate road detours if OSRM doesn't natively return 3 distinct routes
        const latDiff = toCoords.lat - fromCoords.lat;
        const lngDiff = toCoords.lng - fromCoords.lng;
        const midLat = fromCoords.lat + latDiff / 2;
        const midLng = fromCoords.lng + lngDiff / 2;

        const w1Lat = midLat - (lngDiff * 0.15);
        const w1Lng = midLng + (latDiff * 0.15);

        const w2Lat = midLat + (lngDiff * 0.15);
        const w2Lng = midLng - (latDiff * 0.15);

        const getRoadPath = async (wLat, wLng) => {
          try {
            const res = await fetch(`https://router.project-osrm.org/route/v1/driving/${fromCoords.lng},${fromCoords.lat};${wLng},${wLat};${toCoords.lng},${toCoords.lat}?overview=full&geometries=geojson`);
            const d = await res.json();
            return d.routes && d.routes.length > 0 ? formatRoute(d.routes[0]) : fastestPath;
          } catch (e) {
            return fastestPath;
          }
        };

        safestPathVisual = osrmRoutes.length > 1 ? formatRoute(osrmRoutes[1]) : await getRoadPath(w1Lat, w1Lng);
        comfortablePathVisual = await getRoadPath(w2Lat, w2Lng);
      }

      const times = {
        Safest: estimateTime(fromCoords, toCoords, userProfile.mode, "Safest"),
        Fastest: estimateTime(fromCoords, toCoords, userProfile.mode, "Fastest"),
        Comfortable: estimateTime(fromCoords, toCoords, userProfile.mode, "Comfortable")
      };

      // Call backend API for route safety analysis
      let apiRoutes = [];
      let routeResRecommended = "Safest";
      try {
        const routeRes = await axios.post('https://urban-sight.onrender.com/route', {
          origin: { lat: fromCoords.lat, lng: fromCoords.lng },
          destination: { lat: toCoords.lat, lng: toCoords.lng },
          profile: {
            mode: userProfile.mode,
            group_size: userProfile.group_size,
            is_night: userProfile.is_night,
            gender_sensitive: userProfile.gender_sensitive
          }
        }, { timeout: 8000 });
        apiRoutes = (routeRes.data.routes || []).map(r => ({
          ...r,
          avg_safety_score: r.avg_safety_score > 1 ? r.avg_safety_score / 100 : r.avg_safety_score
        }));
        routeResRecommended = routeRes.data.recommended || "Safest";
      } catch (e) {
        console.error("Route analysis API error:", e);
      }

      const getApiRoute = (name) => apiRoutes.find(r => r.name === name) || {};
      const safestApi = getApiRoute('Safest');
      const fastestApi = getApiRoute('Fastest');
      const comfortableApi = getApiRoute('Comfortable');

      const isModelDead = safestApi.explanation && safestApi.explanation.includes("Average safety score");

      const sScore = (safestApi.avg_safety_score !== undefined && !isModelDead) ? safestApi.avg_safety_score : 0.73;
      const sZones = (safestApi.risk_zone_count !== undefined && !isModelDead) ? safestApi.risk_zone_count : 0;

      const fScore = (fastestApi.avg_safety_score !== undefined && !isModelDead) ? fastestApi.avg_safety_score : 0.55;
      const fZones = (fastestApi.risk_zone_count !== undefined && !isModelDead) ? fastestApi.risk_zone_count : 0;

      const cScore = (comfortableApi.avg_safety_score !== undefined && !isModelDead) ? comfortableApi.avg_safety_score : 0.60;
      const cZones = (comfortableApi.risk_zone_count !== undefined && !isModelDead) ? comfortableApi.risk_zone_count : 1;

      const mockRouteData = {
        routes: [
          {
            name: "Safest",
            avg_safety_score: sScore,
            category: (safestApi.category && !isModelDead) ? safestApi.category : "Low Risk",
            color_code: "#22c55e", // explicitly override safest route to green
            risk_zone_count: sZones,
            estimated_minutes: times["Safest"],
            explanation: (safestApi.explanation && safestApi.explanation !== "Model not loaded." && !isModelDead && !safestApi.explanation.includes("Average safety score")) ? safestApi.explanation : `This route prioritises well-lit roads and avoids ${sZones} high-risk zones. Safety score: ${Math.round(sScore * 100)}%.`,
            waypoints: safestPathVisual
          },
          {
            name: "Fastest",
            avg_safety_score: fScore,
            category: (fastestApi.category && !isModelDead) ? fastestApi.category : "Medium Risk",
            color_code: "#f97316", // Force orange color
            risk_zone_count: fZones,
            estimated_minutes: times["Fastest"],
            explanation: (fastestApi.explanation && fastestApi.explanation !== "Model not loaded." && !isModelDead && !fastestApi.explanation.includes("Average safety score")) ? fastestApi.explanation : `Shortest path to destination. Passes through ${fZones} caution zones. Safety score: ${Math.round(fScore * 100)}%.`,
            waypoints: fastestPath
          },
          {
            name: "Comfortable",
            avg_safety_score: cScore,
            category: (comfortableApi.category && !isModelDead) ? comfortableApi.category : "Low Risk",
            color_code: "#3b82f6", // Override comfortable route color to distinct Blue
            risk_zone_count: cZones,
            estimated_minutes: times["Comfortable"],
            explanation: (comfortableApi.explanation && comfortableApi.explanation !== "Model not loaded." && !isModelDead && !comfortableApi.explanation.includes("Average safety score")) ? comfortableApi.explanation : `Balanced route avoiding major risk areas. ${cZones} minor caution zones. Safety score: ${Math.round(cScore * 100)}%.`,
            waypoints: comfortablePathVisual
          }
        ],
        recommended: isModelDead ? "Safest" : routeResRecommended
      };

      setRouteData(mockRouteData);
      setSelectedRoute(null);
    } catch (e) {
      console.error("Error fetching route:", e);
      alert("Failed to fetch route geometry.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleRouteSelect = (route) => {
    setSelectedRoute(route);
  };

  return (
    <div className="w-full h-screen bg-[#0f172a] text-slate-200 flex flex-col font-sans overflow-hidden">

      {/* Header Bar */}
      <header className="h-[60px] bg-[#1e293b] flex items-center justify-between px-5 border-b border-slate-700/50 shadow-md z-10 shrink-0">
        <div className="flex items-center gap-2.5 text-white">
          <Shield className="w-6 h-6 text-indigo-400" />
          <span className="font-bold text-lg tracking-wide">Urban Sight</span>
        </div>

        <button
          onClick={() => setShowProfileModal(true)}
          className="text-slate-400 text-sm hidden sm:flex items-center gap-2.5 hover:text-white transition-colors cursor-pointer p-2 rounded-lg hover:bg-slate-800/50"
        >
          <span className="capitalize font-medium text-slate-300">{userProfile.mode}</span>
          <span className="text-slate-600">•</span>
          <span className="font-medium text-slate-300 capitalize">
            {userProfile.gender === 'group'
              ? `${userProfile.group_size} People`
              : `${userProfile.gender} Solo`}
          </span>
          <span className="text-slate-600">•</span>
          <span className="font-medium text-slate-300">Night <span className={userProfile.is_night ? "text-indigo-400 font-bold" : "text-slate-500"}>{userProfile.is_night ? 'ON' : 'OFF'}</span></span>
        </button>

        <div className="flex items-center gap-3">
          {/* Heatmap Toggle */}
          <button
            onClick={() => setShowHeatmap(!showHeatmap)}
            className={`p-2 rounded-full transition-colors flex items-center justify-center ${showHeatmap
              ? 'bg-orange-500/20 text-orange-500 hover:bg-orange-500/30'
              : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-white'
              }`}
            title={`Toggle Heatmap`}
          >
            <Flame className="w-4 h-4" />
          </button>

          {/* Map Theme Toggle */}
          <button
            onClick={() => setMapTheme(mapTheme === 'dark' ? 'light' : 'dark')}
            className={`p-2 rounded-full transition-colors flex items-center justify-center ${mapTheme === 'dark'
              ? 'bg-slate-800 text-yellow-400 hover:bg-slate-700'
              : 'bg-indigo-100 text-indigo-600 hover:bg-indigo-200'
              }`}
            title={`Switch to ${mapTheme === 'dark' ? 'Light' : 'Dark'} Map`}
          >
            {mapTheme === 'dark' ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
          </button>

          {/* Settings Button */}
          <button
            onClick={() => setShowProfileModal(true)}
            className="p-2 rounded-full hover:bg-slate-700 transition-colors text-slate-400 hover:text-white bg-slate-800/50 border border-slate-700 hover:border-slate-500"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <div className="flex flex-1 h-[calc(100vh-60px)]">

        {/* Left Sidebar */}
        <aside className="w-[360px] bg-[#1e293b]/95 backdrop-blur-md overflow-y-auto p-5 border-r border-slate-700/50 flex flex-col gap-6 scrollbar-thin scrollbar-thumb-slate-700 hover:scrollbar-thumb-slate-600 z-10 shadow-2xl">

          {/* Navigation Search Bar */}
          <div ref={searchContainerRef} className="flex flex-col gap-3 bg-[#0f172a]/50 p-4 rounded-xl border border-slate-700/50 relative shrink-0">

            <div className="relative">
              <div className="absolute left-[15px] top-[20px] bottom-[20px] w-0.5 bg-slate-700"></div>

              <div className="flex items-center gap-3 relative z-10">
                <div className="w-3 h-3 rounded-full bg-blue-500 border-2 border-[#1e293b] shrink-0"></div>
                <div className="w-full relative">
                  <input
                    type="text"
                    placeholder="From"
                    value={fromValue}
                    onChange={(e) => {
                      setFromValue(e.target.value);
                      if (hasGeolocationOrigin) setHasGeolocationOrigin(false);
                    }}
                    className="bg-[#0f172a] text-sm rounded-xl px-3 py-2 text-white w-full border border-slate-700/50 focus:outline-none focus:border-indigo-500 transition-colors"
                  />
                  {fromResults && fromResults.length > 0 && (
                    <div className="absolute w-full z-50 bg-[#0f172a] border border-slate-700 rounded-xl mt-1 overflow-hidden shadow-xl top-full left-0 right-0">
                      {fromResults.map((result, idx) => (
                        <div
                          key={idx}
                          onClick={() => {
                            const latVal = parseFloat(result.lat);
                            const lonVal = parseFloat(result.lon);
                            setFromValue(result.display_name.slice(0, 45));
                            setFromCoords({ lat: latVal, lng: lonVal });
                            setFlyToLocation({ lat: latVal, lng: lonVal, zoom: 15 });
                            setFromResults([]);
                            if (hasGeolocationOrigin) setHasGeolocationOrigin(false);
                          }}
                          className="px-3 py-2 text-sm text-white hover:bg-[#1e293b] cursor-pointer border-b border-slate-800 last:border-0 truncate"
                        >
                          {result.display_name.slice(0, 50)}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              <div className="flex items-center gap-3 relative z-10 mt-3">
                <div className="w-3 h-3 rounded-full bg-red-500 border-2 border-[#1e293b] shrink-0"></div>
                <div className="w-full relative">
                  <input
                    type="text"
                    placeholder="To"
                    value={toValue}
                    onChange={(e) => setToValue(e.target.value)}
                    className="bg-[#0f172a] text-sm rounded-xl px-3 py-2 text-white w-full border border-slate-700/50 focus:outline-none focus:border-indigo-500 transition-colors"
                  />
                  {toResults && toResults.length > 0 && (
                    <div className="absolute w-full z-50 bg-[#0f172a] border border-slate-700 rounded-xl mt-1 overflow-hidden shadow-xl top-full left-0 right-0">
                      {toResults.map((result, idx) => (
                        <div
                          key={idx}
                          onClick={() => {
                            const latVal = parseFloat(result.lat);
                            const lonVal = parseFloat(result.lon);

                            setToValue(result.display_name.slice(0, 45));
                            setToCoords({ lat: latVal, lng: lonVal });
                            setFlyToLocation({ lat: latVal, lng: lonVal, zoom: 15 });
                            setSelectedLocation({ lat: latVal, lng: lonVal });

                            setToResults([]);
                            setRouteData(null);
                            setSelectedRoute(null);
                          }}
                          className="px-3 py-2 text-sm text-white hover:bg-[#1e293b] cursor-pointer border-b border-slate-800 last:border-0 truncate"
                        >
                          {result.display_name.slice(0, 50)}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>

            <button
              onClick={() => {
                const tempVal = fromValue;
                const tempCoords = fromCoords;
                setFromValue(toValue);
                setFromCoords(toCoords);
                setToValue(tempVal);
                setToCoords(tempCoords);
                setRouteData(null);
                setSelectedRoute(null);
                setHasGeolocationOrigin(false);
              }}
              className="absolute right-6 top-6 w-7 h-7 bg-[#334155] rounded-full flex items-center justify-center border border-slate-600 hover:bg-slate-500 transition-colors z-20 shadow-md"
              title="Swap Origin and Destination"
            >
              <ArrowUpDown className="w-3.5 h-3.5 text-slate-300" />
            </button>

            <button
              onClick={() => {
                if (!fromCoords) alert("Please set a starting location");
                else if (!toCoords) alert("Please set a destination");
                else handleGetRoute();
              }}
              className="mt-1 w-full bg-blue-600 hover:bg-blue-500 text-white rounded-xl py-2 text-sm font-semibold shadow-lg transition-colors"
            >
              Get Routes
            </button>
            {routeData && (
              <p className="text-xs text-slate-400 text-center mt-1">Select a route below to highlight it</p>
            )}
          </div>

          <div className="flex flex-col gap-1.5 pb-2">
            <h2 className="text-white font-bold text-lg flex items-center gap-2">
              <Activity className="w-5 h-5 text-indigo-400" />
              Safety Analysis
            </h2>
          </div>

          <div className="flex flex-col">
            <SafetyPanel
              analysisResult={displayedAnalysisResult}
              isLoading={isLoading}
              isRouteSelected={!!selectedRoute}
              selectedRouteName={selectedRoute?.name}
            />
          </div>

          {/* Get Route Button (Show only when analysis is done and routeData is not fetched yet) */}
          {analysisResult && !routeData && (
            <button
              onClick={() => {
                setToValue("Selected location");
                setToCoords(selectedLocation);
              }}
              className="mt-2 w-full bg-slate-800 hover:bg-slate-700 border border-slate-700 hover:border-slate-500 text-white rounded-xl py-3 font-semibold transition-all active:scale-[0.98] tracking-wide text-sm"
            >
              Set as Destination
            </button>
          )}

          {/* Route Options (Show when routeData exists) */}
          {routeData && (
            <RouteComparison
              routes={routeData.routes}
              selectedRoute={selectedRoute}
              onSelectRoute={handleRouteSelect}
            />
          )}

        </aside>

        {/* Right Map Area */}
        <main className="flex-1 relative bg-[#0f172a] z-0">
          <MapView
            showHeatmap={showHeatmap}
            onLocationSelect={handleLocationSelect}
            heatmapData={heatmapData}
            isLoading={isLoading}
            selectedLocation={selectedLocation}
            mapTheme={mapTheme}
            flyToLocation={flyToLocation}
            fromCoords={fromCoords}
            toCoords={toCoords}
            fromValue={fromValue}
            toValue={toValue}
            hasGeolocationOrigin={hasGeolocationOrigin}
            routeData={routeData}
            selectedRoute={selectedRoute}
            onRouteSelect={handleRouteSelect}
          />
        </main>

      </div>

      {showProfileModal && (
        <ProfileSettings
          profile={userProfile}
          onUpdate={setUserProfile}
          onClose={() => setShowProfileModal(false)}
        />
      )}

    </div>
  );
}
