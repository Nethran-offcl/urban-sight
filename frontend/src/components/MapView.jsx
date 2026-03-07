import React from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, Tooltip, useMapEvents, useMap, Polyline } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

function MapEventsHandler({ onClick }) {
    useMapEvents({
        click: onClick,
    });
    return null;
}

function MapPanner({ flyToLocation, routeData, fromCoords, toCoords }) {
    const map = useMap();

    // Fly to specific location (search or geolocation)
    React.useEffect(() => {
        if (flyToLocation) {
            map.flyTo([flyToLocation.lat, flyToLocation.lng], flyToLocation.zoom || 15);
        }
    }, [flyToLocation, map]);

    // Fit bounds to show entire route
    React.useEffect(() => {
        if (routeData && fromCoords && toCoords) {
            const bounds = [
                [fromCoords.lat, fromCoords.lng],
                [toCoords.lat, toCoords.lng]
            ];
            map.fitBounds(bounds, { padding: [60, 60] });
        }
    }, [routeData, fromCoords, toCoords, map]);

    return null;
}

export default function MapView({
    showHeatmap,
    onLocationSelect,
    heatmapData = [],
    isLoading,
    selectedLocation,
    mapTheme = 'dark',
    flyToLocation,
    fromCoords,
    toCoords,
    fromValue,
    toValue,
    hasGeolocationOrigin,
    routeData,
    selectedRoute,
    onRouteSelect
}) {
    const center = [12.9760, 77.5877]; // Center of Bengaluru

    const handleMapClick = (e) => {
        onLocationSelect(e.latlng.lat, e.latlng.lng);
    };

    const tileUrl = mapTheme === 'light'
        ? "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
        : "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png";

    return (
        <MapContainer
            center={center}
            zoom={13}
            className="w-full h-full"
            zoomControl={false}
        >
            <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
                url={tileUrl}
            />
            <MapEventsHandler onClick={handleMapClick} />
            <MapPanner flyToLocation={flyToLocation} routeData={routeData} fromCoords={fromCoords} toCoords={toCoords} />

            {/* Heatmap Layer */}
            {showHeatmap && heatmapData.map((pt, i) => (
                <CircleMarker
                    key={i}
                    center={[pt.lat, pt.lng]}
                    radius={40}
                    pathOptions={{
                        fillColor: pt.color_code,
                        fillOpacity: 0.35,
                        stroke: false,
                        className: "blur-[6px]"
                    }}
                >
                    <Popup className="bg-[#1e293b] border-[#1e293b] text-slate-800">
                        Safety Score: {(pt.safety_score * 10).toFixed(1)}/10
                    </Popup>
                </CircleMarker>
            ))}

            {/* Selected Location Marker */}
            {selectedLocation && !toCoords && (
                <CircleMarker
                    center={[selectedLocation.lat, selectedLocation.lng]}
                    radius={10}
                    pathOptions={{
                        fillColor: mapTheme === 'light' ? '#4f46e5' : '#ffffff',
                        fillOpacity: 1,
                        color: mapTheme === 'light' ? '#c7d2fe' : '#818cf8',
                        weight: 3
                    }}
                >
                    <Popup autoPan={false}>
                        {isLoading ? "Analyzing..." : "Selected"}
                    </Popup>
                </CircleMarker>
            )}

            {/* Draw Routes */}
            {routeData && routeData.routes.map((route, i) => {
                const isSelected = selectedRoute && selectedRoute.name === route.name;
                const isOther = selectedRoute && selectedRoute.name !== route.name;
                const midIndex = Math.floor(route.waypoints.length / 2);
                const midpoint = route.waypoints[midIndex];

                return (
                    <React.Fragment key={`route-${i}`}>
                        <Polyline
                            positions={route.waypoints}
                            pathOptions={{
                                color: route.color_code,
                                weight: isSelected ? 6 : isOther ? 3 : 4,
                                opacity: isSelected ? 1 : isOther ? 0.2 : 0.35,
                                dashArray: isSelected ? null : "8 6",
                                lineCap: "round",
                                lineJoin: "round"
                            }}
                            eventHandlers={{
                                click: () => {
                                    if (onRouteSelect) onRouteSelect(route);
                                }
                            }}
                            className="transition-all duration-300"
                        />
                        {/* Selected Route Info Tooltip */}
                        {isSelected && midpoint && (
                            <CircleMarker center={[midpoint[0], midpoint[1]]} radius={0} opacity={0} fillOpacity={0}>
                                <Tooltip permanent direction="top" offset={[0, -5]} className="bg-[#1e293b] text-white text-xs rounded-lg px-2 py-1 border-0 shadow-xl font-medium whitespace-nowrap">
                                    {route.name} · {route.estimated_minutes} min · Safety {Math.round(route.avg_safety_score * 100)}%
                                </Tooltip>
                            </CircleMarker>
                        )}
                    </React.Fragment>
                );
            })}

            {/* Origin Marker */}
            {fromCoords && (
                <CircleMarker
                    center={[fromCoords.lat, fromCoords.lng]}
                    radius={10}
                    pathOptions={{
                        fillColor: '#3b82f6',
                        fillOpacity: 1,
                        color: '#ffffff',
                        weight: 2,
                        stroke: !hasGeolocationOrigin
                    }}
                    className={hasGeolocationOrigin ? "animate-pulse" : ""}
                >
                    <Tooltip permanent direction="top" offset={[0, -10]} className="bg-[#1e293b] text-white border-0 shadow-lg font-semibold">
                        Start: {fromValue ? fromValue.substring(0, 25) : "Origin"}
                    </Tooltip>
                </CircleMarker>
            )}

            {/* Destination Marker */}
            {toCoords && (
                <CircleMarker
                    center={[toCoords.lat, toCoords.lng]}
                    radius={10}
                    pathOptions={{
                        fillColor: '#ef4444',
                        fillOpacity: 1,
                        color: '#ffffff',
                        weight: 2,
                    }}
                >
                    <Tooltip permanent direction="bottom" offset={[0, 10]} className="bg-[#1e293b] text-white border-0 shadow-lg font-semibold">
                        End: {toValue ? toValue.substring(0, 25) : "Destination"}
                    </Tooltip>
                </CircleMarker>
            )}
        </MapContainer>
    );
}
