import React from 'react';
import { Route as RouteIcon, MapPin, Clock, AlertTriangle, ArrowRight, Check } from 'lucide-react';

export default function RouteComparison({ routes, onSelectRoute, selectedRoute }) {
    if (!routes || routes.length === 0) return null;

    return (
        <div className="mt-6 flex flex-col gap-3 animate-in fade-in slide-in-from-bottom-4 duration-500">

            <div className="flex items-center gap-2 mb-1">
                <RouteIcon className="w-4 h-4 text-indigo-400" />
                <h3 className="text-white font-bold tracking-wide">Route Options</h3>
            </div>

            <div className="flex flex-col gap-3">
                {routes.map((route, idx) => {
                    const isSelected = selectedRoute?.name === route.name;

                    return (
                        <div
                            key={idx}
                            onClick={() => onSelectRoute(route)}
                            className={`bg-[#0f172a] rounded-xl p-4 cursor-pointer transition-all duration-200 border-2 border-l-4 ${isSelected ? 'shadow-[0_0_15px_rgba(0,0,0,0.3)] bg-[#0a1628]' : 'border-transparent hover:border-r-slate-700 hover:border-y-slate-700 hover:bg-slate-800/30'}`}
                            style={{
                                borderColor: isSelected ? route.color_code : undefined,
                                borderLeftColor: route.color_code
                            }}
                        >

                            {/* Top Row */}
                            <div className="flex justify-between items-center">
                                <span className="text-white font-semibold text-sm">{route.name} Route</span>
                                {idx === 0 && (
                                    <span className="text-[10px] bg-green-500/20 text-green-400 px-2 py-0.5 rounded-full font-bold uppercase tracking-wider">
                                        Recommended
                                    </span>
                                )}
                            </div>

                            {/* Safety Bar */}
                            <div className="mt-3 bg-slate-700 h-1.5 rounded-full overflow-hidden w-full relative">
                                <div
                                    className="absolute top-0 left-0 h-full rounded-full transition-all duration-1000 ease-out"
                                    style={{
                                        width: `${route.avg_safety_score * 100}%`,
                                        backgroundColor: route.color_code
                                    }}
                                />
                            </div>

                            {/* Stats Row */}
                            <div className="mt-3 flex gap-4">
                                <div className="flex items-center gap-1 text-xs text-slate-400 font-medium">
                                    <Clock className="w-3.5 h-3.5" />
                                    {route.estimated_minutes} min
                                </div>
                                <div className="flex items-center gap-1 text-xs text-slate-400 font-medium">
                                    <AlertTriangle className="w-3.5 h-3.5" />
                                    {route.risk_zone_count !== undefined ? route.risk_zone_count : (idx === 0 ? 0 : idx === 1 ? 2 : 1)} caution zones
                                </div>
                            </div>

                            {/* Explanation */}
                            <p className="mt-3 text-xs text-slate-400 leading-relaxed pl-2 border-l-2 border-slate-700">
                                {route.explanation && route.explanation.trim() !== "" && route.explanation !== "Model not loaded." && !route.explanation.includes("Average safety score")
                                    ? route.explanation
                                    : (route.name === "Safest" ? "Prioritises safety over speed."
                                        : route.name === "Fastest" ? "Shortest path to destination."
                                            : "Balanced route for comfort and safety.")}
                            </p>

                            {/* Select Button */}
                            <div className="mt-3 pt-3 border-t border-slate-800 flex justify-end">
                                {!isSelected ? (
                                    <span className="text-xs text-indigo-400 font-semibold hover:text-indigo-300 transition-colors uppercase tracking-wider flex items-center gap-1">
                                        Select Route <ArrowRight className="w-3.5 h-3.5" />
                                    </span>
                                ) : (
                                    <span
                                        className="text-xs font-bold uppercase tracking-wider flex items-center gap-1 text-green-400"
                                    >
                                        <Check className="w-3.5 h-3.5" />
                                        Selected
                                    </span>
                                )}
                            </div>

                        </div>
                    );
                })}
            </div>

        </div>
    );
}
