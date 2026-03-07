import React from 'react';
import { Map as MapIcon, Shield, ArrowRight, AlertTriangle } from 'lucide-react';

export default function SafetyPanel({ analysisResult, isLoading, isRouteSelected, selectedRouteName }) {

    if (isLoading) {
        return (
            <div className="flex flex-col gap-3 py-2">
                <div className="h-16 bg-[#334155] rounded-xl animate-pulse"></div>
                <div className="h-16 bg-[#334155] rounded-xl animate-pulse"></div>
                <div className="h-16 bg-[#334155] rounded-xl animate-pulse"></div>
            </div>
        );
    }

    if (!analysisResult) {
        return (
            <div className="flex-1 flex flex-col items-center justify-center text-center gap-4 py-12">
                <div className="w-16 h-16 rounded-full bg-slate-800/50 flex items-center justify-center border border-slate-700/50">
                    <MapIcon className="w-8 h-8 text-slate-500" />
                </div>
                <p className="text-sm text-slate-400 max-w-[200px] leading-relaxed">
                    Click anywhere on the map to analyze safety metrics and risk factors.
                </p>
            </div>
        );
    }

    // Fix risk label logic
    const score = analysisResult.safety_pct;
    let displayCategory = analysisResult.category;
    let displayColor = analysisResult.color_code;

    if (score >= 70) {
        displayCategory = "High Safety";
        displayColor = "#22c55e";  // Green
    } else if (score >= 40) {
        displayCategory = "Moderate Risk";
        displayColor = "#f97316";  // Orange
    } else {
        displayCategory = "High Risk";
        displayColor = "#ef4444";  // Red
    }

    // Calculate SVG arc parameters. High Safety (100%) means Low Risk (0% circle fill).
    // Low Safety (0%) means High Risk (100% circle fill).
    const radius = 50;
    const circumference = 2 * Math.PI * radius;
    const risk_pct = 100 - score;
    const strokeDashoffset = circumference - (risk_pct / 100) * circumference;

    return (
        <div className="flex flex-col animate-in fade-in slide-in-from-bottom-2 duration-300">

            {/* Safety Meter SVG Circular Chart */}
            <div className="flex flex-col items-center justify-center py-4 bg-[#0f172a]/50 rounded-xl border border-slate-700/50">
                <span className="text-xs text-slate-400 mb-3 font-medium">
                    {isRouteSelected && selectedRouteName
                        ? `Route Safety: ${selectedRouteName} Path`
                        : "Destination Safety Score"}
                </span>

                <div className="relative w-[120px] h-[120px]">
                    <svg className="w-full h-full transform -rotate-90">
                        {/* Background Circle */}
                        <circle
                            cx="60"
                            cy="60"
                            r={radius}
                            fill="transparent"
                            stroke="#1e293b"
                            strokeWidth="10"
                        />
                        {/* Progress Arc */}
                        <circle
                            cx="60"
                            cy="60"
                            r={radius}
                            fill="transparent"
                            stroke={displayColor}
                            strokeWidth="10"
                            strokeDasharray={circumference}
                            strokeDashoffset={strokeDashoffset}
                            strokeLinecap="round"
                            className="transition-all duration-1000 ease-out"
                        />
                    </svg>
                    {/* Centered Large Number */}
                    <div className="absolute inset-0 flex items-center justify-center">
                        <span className="text-2xl font-bold text-white tracking-tighter">
                            {risk_pct}%
                        </span>
                    </div>
                </div>

                {/* Risk Label */}
                <span
                    className="mt-3 font-semibold text-sm tracking-wide"
                    style={{ color: displayColor }}
                >
                    {displayCategory}
                </span>
            </div>

            {/* AI Insights Card */}
            <div className="bg-[#0f172a] rounded-xl p-4 mt-4 border border-slate-700/50">
                <div className="flex items-center gap-1.5">
                    <Shield className="w-4 h-4 text-blue-400" />
                    <span className="text-xs text-slate-400 font-medium uppercase tracking-wider">AI Insight</span>
                </div>
                <p className="text-sm text-white mt-2 leading-relaxed">
                    {(!analysisResult.explanation || analysisResult.explanation.includes("Model not loaded")) ? "Analysis complete. See safety score above." : analysisResult.explanation}
                </p>

                {/* Feature Chips */}
                {analysisResult.top_features && analysisResult.top_features.length > 0 && (
                    <div className="mt-3 flex flex-wrap gap-2">
                        {analysisResult.top_features.map((feature, idx) => (
                            <span key={idx} className="rounded-full px-2.5 py-1 bg-[#1e293b] text-xs text-blue-400 font-medium">
                                {feature}
                            </span>
                        ))}
                    </div>
                )}
            </div>

            {/* Profile Adjustments */}
            {analysisResult.adjustments_applied && analysisResult.adjustments_applied.length > 0 && (
                <div className="mt-4 flex flex-col">
                    <h3 className="text-xs text-slate-400 mb-2 uppercase tracking-wider font-semibold">Profile Adjustments</h3>
                    <div className="flex flex-col">
                        {analysisResult.adjustments_applied.map((adj, idx) => (
                            <div key={idx} className="rounded px-2.5 py-1.5 bg-[#1e293b] text-xs text-slate-300 mb-1.5 flex items-center gap-2">
                                <ArrowRight className="w-3 h-3 text-slate-500" />
                                {adj}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Recommendations */}
            {analysisResult.recommendations && analysisResult.recommendations.length > 0 && (
                <div className="mt-4 flex flex-col">
                    <h3 className="text-xs text-slate-400 mb-2 uppercase tracking-wider font-semibold">Recommendations</h3>
                    <div className="flex flex-col">
                        {analysisResult.recommendations.map((rec, idx) => (
                            <div key={idx} className="flex items-start gap-2 mb-2">
                                <AlertTriangle className="text-yellow-400 w-4 h-4 mt-0.5 shrink-0" />
                                <span className="text-sm text-slate-300 leading-snug">{rec}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

        </div>
    );
}
