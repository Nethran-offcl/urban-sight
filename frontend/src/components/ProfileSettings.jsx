import React, { useState } from 'react';
import { X, Footprints, Bike, Car, User, Users } from 'lucide-react';

export default function ProfileSettings({ profile, onUpdate, onClose }) {
    const [localState, setLocalState] = useState({ ...profile });

    const handleChange = (key, value) => {
        setLocalState(prev => ({ ...prev, [key]: value }));
    };

    const handleSave = () => {
        onUpdate(localState);
        onClose();
    };

    return (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 animate-in fade-in duration-200">

            <div className="bg-[#1e293b] rounded-2xl p-6 w-96 max-w-[90vw] shadow-2xl border border-slate-700/50 animate-in zoom-in-95 duration-200 relative">

                {/* Header */}
                <div className="flex items-center justify-between mb-6">
                    <h2 className="text-white font-bold text-lg">Traveller Profile</h2>
                    <button
                        onClick={onClose}
                        className="text-slate-400 hover:text-white transition-colors p-1 rounded-full hover:bg-slate-700 absolute top-4 right-4"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Travel Mode */}
                <div className="mt-4">
                    <label className="block text-xs text-slate-400 mb-2 uppercase font-semibold tracking-wider">Travel Mode</label>
                    <div className="grid grid-cols-3 gap-2">

                        <button
                            type="button"
                            onClick={() => handleChange('mode', 'walking')}
                            className={`flex flex-col items-center p-3 rounded-xl text-xs gap-1.5 font-medium transition-all ${localState.mode === 'walking'
                                ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/20'
                                : 'bg-[#0f172a] text-slate-400 hover:bg-slate-800'
                                }`}
                        >
                            <Footprints className="w-5 h-5" />
                            Walking
                        </button>

                        <button
                            type="button"
                            onClick={() => handleChange('mode', 'cycling')}
                            className={`flex flex-col items-center p-3 rounded-xl text-xs gap-1.5 font-medium transition-all ${localState.mode === 'cycling'
                                ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/20'
                                : 'bg-[#0f172a] text-slate-400 hover:bg-slate-800'
                                }`}
                        >
                            <Bike className="w-5 h-5" />
                            Cycling
                        </button>

                        <button
                            type="button"
                            onClick={() => handleChange('mode', 'driving')}
                            className={`flex flex-col items-center p-3 rounded-xl text-xs gap-1.5 font-medium transition-all ${localState.mode === 'driving'
                                ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/20'
                                : 'bg-[#0f172a] text-slate-400 hover:bg-slate-800'
                                }`}
                        >
                            <Car className="w-5 h-5" />
                            Driving
                        </button>

                    </div>
                </div>

                {/* Gender / Group Selection */}
                <div className="mt-6">
                    <label className="block text-xs text-slate-400 mb-2 uppercase font-semibold tracking-wider">Party Type</label>
                    <div className="grid grid-cols-3 gap-2">

                        <button
                            type="button"
                            onClick={() => {
                                handleChange('gender', 'female');
                                handleChange('group_size', 1);
                                handleChange('gender_sensitive', true);
                            }}
                            className={`flex flex-col items-center p-3 rounded-xl text-xs gap-1.5 font-medium transition-all ${localState.gender === 'female'
                                ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/20'
                                : 'bg-[#0f172a] text-slate-400 hover:bg-slate-800'
                                }`}
                        >
                            <User className="w-5 h-5" />
                            Female Solo
                        </button>

                        <button
                            type="button"
                            onClick={() => {
                                handleChange('gender', 'male');
                                handleChange('group_size', 1);
                                handleChange('gender_sensitive', true);
                            }}
                            className={`flex flex-col items-center p-3 rounded-xl text-xs gap-1.5 font-medium transition-all ${localState.gender === 'male'
                                ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/20'
                                : 'bg-[#0f172a] text-slate-400 hover:bg-slate-800'
                                }`}
                        >
                            <User className="w-5 h-5" />
                            Male Solo
                        </button>

                        <button
                            type="button"
                            onClick={() => {
                                handleChange('gender', 'group');
                                handleChange('gender_sensitive', false);
                                if (localState.group_size === 1) handleChange('group_size', 2);
                            }}
                            className={`flex flex-col items-center p-3 rounded-xl text-xs gap-1.5 font-medium transition-all ${localState.gender === 'group'
                                ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/20'
                                : 'bg-[#0f172a] text-slate-400 hover:bg-slate-800'
                                }`}
                        >
                            <Users className="w-5 h-5" />
                            Group
                        </button>

                    </div>
                </div>

                {/* Group Size Range (Only show if Group is selected) */}
                <div className={`mt-6 transition-all duration-300 overflow-hidden ${localState.gender === 'group' ? 'max-h-24 opacity-100' : 'max-h-0 opacity-0'}`}>
                    <label className="block text-xs text-slate-400 mb-2 uppercase font-semibold tracking-wider flex justify-between">
                        <span>Group Size</span>
                        <span className="text-white pr-1">{localState.group_size}</span>
                    </label>
                    <input
                        type="range"
                        min="2"
                        max="8"
                        step="1"
                        value={localState.group_size}
                        onChange={(e) => handleChange('group_size', parseInt(e.target.value))}
                        className="w-full accent-blue-500 mt-1 cursor-pointer"
                    />
                    <div className="flex justify-between text-[10px] text-slate-500 mt-1 px-1">
                        <span>2 People</span>
                        <span>8 People</span>
                    </div>
                </div>

                {/* Toggles */}
                <div className="mt-8 flex flex-col gap-5">

                    {/* Night Mode Toggle */}
                    <div className="flex justify-between items-center">
                        <span className="text-sm text-white font-medium">Night Mode Awareness</span>
                        <button
                            type="button"
                            onClick={() => handleChange('is_night', !localState.is_night)}
                            className={`relative w-12 h-6 rounded-full cursor-pointer transition-colors duration-300 focus:outline-none shrink-0 ${localState.is_night ? 'bg-blue-600' : 'bg-slate-600'}`}
                        >
                            <div className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full transition-transform duration-300 shadow-sm ${localState.is_night ? 'translate-x-6' : 'translate-x-0'}`} />
                        </button>
                    </div>

                    {/* Gender Sensitive Toggle */}
                    <div className="flex justify-between items-start gap-4">
                        <div className="flex flex-col gap-1 pr-2">
                            <span className="text-sm text-white font-medium">Gender-Sensitive Routing</span>
                            <span className="text-xs text-slate-400 leading-tight">Prioritizes well-lit, heavily populated active areas</span>
                        </div>
                        <button
                            type="button"
                            onClick={() => handleChange('gender_sensitive', !localState.gender_sensitive)}
                            className={`relative w-12 h-6 rounded-full cursor-pointer transition-colors duration-300 focus:outline-none shrink-0 mt-0.5 ${localState.gender_sensitive ? 'bg-blue-600' : 'bg-slate-600'}`}
                        >
                            <div className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full transition-transform duration-300 shadow-sm ${localState.gender_sensitive ? 'translate-x-6' : 'translate-x-0'}`} />
                        </button>
                    </div>

                </div>

                {/* Done Button */}
                <button
                    type="button"
                    onClick={handleSave}
                    className="mt-8 w-full bg-slate-700 text-white rounded-xl py-2.5 font-semibold hover:bg-slate-600 transition-colors shadow-lg shadow-slate-900/20 active:scale-[0.98]"
                >
                    Done
                </button>

            </div>
        </div>
    );
}
