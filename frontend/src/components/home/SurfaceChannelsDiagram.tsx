"use client";

import React, { useState } from "react";

interface ChannelSpec {
  id: string;
  name: string;
  unit: string;
  source: string;
  role: string;
}

const CHANNELS: ChannelSpec[] = [
  { id: "SST", name: "Sea Surface Temperature", unit: "°C", source: "OSTIA MetOffice L4", role: "Direct thermal boundary condition for the oceanic mixed layer." },
  { id: "SSS", name: "Sea Surface Salinity", unit: "psu", source: "CMEMS Multi-platform L4", role: "Governs freshwater lens and salinity barrier layer stratification." },
  { id: "SLA", name: "Sea Level Anomaly", unit: "m", source: "DUACS Merged Altimetry", role: "Integrated indicator of steric expansion and thermocline depth heaving." },
  { id: "CUR_U", name: "Zonal Geostrophic Current", unit: "m/s", source: "Derived from SLA", role: "East-west advective heat transport across the equatorial boundary." },
  { id: "CUR_V", name: "Meridional Current", unit: "m/s", source: "Derived from SLA", role: "North-south coastal boundary currents and mesoscale eddy circulation." },
  { id: "WND_U", name: "Zonal Wind Stress", unit: "m/s", source: "ASCAT Scatterometer", role: "Drives surface wind-stress curl and mixed-layer deepening." },
  { id: "WND_V", name: "Meridional Wind Stress", unit: "m/s", source: "ASCAT Scatterometer", role: "Forces coastal upwelling along the western boundary." },
];

export default function SurfaceChannelsDiagram() {
  const [activeChannelId, setActiveChannelId] = useState<string>("SLA");
  const active = CHANNELS.find((c) => c.id === activeChannelId) || CHANNELS[2];

  return (
    <div className="w-full space-y-12">
      {/* 7-Channel Horizontal Flow (Open Typographic Composition — NO CARDS) */}
      <div className="w-full">
        <div className="flex flex-wrap items-baseline justify-between gap-4 pb-4 rule-b font-mono text-xs">
          <span className="text-neutral-400 uppercase tracking-wider">
            7 Spaceborne Observables [Shape: 7 × 69 × 81 at 0.25°]
          </span>
          <span className="text-neutral-500">
            Click observable to inspect vertical coupling
          </span>
        </div>

        {/* Clean Open Horizontal Table Strip */}
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 divide-y sm:divide-y-0 sm:divide-x divide-white/[0.08] rule-b">
          {CHANNELS.map((ch) => {
            const isSelected = ch.id === activeChannelId;
            return (
              <button
                key={ch.id}
                onClick={() => setActiveChannelId(ch.id)}
                className={`text-left p-6 transition-colors group cursor-pointer ${
                  isSelected ? "bg-white/[0.04]" : "hover:bg-white/[0.02]"
                }`}
              >
                <div className="flex items-center justify-between font-mono text-xs">
                  <span
                    className={`font-semibold tracking-wider ${
                      isSelected ? "text-cyan-400" : "text-neutral-100 group-hover:text-white"
                    }`}
                  >
                    {ch.id}
                  </span>
                  <span className="text-neutral-500 text-[11px]">{ch.unit}</span>
                </div>
                <div className="text-sm text-neutral-300 mt-2 font-sans font-normal leading-snug">
                  {ch.name}
                </div>
                <div className="font-mono text-[11px] text-neutral-500 mt-4 truncate">
                  {ch.source}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Selected Observable Dynamic Coupling Details */}
      <div className="py-6 px-8 bg-[#0a0a0a] rule-b rule-t flex flex-col md:flex-row items-start md:items-center justify-between gap-6 font-mono text-xs">
        <div className="max-w-2xl space-y-1">
          <div className="text-[11px] text-cyan-400 uppercase tracking-wider">
            Physical Coupling Mechanism: {active.name} ({active.id})
          </div>
          <p className="text-neutral-300 text-sm font-sans leading-relaxed">
            {active.role}
          </p>
        </div>
        <div className="flex items-center gap-8 text-[11px] text-neutral-400 shrink-0 border-t md:border-t-0 md:border-l border-white/[0.08] pt-4 md:pt-0 md:pl-8">
          <div>
            <div className="text-neutral-500 uppercase text-[9px]">Sensor Source</div>
            <div className="text-neutral-200 mt-0.5">{active.source}</div>
          </div>
          <div>
            <div className="text-neutral-500 uppercase text-[9px]">Grid Mesh</div>
            <div className="text-neutral-200 mt-0.5">0.25° Daily L4</div>
          </div>
        </div>
      </div>

      {/* Vertical Transformation Flow to Subsurface */}
      <div className="relative py-8 flex flex-col items-center justify-center text-center">
        <div className="w-px h-12 bg-white/20" />
        <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest my-3 px-4 py-1.5 border border-white/10 bg-[#0a0a0a]">
          Deep Spatial Embedding & Spectral Operator
        </div>
        <div className="w-px h-12 bg-white/20" />

        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 w-full mt-8 text-left">
          <div className="space-y-2 rule-t pt-6">
            <div className="font-mono text-xs text-neutral-500 uppercase">
              Latent Ocean Embedding [256, 18, 21]
            </div>
            <h4 className="text-lg font-medium text-white">Coupled Dynamical Manifold</h4>
            <p className="text-neutral-400 text-sm leading-relaxed font-sans">
              A multiscale CNN encoder extracts non-local spatial correlations, mapping 2D surface
              observables into a 256-channel latent manifold that captures baroclinic pressure
              gradients and wind-stress curl divergence.
            </p>
          </div>

          <div className="space-y-2 rule-t pt-6">
            <div className="font-mono text-xs text-neutral-500 uppercase">
              Fourier Neural Operator [15, 69, 81]
            </div>
            <h4 className="text-lg font-medium text-white">Continuous Subsurface Inversion</h4>
            <p className="text-neutral-400 text-sm leading-relaxed font-sans">
              Spectral convolution layers evaluate parameterised continuous integral kernels in Fourier
              frequency space, reconstructing the complete 3D temperature volume across 15 standard
              depth horizons down to 1000 meters.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
