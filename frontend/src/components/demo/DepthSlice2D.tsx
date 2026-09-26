"use client";

import React, { useMemo } from "react";
import PlotlyChart from "@/components/common/PlotlyChart";

interface DepthSlice2DProps {
  prediction: (number | null)[][][];      // [15, 69, 81]
  groundTruth?: (number | null)[][][];    // [15, 69, 81]
  error?: (number | null)[][][];          // [15, 69, 81]
  depthIdx: number;                       // 0 to 14
  depth_m: number;
  lats: number[];
  lons: number[];
  model_name: string;
  onProbePoint?: (lat: number, lon: number, val: number) => void;
}

export default function DepthSlice2D({
  prediction,
  groundTruth,
  error,
  depthIdx,
  depth_m,
  lats,
  lons,
  model_name,
  onProbePoint,
}: DepthSlice2DProps) {
  // Extract 2D matrix slices for the active depth level
  const predSlice = prediction?.[depthIdx] || [];
  const gtSlice = groundTruth?.[depthIdx] || predSlice;
  const errSlice = error?.[depthIdx] || [];

  // Compute thermal scale bounds based on physical ocean data at this depth
  const { zMin, zMax, predMean, gtMean, mae } = useMemo(() => {
    let pSum = 0, pCount = 0;
    let gSum = 0, gCount = 0;
    let eSum = 0, eCount = 0;
    let minT = 999, maxT = -999;

    for (let i = 0; i < predSlice.length; i++) {
      for (let j = 0; j < (predSlice[i]?.length || 0); j++) {
        const p = predSlice[i][j];
        const g = gtSlice[i]?.[j];
        const e = errSlice[i]?.[j];

        if (p !== null && !isNaN(p)) {
          pSum += p;
          pCount++;
          if (p < minT) minT = p;
          if (p > maxT) maxT = p;
        }
        if (g !== null && !isNaN(g)) {
          gSum += g;
          gCount++;
          if (g < minT) minT = g;
          if (g > maxT) maxT = g;
        }
        if (e !== null && !isNaN(e)) {
          eSum += Math.abs(e);
          eCount++;
        }
      }
    }

    // Default physical fallback bounds if sparse
    const fallbackMin = depth_m >= 500 ? 5 : depth_m >= 150 ? 11 : 20;
    const fallbackMax = depth_m >= 500 ? 15 : depth_m >= 150 ? 25 : 31.5;

    return {
      zMin: minT !== 999 ? Math.floor(minT * 10) / 10 : fallbackMin,
      zMax: maxT !== -999 ? Math.ceil(maxT * 10) / 10 : fallbackMax,
      predMean: pCount > 0 ? (pSum / pCount).toFixed(2) : "—",
      gtMean: gCount > 0 ? (gSum / gCount).toFixed(2) : "—",
      mae: eCount > 0 ? (eSum / eCount).toFixed(2) : "—",
    };
  }, [predSlice, gtSlice, errSlice, depth_m]);

  // Scientific Continuous Temperature Colormap (cmocean thermal)
  const tempColorscale = [
    [0, "#030712"],      // Coldest / deep
    [0.14, "#1e3a8a"],
    [0.28, "#0284c7"],
    [0.44, "#06b6d4"],   // Cyan transition
    [0.60, "#10b981"],   // Intermediate
    [0.74, "#f59e0b"],   // Amber warm
    [0.88, "#f97316"],   // Coral warm
    [1.0, "#ef4444"],    // Highest SST
  ];

  // Diverging RdBu / Cool-Warm Scale for Residual Error (centered at 0)
  const residualColorscale = [
    [0, "#2563eb"],     // -2.5 °C (Underestimation - Blue)
    [0.32, "#93c5fd"],  // -0.9 °C
    [0.5, "#18181b"],   //  0.0 °C (Zero Error - Neutral Dark)
    [0.68, "#fca5a5"],  // +0.9 °C
    [1.0, "#dc2626"],   // +2.5 °C (Overestimation - Red)
  ];

  // Shared synchronized layout options preserving geographic aspect ratio (1 deg lon = 1 deg lat)
  const baseLayout = {
    title: undefined,
    xaxis: {
      title: { text: "Lon [°E]", font: { family: "IBM Plex Sans, sans-serif", size: 11, color: "#a1a1aa" }, standoff: 8 },
      range: [80, 100],
      tickfont: { family: "IBM Plex Mono, monospace", size: 9, color: "#71717a" },
      gridcolor: "rgba(255, 255, 255, 0.05)",
      linecolor: "rgba(255, 255, 255, 0.12)",
      constrain: "domain" as const,
    },
    yaxis: {
      title: { text: "Lat [°N]", font: { family: "IBM Plex Sans, sans-serif", size: 11, color: "#a1a1aa" }, standoff: 8 },
      range: [5, 22],
      tickfont: { family: "IBM Plex Mono, monospace", size: 9, color: "#71717a" },
      gridcolor: "rgba(255, 255, 255, 0.05)",
      linecolor: "rgba(255, 255, 255, 0.12)",
      scaleanchor: "x",
      scaleratio: 1,
      constrain: "domain" as const,
    },
    margin: { l: 52, r: 24, t: 16, b: 48 },
    height: 380,
  };

  const handleSliceClick = (e: any) => {
    if (e.points && e.points[0] && onProbePoint) {
      const pt = e.points[0];
      onProbePoint(pt.y, pt.x, pt.z);
    }
  };

  // Traces for all three synchronized panels
  const predTrace = {
    z: predSlice,
    x: lons,
    y: lats,
    type: "heatmap",
    colorscale: tempColorscale,
    zmin: zMin,
    zmax: zMax,
    colorbar: {
      title: { text: "T [°C]", font: { family: "IBM Plex Mono, monospace", size: 10, color: "#a1a1aa" } },
      tickfont: { family: "IBM Plex Mono, monospace", size: 9, color: "#71717a" },
      thickness: 10,
      len: 0.88,
      y: 0.5,
    },
    hoverongaps: false,
    hovertemplate: "Lat: %{y:.2f}°N<br>Lon: %{x:.2f}°E<br>Pred: %{z:.2f} °C<extra></extra>",
  };

  const gtTrace = {
    z: gtSlice,
    x: lons,
    y: lats,
    type: "heatmap",
    colorscale: tempColorscale,
    zmin: zMin,
    zmax: zMax,
    colorbar: {
      title: { text: "T [°C]", font: { family: "IBM Plex Mono, monospace", size: 10, color: "#a1a1aa" } },
      tickfont: { family: "IBM Plex Mono, monospace", size: 9, color: "#71717a" },
      thickness: 10,
      len: 0.88,
      y: 0.5,
    },
    hoverongaps: false,
    hovertemplate: "Lat: %{y:.2f}°N<br>Lon: %{x:.2f}°E<br>GLORYS: %{z:.2f} °C<extra></extra>",
  };

  const errTrace = {
    z: errSlice,
    x: lons,
    y: lats,
    type: "heatmap",
    colorscale: residualColorscale,
    zmin: -2.5,
    zmax: 2.5,
    colorbar: {
      title: { text: "ΔT [°C]", font: { family: "IBM Plex Mono, monospace", size: 10, color: "#a1a1aa" } },
      tickfont: { family: "IBM Plex Mono, monospace", size: 9, color: "#71717a" },
      thickness: 10,
      len: 0.88,
      y: 0.5,
    },
    hoverongaps: false,
    hovertemplate: "Lat: %{y:.2f}°N<br>Lon: %{x:.2f}°E<br>Residual: %{z:+.2f} °C<extra></extra>",
  };

  return (
    <div className="w-full space-y-6 bg-[#050505] font-mono text-xs">
      {/* ── Synchronized Telemetry Strip ─────────────────────────────── */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-3 rule-b">
        <div className="flex items-center gap-3">
          <span className="text-neutral-500 uppercase tracking-wider text-[11px]">
            SYNCHRONIZED HORIZON:
          </span>
          <span className="text-white font-bold text-sm">{depth_m} m Depth</span>
          <span className="text-neutral-600">|</span>
          <span className="text-neutral-400 text-xs">69 × 81 Grid (5°–22°N, 80°–100°E)</span>
        </div>

        <div className="flex items-center gap-6 text-[11px] text-neutral-400">
          <div>
            <span className="text-neutral-500">Pred Mean: </span>
            <span className="text-cyan-400 font-semibold">{predMean} °C</span>
          </div>
          <div>
            <span className="text-neutral-500">GLORYS Mean: </span>
            <span className="text-amber-400 font-semibold">{gtMean} °C</span>
          </div>
          <div>
            <span className="text-neutral-500">Mean Abs Error: </span>
            <span className="text-neutral-200 font-semibold">{mae} °C</span>
          </div>
        </div>
      </div>

      {/* ── 3 Synchronized Panels Displayed Simultaneously (Geographic Aspect Ratio Preserved) ─ */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 w-full">
        {/* Panel 1: Prediction */}
        <div className="space-y-2 border border-white/[0.08] p-3 rounded-none bg-[#080808]">
          <div className="flex items-center justify-between text-xs pb-2 rule-b">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400" />
              <span className="font-semibold text-white font-sans text-xs">
                {model_name}
              </span>
            </div>
            <span className="text-[10px] text-cyan-400 uppercase font-mono">Prediction</span>
          </div>
          <div className="w-full h-[380px]">
            <PlotlyChart data={[predTrace]} layout={baseLayout} onClick={handleSliceClick} style={{ height: 380, minHeight: 380 }} />
          </div>
        </div>

        {/* Panel 2: GLORYS12 Ground Truth */}
        <div className="space-y-2 border border-white/[0.08] p-3 rounded-none bg-[#080808]">
          <div className="flex items-center justify-between text-xs pb-2 rule-b">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-amber-400" />
              <span className="font-semibold text-white font-sans text-xs">
                GLORYS12 Reanalysis
              </span>
            </div>
            <span className="text-[10px] text-amber-400 uppercase font-mono">Ground Truth</span>
          </div>
          <div className="w-full h-[380px]">
            <PlotlyChart data={[gtTrace]} layout={baseLayout} onClick={handleSliceClick} style={{ height: 380, minHeight: 380 }} />
          </div>
        </div>

        {/* Panel 3: Residual Divergence (ΔT) */}
        <div className="space-y-2 border border-white/[0.08] p-3 rounded-none bg-[#080808] md:col-span-2 xl:col-span-1">
          <div className="flex items-center justify-between text-xs pb-2 rule-b">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-blue-500" />
              <span className="font-semibold text-white font-sans text-xs">
                Residual Error (Pred − Ref)
              </span>
            </div>
            <span className="text-[10px] text-neutral-400 uppercase font-mono">Divergence</span>
          </div>
          <div className="w-full h-[380px]">
            <PlotlyChart data={[errTrace]} layout={baseLayout} onClick={handleSliceClick} style={{ height: 380, minHeight: 380 }} />
          </div>
        </div>
      </div>
    </div>
  );
}
