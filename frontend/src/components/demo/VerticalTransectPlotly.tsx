"use client";

import React, { useState } from "react";
import PlotlyChart from "@/components/common/PlotlyChart";

interface VerticalTransectProps {
  prediction: (number | null)[][][]; // [15, 69, 81]
  depths: number[];                  // 15 depths
  lats: number[];                    // 69 lats
  lons: number[];                    // 81 lons
  model_name: string;
}

export default function VerticalTransectPlotly({
  prediction,
  depths,
  lats,
  lons,
  model_name,
}: VerticalTransectProps) {
  const [transectType, setTransectType] = useState<"meridional" | "zonal">("meridional");
  const [lonIndex, setLonIndex] = useState<number>(36); // ~89°E central basin
  const [latIndex, setLatIndex] = useState<number>(34); // ~13.5°N central basin

  const zMatrix: (number | null)[][] = [];
  const xValues = transectType === "meridional" ? lats : lons;

  for (let d = 0; d < depths.length; d++) {
    const row: (number | null)[] = [];
    if (transectType === "meridional") {
      for (let i = 0; i < lats.length; i++) {
        row.push(prediction[d]?.[i]?.[lonIndex] ?? null);
      }
    } else {
      for (let j = 0; j < lons.length; j++) {
        row.push(prediction[d]?.[latIndex]?.[j] ?? null);
      }
    }
    zMatrix.push(row);
  }

  const trace = {
    z: zMatrix,
    x: xValues,
    y: depths,
    type: "contour",
    colorscale: [
      [0, "#030712"],
      [0.12, "#1e3a8a"],
      [0.28, "#0284c7"],
      [0.48, "#06b6d4"],
      [0.65, "#059669"],
      [0.82, "#d97706"],
      [1, "#dc2626"],
    ],
    zmin: 4,
    zmax: 30,
    contours: {
      coloring: "heatmap",
      showlines: false,
    },
    line: { smoothing: 1.3 },
    colorbar: {
      title: {
        text: "Temp [°C]",
        font: { family: "IBM Plex Mono, monospace", size: 11, color: "#d4d4d8" },
      },
      tickfont: { family: "IBM Plex Mono, monospace", size: 10, color: "#a1a1aa" },
      thickness: 14,
      len: 0.9,
    },
    hoverongaps: false,
    hovertemplate: "%{x:.2f}° / Depth: %{y}m<br>Temp: %{z:.2f} °C<extra></extra>",
  };

  const layout = {
    title: undefined, // Controlled by HTML header outside!
    xaxis: {
      title: { text: transectType === "meridional" ? "Latitude [°N]" : "Longitude [°E]" },
      gridcolor: "rgba(255,255,255,0.05)",
      linecolor: "rgba(255,255,255,0.12)",
      tickfont: { family: "IBM Plex Mono, monospace", size: 10 },
    },
    yaxis: {
      title: { text: "Depth [m]" },
      autorange: "reversed",
      gridcolor: "rgba(255,255,255,0.05)",
      linecolor: "rgba(255,255,255,0.12)",
      tickfont: { family: "IBM Plex Mono, monospace", size: 10 },
      tickvals: [0, 30, 75, 150, 300, 500, 700, 1000],
    },
    margin: { l: 80, r: 40, t: 20, b: 64 },
    height: 600,
  };

  return (
    <div className="w-full bg-[#050505] font-mono text-xs space-y-4">
      {/* Transect Controls */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-3 rule-b">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setTransectType("meridional")}
            className={`px-3 py-1 text-xs transition-colors cursor-pointer ${
              transectType === "meridional"
                ? "bg-white/10 text-white font-medium"
                : "text-neutral-400 hover:text-white"
            }`}
          >
            Meridional Transect (N–S)
          </button>
          <button
            onClick={() => setTransectType("zonal")}
            className={`px-3 py-1 text-xs transition-colors cursor-pointer ${
              transectType === "zonal"
                ? "bg-white/10 text-white font-medium"
                : "text-neutral-400 hover:text-white"
            }`}
          >
            Zonal Transect (W–E)
          </button>
        </div>

        <div className="flex items-center gap-3">
          {transectType === "meridional" ? (
            <>
              <span className="text-neutral-500 text-[11px]">Longitude Slice:</span>
              <input
                type="range"
                min={0}
                max={lons.length - 1}
                value={lonIndex}
                onChange={(e) => setLonIndex(Number(e.target.value))}
                className="w-32 accent-white"
              />
              <span className="text-neutral-200 text-xs w-16 text-right">
                {lons[lonIndex]?.toFixed(1)}°E
              </span>
            </>
          ) : (
            <>
              <span className="text-neutral-500 text-[11px]">Latitude Slice:</span>
              <input
                type="range"
                min={0}
                max={lats.length - 1}
                value={latIndex}
                onChange={(e) => setLatIndex(Number(e.target.value))}
                className="w-32 accent-white"
              />
              <span className="text-neutral-200 text-xs w-16 text-right">
                {lats[latIndex]?.toFixed(1)}°N
              </span>
            </>
          )}
        </div>
      </div>

      <div className="w-full h-[600px]">
        <PlotlyChart data={[trace]} layout={layout} style={{ minHeight: 600 }} />
      </div>
    </div>
  );
}
