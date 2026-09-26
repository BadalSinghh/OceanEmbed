"use client";

import { useState } from "react";
import HeatmapCanvas from "./HeatmapCanvas";

interface TransectChartProps {
  prediction: (number | null)[][][];
  lats: number[];
  lons: number[];
  depths: number[];
}

export default function TransectChart({
  prediction,
  lats,
  lons,
  depths,
}: TransectChartProps) {
  const [axis, setAxis] = useState<"lat" | "lon">("lat");
  const [sliceIdx, setSliceIdx] = useState(Math.floor(lats.length / 2));

  // Extract 2D transect: [n_depths, n_lons] or [n_depths, n_lats]
  const transect: (number | null)[][] = depths.map((_, di) => {
    if (axis === "lat") {
      // Fixed lat, vary lon
      return (prediction[di]?.[sliceIdx] ?? []) as (number | null)[];
    } else {
      // Fixed lon, vary lat
      return (prediction[di]?.map((row) => row[sliceIdx]) ?? []) as (number | null)[];
    }
  });

  const axisValues = axis === "lat" ? lons : lats;
  const axisMax = axisValues.length - 1;
  const currentVal = axisValues[sliceIdx];
  const axisLabel = axis === "lat"
    ? `Fixed latitude: ${lats[sliceIdx]?.toFixed(2)}°N — across longitude`
    : `Fixed longitude: ${lons[sliceIdx]?.toFixed(2)}°E — across latitude`;

  return (
    <div>
      <div
        style={{ display: "flex", gap: 16, alignItems: "center", marginBottom: 16, flexWrap: "wrap" }}
      >
        <div>
          <label className="field-label">Slice orientation</label>
          <select
            className="field-select"
            style={{ width: 200 }}
            value={axis}
            onChange={(e) => {
              setAxis(e.target.value as "lat" | "lon");
              setSliceIdx(
                e.target.value === "lat"
                  ? Math.floor(lats.length / 2)
                  : Math.floor(lons.length / 2)
              );
            }}
          >
            <option value="lat">Fixed Latitude (Zonal)</option>
            <option value="lon">Fixed Longitude (Meridional)</option>
          </select>
        </div>
        <div style={{ flex: 1, minWidth: 200 }}>
          <label className="field-label">
            {axis === "lat" ? `Latitude — ${lats[sliceIdx]?.toFixed(2)}°N` : `Longitude — ${lons[sliceIdx]?.toFixed(2)}°E`}
          </label>
          <input
            type="range"
            min={0}
            max={axisMax}
            value={sliceIdx}
            onChange={(e) => setSliceIdx(Number(e.target.value))}
            style={{ width: "100%", accentColor: "var(--col-teal)" }}
          />
        </div>
      </div>

      <div
        style={{
          fontSize: "0.78rem",
          color: "var(--col-text-tertiary)",
          marginBottom: 10,
        }}
      >
        {axisLabel} · Depths 0–1000 m (rows, top=surface) · Temperature (°C)
      </div>

      <div className="heatmap-wrap">
        <HeatmapCanvas
          data={transect}
          lats={depths}
          lons={axisValues}
          diverging={false}
          height={320}
        />
      </div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: "0.65rem",
          color: "var(--col-text-tertiary)",
          marginTop: 4,
          fontFamily: "var(--font-mono)",
        }}
      >
        <span>{axisValues[0]?.toFixed(1)}</span>
        <span>{axis === "lat" ? "Longitude (°E)" : "Latitude (°N)"}</span>
        <span>{axisValues[axisValues.length - 1]?.toFixed(1)}</span>
      </div>
    </div>
  );
}
