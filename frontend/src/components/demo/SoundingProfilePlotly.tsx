"use client";

import React from "react";
import PlotlyChart from "@/components/common/PlotlyChart";

interface SoundingProfileProps {
  prediction: (number | null)[][][];
  groundTruth?: (number | null)[][][];
  depths: number[];
  lats: number[];
  lons: number[];
  probeLat: number;
  probeLon: number;
  model_name: string;
}

export default function SoundingProfilePlotly({
  prediction,
  groundTruth,
  depths,
  lats,
  lons,
  probeLat,
  probeLon,
  model_name,
}: SoundingProfileProps) {
  const latIdx = lats.reduce(
    (best, curr, idx) => (Math.abs(curr - probeLat) < Math.abs(lats[best] - probeLat) ? idx : best),
    0
  );
  const lonIdx = lons.reduce(
    (best, curr, idx) => (Math.abs(curr - probeLon) < Math.abs(lons[best] - probeLon) ? idx : best),
    0
  );

  const predProfile = depths.map((_, d) => prediction[d]?.[latIdx]?.[lonIdx] ?? null);
  const gtProfile = groundTruth ? depths.map((_, d) => groundTruth[d]?.[latIdx]?.[lonIdx] ?? null) : null;

  const traces: any[] = [
    {
      x: predProfile,
      y: depths,
      type: "scatter",
      mode: "lines+markers",
      name: `${model_name} Prediction`,
      line: { color: "#06b6d4", width: 2.6 },
      marker: { color: "#06b6d4", size: 6 },
    },
  ];

  if (gtProfile) {
    traces.push({
      x: gtProfile,
      y: depths,
      type: "scatter",
      mode: "lines+markers",
      name: "GLORYS12 Ground Truth",
      line: { color: "#fb923c", width: 2.0, dash: "dash" },
      marker: { color: "#fb923c", size: 5 },
    });
  }

  const layout = {
    title: undefined, // Controlled by HTML header outside!
    xaxis: {
      title: { text: "Temperature [°C]" },
      range: [4, 32],
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
      tickvals: [0, 50, 100, 150, 200, 300, 500, 700, 1000],
    },
    legend: {
      orientation: "h",
      x: 0.0,
      y: 1.10,
      font: { family: "IBM Plex Mono, monospace", size: 11, color: "#a1a1aa" },
    },
    margin: { l: 80, r: 40, t: 48, b: 64 },
    height: 600,
  };

  return (
    <div className="w-full bg-[#050505] p-2 font-mono text-xs">
      <div className="w-full h-[600px]">
        <PlotlyChart data={traces} layout={layout} style={{ minHeight: 600 }} />
      </div>
    </div>
  );
}
