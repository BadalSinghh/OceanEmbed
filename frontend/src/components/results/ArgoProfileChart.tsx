"use client";

import React from "react";
import PlotlyChart from "@/components/common/PlotlyChart";
import type { ArgoObservation } from "@/types";

interface ArgoProfileChartProps {
  observations: ArgoObservation[];
}

export default function ArgoProfileChart({ observations }: ArgoProfileChartProps) {
  const sorted = [...observations].sort(
    (a, b) => (a.depth_m ?? 0) - (b.depth_m ?? 0)
  );

  const depths = sorted.map((o) => o.depth_m);
  const obsTemp = sorted.map((o) => o.obs_temp);
  const glorysTemp = sorted.map((o) => o.glorys_temp);
  const fnoTemp = sorted.map((o) => o.cbam_temp);
  const cbamTemp = sorted.map((o) => o.oe_temp);

  const traces: any[] = [
    {
      x: obsTemp,
      y: depths,
      type: "scatter",
      mode: "lines+markers",
      name: "In-Situ Argo Sounding (Ground Truth)",
      line: { color: "#ffffff", width: 2.8 },
      marker: { color: "#ffffff", size: 6 },
    },
    {
      x: fnoTemp,
      y: depths,
      type: "scatter",
      mode: "lines+markers",
      name: "OceanEmbed (CNN + FNO)",
      line: { color: "#06b6d4", width: 2.5 },
      marker: { color: "#06b6d4", size: 5 },
    },
    {
      x: glorysTemp,
      y: depths,
      type: "scatter",
      mode: "lines",
      name: "GLORYS12 Reanalysis",
      line: { color: "#fb923c", width: 2.0, dash: "dash" },
    },
    {
      x: cbamTemp,
      y: depths,
      type: "scatter",
      mode: "lines",
      name: "CBAM-CNN",
      line: { color: "#818cf8", width: 1.8, dash: "dot" },
    },
  ];

  const layout = {
    title: undefined, // Controlled by HTML header outside!
    xaxis: {
      title: { text: "Temperature [°C]" },
      gridcolor: "rgba(255, 255, 255, 0.05)",
      linecolor: "rgba(255, 255, 255, 0.12)",
      tickfont: { family: "IBM Plex Mono, monospace", size: 10 },
    },
    yaxis: {
      title: { text: "Sounding Depth [m]" },
      autorange: "reversed",
      gridcolor: "rgba(255, 255, 255, 0.05)",
      linecolor: "rgba(255, 255, 255, 0.12)",
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
    height: 580,
  };

  return (
    <div className="w-full space-y-4 font-mono text-xs">
      <div className="w-full h-[580px]">
        <PlotlyChart data={traces} layout={layout} style={{ minHeight: 580 }} />
      </div>
    </div>
  );
}
