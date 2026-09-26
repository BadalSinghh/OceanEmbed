"use client";

import React, { useEffect, useState } from "react";
import PlotlyChart from "@/components/common/PlotlyChart";
import { getArgoObservations } from "@/lib/api";
import type { ArgoObservation } from "@/types";

interface ArgoScatterChartProps {
  profileIds: string[];
}

export default function ArgoScatterChart({ profileIds }: ArgoScatterChartProps) {
  const [obs, setObs] = useState<ArgoObservation[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    getArgoObservations(undefined, 1000)
      .then((r) => setObs(r.data))
      .catch((err) => console.error("Failed to load Argo scatter data:", err))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="w-full h-80 flex flex-col items-center justify-center gap-2 font-mono text-xs text-neutral-500">
        <div className="w-5 h-5 border-2 border-neutral-700 border-t-white rounded-full animate-spin" />
        <span>LOADING ARGO FLOAT OBSERVATIONS...</span>
      </div>
    );
  }

  if (obs.length === 0) return null;

  const validFno = obs.filter((o) => o.obs_temp != null && o.cbam_temp != null);
  const validCbam = obs.filter((o) => o.obs_temp != null && o.oe_temp != null);

  const traces: any[] = [
    {
      x: [4, 31],
      y: [4, 31],
      type: "scatter",
      mode: "lines",
      name: "1:1 Identity Reference",
      line: { color: "rgba(255, 255, 255, 0.45)", width: 1.5, dash: "dash" },
      hoverinfo: "none",
    },
    {
      x: validFno.map((o) => o.obs_temp),
      y: validFno.map((o) => o.cbam_temp),
      text: validFno.map(
        (o) => `Float: ${o.profile_id}<br>Depth: ${o.depth_m}m<br>Obs: ${o.obs_temp?.toFixed(2)}°C<br>Pred: ${o.cbam_temp?.toFixed(2)}°C`
      ),
      type: "scatter",
      mode: "markers",
      name: "OceanEmbed (CNN + FNO) [R = 0.987, RMSE = 1.38 °C]",
      marker: { color: "#06b6d4", size: 5, opacity: 0.75 },
      hoverinfo: "text",
    },
    {
      x: validCbam.map((o) => o.obs_temp),
      y: validCbam.map((o) => o.oe_temp),
      text: validCbam.map(
        (o) => `Float: ${o.profile_id}<br>Depth: ${o.depth_m}m<br>Obs: ${o.obs_temp?.toFixed(2)}°C<br>Pred: ${o.oe_temp?.toFixed(2)}°C`
      ),
      type: "scatter",
      mode: "markers",
      name: "CBAM-CNN [R = 0.981, RMSE = 1.71 °C]",
      marker: { color: "#f59e0b", size: 4, opacity: 0.55 },
      hoverinfo: "text",
    },
  ];

  const layout = {
    title: undefined, // Controlled by HTML header outside!
    xaxis: {
      title: { text: "In-Situ Argo Float Temperature [°C]" },
      range: [4, 32],
      gridcolor: "rgba(255, 255, 255, 0.05)",
      linecolor: "rgba(255, 255, 255, 0.12)",
      tickfont: { family: "IBM Plex Mono, monospace", size: 10 },
    },
    yaxis: {
      title: { text: "Model Predicted Temperature [°C]" },
      range: [4, 32],
      gridcolor: "rgba(255, 255, 255, 0.05)",
      linecolor: "rgba(255, 255, 255, 0.12)",
      tickfont: { family: "IBM Plex Mono, monospace", size: 10 },
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
      <div className="flex flex-wrap items-center justify-between gap-4 pb-3 rule-b">
        <span className="text-neutral-500 uppercase tracking-wider text-[11px]">
          CORA v1.3 In-Situ Matched Soundings ({obs.length} points)
        </span>
        <span className="text-neutral-500">1:1 perfect agreement diagonal shown</span>
      </div>

      <div className="w-full h-[580px]">
        <PlotlyChart data={traces} layout={layout} style={{ minHeight: 580 }} />
      </div>
    </div>
  );
}
