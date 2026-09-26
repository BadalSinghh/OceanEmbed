"use client";

import React, { useState } from "react";
import PlotlyChart from "@/components/common/PlotlyChart";
import type { PerDepthRow } from "@/types";

interface DepthMetricsChartProps {
  data: PerDepthRow[];
}

export default function DepthMetricsChart({ data }: DepthMetricsChartProps) {
  const [metric, setMetric] = useState<"rmse" | "mae" | "r2">("rmse");

  const depths = data.map((d) => d.depth_m);

  const getMetricData = (model: "oceanembed" | "cbam" | "cnn") => {
    return data.map((d) => {
      if (metric === "rmse") {
        return model === "oceanembed" ? d.oceanembed_rmse : model === "cbam" ? d.cbam_rmse : d.cnn_rmse;
      }
      if (metric === "mae") {
        return model === "oceanembed" ? d.oceanembed_mae : model === "cbam" ? d.cbam_mae : d.cnn_mae;
      }
      return model === "oceanembed" ? d.oceanembed_r2 : model === "cbam" ? d.cbam_r2 : 0;
    });
  };

  const traces: any[] = [
    {
      x: getMetricData("oceanembed"),
      y: depths,
      type: "scatter",
      mode: "lines+markers",
      name: "OceanEmbed (CNN + FNO)",
      line: { color: "#06b6d4", width: 2.6 },
      marker: { color: "#06b6d4", size: 6 },
    },
    {
      x: getMetricData("cbam"),
      y: depths,
      type: "scatter",
      mode: "lines+markers",
      name: "CBAM-CNN",
      line: { color: "#f59e0b", width: 1.8, dash: "dot" },
      marker: { color: "#f59e0b", size: 5 },
    },
    {
      x: getMetricData("cnn"),
      y: depths,
      type: "scatter",
      mode: "lines+markers",
      name: "Standard CNN",
      line: { color: "#818cf8", width: 1.6, dash: "dash" },
      marker: { color: "#818cf8", size: 4 },
    },
  ];

  const layout = {
    title: undefined, // Controlled by HTML header outside!
    xaxis: {
      title: { text: metric === "r2" ? "Coefficient of Determination (R²)" : `${metric.toUpperCase()} [°C]` },
      gridcolor: "rgba(255, 255, 255, 0.05)",
      linecolor: "rgba(255, 255, 255, 0.12)",
      tickfont: { family: "IBM Plex Mono, monospace", size: 10 },
    },
    yaxis: {
      title: { text: "Physical Depth [m]" },
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
      <div className="flex flex-wrap items-center justify-between gap-4 pb-3 rule-b">
        <span className="text-neutral-500 uppercase tracking-wider text-[11px]">
          Depth Level Error Trajectory (0 to 1000 m)
        </span>

        {/* Metric Selector Buttons */}
        <div className="flex items-center gap-2">
          {(["rmse", "mae", "r2"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMetric(m)}
              className={`px-3 py-1 text-xs transition-colors cursor-pointer uppercase ${
                metric === m ? "bg-white/10 text-white font-medium" : "text-neutral-400 hover:text-white"
              }`}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      <div className="w-full h-[580px]">
        <PlotlyChart data={traces} layout={layout} style={{ minHeight: 580 }} />
      </div>
    </div>
  );
}
