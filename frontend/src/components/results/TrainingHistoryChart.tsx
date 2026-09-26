"use client";

import React, { useState } from "react";
import PlotlyChart from "@/components/common/PlotlyChart";
import type { TrainingHistory } from "@/types";

interface TrainingHistoryChartProps {
  histories: Record<string, TrainingHistory>;
}

const MODEL_CONFIGS: Record<string, { name: string; color: string; dash?: string }> = {
  ocean_embed: { name: "OceanEmbed (CNN + FNO)", color: "#06b6d4" },
  cbam_cnn: { name: "CBAM-CNN", color: "#f59e0b", dash: "dot" },
  cnn_baseline: { name: "Standard CNN", color: "#818cf8", dash: "dash" },
};

export default function TrainingHistoryChart({ histories }: TrainingHistoryChartProps) {
  const [lossView, setLossView] = useState<"both" | "val" | "train">("val");
  const [useLogScale, setUseLogScale] = useState<boolean>(false);

  const traces: any[] = [];

  Object.entries(histories).forEach(([modelId, hist]) => {
    const config = MODEL_CONFIGS[modelId] || { name: modelId, color: "#888" };
    const epochs = Array.from({ length: hist.val_loss?.length || 0 }, (_, i) => i + 1);

    if (lossView === "val" || lossView === "both") {
      traces.push({
        x: epochs,
        y: hist.val_loss,
        type: "scatter",
        mode: "lines",
        name: `${config.name} (Validation)`,
        line: { color: config.color, width: 2.2, dash: config.dash },
      });
    }

    if (lossView === "train" || lossView === "both") {
      traces.push({
        x: epochs,
        y: hist.train_loss,
        type: "scatter",
        mode: "lines",
        name: `${config.name} (Training)`,
        line: { color: config.color, width: 1.4, dash: "dot" },
        opacity: 0.65,
      });
    }
  });

  const layout = {
    title: undefined, // Controlled by HTML header outside!
    xaxis: {
      title: { text: "Training Epoch" },
      gridcolor: "rgba(255, 255, 255, 0.05)",
      linecolor: "rgba(255, 255, 255, 0.12)",
      tickfont: { family: "IBM Plex Mono, monospace", size: 10 },
    },
    yaxis: {
      title: { text: "Masked MSE Loss [°C²]" },
      type: useLogScale ? "log" : "linear",
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
          Masked MSE Loss Trajectory
        </span>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setLossView("val")}
              className={`px-3 py-1 text-xs transition-colors cursor-pointer ${
                lossView === "val" ? "bg-white/10 text-white font-medium" : "text-neutral-400 hover:text-white"
              }`}
            >
              Validation Loss
            </button>
            <button
              onClick={() => setLossView("train")}
              className={`px-3 py-1 text-xs transition-colors cursor-pointer ${
                lossView === "train" ? "bg-white/10 text-white font-medium" : "text-neutral-400 hover:text-white"
              }`}
            >
              Training Loss
            </button>
            <button
              onClick={() => setLossView("both")}
              className={`px-3 py-1 text-xs transition-colors cursor-pointer ${
                lossView === "both" ? "bg-white/10 text-white font-medium" : "text-neutral-400 hover:text-white"
              }`}
            >
              Both
            </button>
          </div>

          <button
            onClick={() => setUseLogScale(!useLogScale)}
            className={`px-3 py-1 border text-xs transition-colors cursor-pointer ${
              useLogScale ? "border-white text-white" : "border-white/10 text-neutral-400"
            }`}
          >
            {useLogScale ? "Log Y" : "Linear Y"}
          </button>
        </div>
      </div>

      <div className="w-full h-[580px]">
        <PlotlyChart data={traces} layout={layout} style={{ minHeight: 580 }} />
      </div>
    </div>
  );
}
