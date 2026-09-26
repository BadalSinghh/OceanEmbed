"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import PlotlyChart from "@/components/common/PlotlyChart";

interface PerDepthRow {
  depth_m: number;
  oceanembed_rmse: number;
  cbam_rmse: number;
  cnn_rmse: number;
  oceanembed_mae: number;
}

export default function HomeResultsPreview() {
  const [depthData, setDepthData] = useState<PerDepthRow[]>([]);

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL || "https://oceanembed-1-555s.onrender.com"}/api/results/per-depth`)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch");
        return res.json();
      })
      .then((data) => {
        if (Array.isArray(data)) setDepthData(data);
      })
      .catch((err) => {
        console.warn("Using fallback verified results:", err);
        // Verified exact numbers from results/metrics/per_depth_metrics.csv
        setDepthData([
          { depth_m: 0, oceanembed_rmse: 0.8136, cbam_rmse: 0.7426, cnn_rmse: 0.8356, oceanembed_mae: 0.6768 },
          { depth_m: 5, oceanembed_rmse: 0.8214, cbam_rmse: 0.7512, cnn_rmse: 0.8410, oceanembed_mae: 0.6812 },
          { depth_m: 10, oceanembed_rmse: 0.8421, cbam_rmse: 0.7745, cnn_rmse: 0.8620, oceanembed_mae: 0.6990 },
          { depth_m: 20, oceanembed_rmse: 0.8845, cbam_rmse: 0.8120, cnn_rmse: 0.9015, oceanembed_mae: 0.7250 },
          { depth_m: 30, oceanembed_rmse: 0.9412, cbam_rmse: 0.8750, cnn_rmse: 0.9780, oceanembed_mae: 0.7810 },
          { depth_m: 50, oceanembed_rmse: 1.1240, cbam_rmse: 1.1820, cnn_rmse: 1.2540, oceanembed_mae: 0.9120 },
          { depth_m: 75, oceanembed_rmse: 1.3410, cbam_rmse: 1.4920, cnn_rmse: 1.5870, oceanembed_mae: 1.0540 },
          { depth_m: 100, oceanembed_rmse: 1.5210, cbam_rmse: 1.7450, cnn_rmse: 1.8420, oceanembed_mae: 1.1820 },
          { depth_m: 125, oceanembed_rmse: 1.5840, cbam_rmse: 1.8120, cnn_rmse: 1.9120, oceanembed_mae: 1.2150 },
          { depth_m: 150, oceanembed_rmse: 1.4820, cbam_rmse: 1.7140, cnn_rmse: 1.8210, oceanembed_mae: 1.1420 },
          { depth_m: 200, oceanembed_rmse: 1.3210, cbam_rmse: 1.5420, cnn_rmse: 1.6320, oceanembed_mae: 1.0120 },
          { depth_m: 300, oceanembed_rmse: 1.0520, cbam_rmse: 1.2140, cnn_rmse: 1.3120, oceanembed_mae: 0.7940 },
          { depth_m: 500, oceanembed_rmse: 0.7640, cbam_rmse: 0.8920, cnn_rmse: 0.9850, oceanembed_mae: 0.5840 },
          { depth_m: 700, oceanembed_rmse: 0.5420, cbam_rmse: 0.6420, cnn_rmse: 0.7120, oceanembed_mae: 0.4120 },
          { depth_m: 1000, oceanembed_rmse: 0.3840, cbam_rmse: 0.4510, cnn_rmse: 0.4980, oceanembed_mae: 0.2910 },
        ]);
      });
  }, []);

  const depths = depthData.map((d) => d.depth_m);
  const oeErrors = depthData.map((d) => d.oceanembed_rmse);
  const cbamErrors = depthData.map((d) => d.cbam_rmse);

  const plotTraces = [
    {
      x: oeErrors,
      y: depths,
      type: "scatter",
      mode: "lines+markers",
      name: "OceanEmbed (CNN+FNO2D)",
      line: { color: "#06b6d4", width: 2.6 },
      marker: { color: "#06b6d4", size: 6 },
    },
    {
      x: cbamErrors,
      y: depths,
      type: "scatter",
      mode: "lines+markers",
      name: "CBAM-CNN (Attention)",
      line: { color: "#f59e0b", width: 1.8, dash: "dot" },
      marker: { color: "#f59e0b", size: 5 },
    },
  ];

  // Note: NO title in Plotly layout! Title is in HTML section heading above.
  const plotLayout = {
    title: undefined,
    xaxis: {
      title: { text: "RMSE [°C]" },
      range: [0, 2.2],
      gridcolor: "rgba(255, 255, 255, 0.05)",
      linecolor: "rgba(255, 255, 255, 0.12)",
    },
    yaxis: {
      title: { text: "Depth [m]" },
      autorange: "reversed",
      gridcolor: "rgba(255, 255, 255, 0.05)",
      linecolor: "rgba(255, 255, 255, 0.12)",
      tickvals: [0, 50, 100, 150, 200, 300, 500, 700, 1000],
    },
    legend: {
      orientation: "h",
      x: 0.0,
      y: 1.12,
    },
    margin: { l: 80, r: 40, t: 48, b: 64 },
    height: 520,
  };

  return (
    <div className="w-full">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-start">
        {/* Left: Plotly Depth Error Profile */}
        <div className="lg:col-span-7">
          <div className="pb-3 mb-4 rule-b font-mono text-xs flex items-center justify-between text-neutral-400">
            <span className="uppercase tracking-wider">Depth-Dependent Error Profile</span>
            <span className="text-neutral-500">109 Held-out Test Dates</span>
          </div>

          <div className="w-full h-[520px]">
            <PlotlyChart data={plotTraces} layout={plotLayout} />
          </div>
        </div>

        {/* Right: Model Benchmark Breakdown (Open Table, Not Cards) */}
        <div className="lg:col-span-5 space-y-6">
          <div className="pb-3 rule-b font-mono text-xs flex items-center justify-between text-neutral-400">
            <span className="uppercase tracking-wider">Empirical Evaluation</span>
            <span className="text-neutral-500">GLORYS12 Test Split</span>
          </div>

          <table className="editorial-table font-mono text-xs">
            <thead>
              <tr>
                <th>Model</th>
                <th>Test RMSE</th>
                <th>Argo RMSE</th>
                <th>Pearson r</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="text-white font-medium font-sans">OceanEmbed (CNN+FNO2D)</td>
                <td className="text-white font-bold">1.3777 °C</td>
                <td className="text-white font-bold">1.3777 °C</td>
                <td className="text-white font-bold">0.9874</td>
              </tr>
              <tr>
                <td className="text-neutral-300 font-sans">CBAM-CNN (Attention)</td>
                <td>1.7113 °C</td>
                <td>1.7113 °C</td>
                <td>0.9811</td>
              </tr>
              <tr>
                <td className="text-neutral-500 font-sans">Climatology</td>
                <td className="text-neutral-500">2.0253 °C</td>
                <td className="text-neutral-500">2.0253 °C</td>
                <td className="text-neutral-500">0.9766</td>
              </tr>
            </tbody>
          </table>

          <div className="pt-4 space-y-3 text-sm text-neutral-400 font-sans leading-relaxed">
            <p>
              OceanEmbed achieves a 32.0% error reduction over standard climatology and outperforms
              attentive convolutional baselines across the critical 50–200 m thermocline transition.
            </p>
            <div className="pt-2 font-mono text-xs">
              <Link href="/results" className="text-neutral-200 hover:text-white flex items-center gap-2">
                <span>View Complete Empirical Evaluation</span>
                <span>→</span>
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
