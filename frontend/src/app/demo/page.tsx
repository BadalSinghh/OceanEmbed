"use client";

import React, { useState, useEffect, useCallback } from "react";
import dynamic from "next/dynamic";
import { getTestDates, runPredict } from "@/lib/api";
import type { PredictResponse } from "@/types";

const OceanVolume3D = dynamic(() => import("@/components/demo/OceanVolume3D"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-[640px] bg-[#050505] flex flex-col items-center justify-center gap-3 font-mono text-xs text-neutral-500">
      <div className="w-5 h-5 border-2 border-neutral-700 border-t-white rounded-full animate-spin" />
      <span>LOADING 3D TEMPERATURE FIELD...</span>
    </div>
  ),
});

const DepthSlice2D = dynamic(() => import("@/components/demo/DepthSlice2D"), {
  ssr: false,
  loading: () => <div className="w-full h-[600px] bg-[#050505]" />,
});

const VerticalTransectPlotly = dynamic(() => import("@/components/demo/VerticalTransectPlotly"), {
  ssr: false,
  loading: () => <div className="w-full h-[600px] bg-[#050505]" />,
});

const SoundingProfilePlotly = dynamic(() => import("@/components/demo/SoundingProfilePlotly"), {
  ssr: false,
  loading: () => <div className="w-full h-[600px] bg-[#050505]" />,
});

const TARGET_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000];

const MODELS = [
  { id: "cbam_cnn", name: "OceanEmbed (CNN + FNO)", type: "Operator Learning", params: "2.75M" },
  { id: "ocean_embed", name: "CBAM-CNN", type: "Attention Baseline", params: "1.84M" },
];

const SURFACE_CHANNELS = [
  { id: "SST", label: "Sea Surface Temperature", unit: "°C" },
  { id: "SSS", label: "Sea Surface Salinity", unit: "psu" },
  { id: "SLA", label: "Sea Level Anomaly", unit: "m" },
  { id: "CUR_U", label: "Geostrophic Current U", unit: "m/s" },
  { id: "CUR_V", label: "Geostrophic Current V", unit: "m/s" },
  { id: "WND_U", label: "Wind Stress U", unit: "m/s" },
  { id: "WND_V", label: "Wind Stress V", unit: "m/s" },
];

type VizMode = "volume3d" | "depthSlice2d" | "transect" | "profile";
type FieldType = "prediction" | "groundTruth" | "error";

export default function ReconstructionLabPage() {
  const [dates, setDates] = useState<string[]>([]);
  const [sampleIdx, setSampleIdx] = useState<number>(0);
  const [modelId, setModelId] = useState<string>("cbam_cnn");
  const [depthIdx, setDepthIdx] = useState<number>(7); // 100m default
  const [vizMode, setVizMode] = useState<VizMode>("volume3d");
  const [fieldType, setFieldType] = useState<FieldType>("prediction");

  // Probe coordinate inside Bay of Bengal
  const [probeLat, setProbeLat] = useState<number>(13.5);
  const [probeLon, setProbeLon] = useState<number>(88.5);

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [inferenceLatency, setInferenceLatency] = useState<number | null>(null);

  useEffect(() => {
    getTestDates()
      .then((res) => {
        if (res.dates && res.dates.length > 0) {
          setDates(res.dates);
        }
      })
      .catch(() => {
        setErrorMsg("Backend unavailable. Ensure FastAPI server is running on :8000.");
      });
  }, []);

  const executeReconstruction = useCallback(async () => {
    setLoading(true);
    setErrorMsg(null);
    const start = performance.now();
    try {
      const res = await runPredict({
        model_id: modelId,
        sample_index: sampleIdx,
        depth_index: null,
      });
      setResult(res);
      setInferenceLatency(Math.round(performance.now() - start));
    } catch (e: unknown) {
      setErrorMsg(e instanceof Error ? e.message : "Inference failed");
    } finally {
      setLoading(false);
    }
  }, [modelId, sampleIdx]);

  useEffect(() => {
    if (dates.length > 0 && !result && !loading) {
      executeReconstruction();
    }
  }, [dates, result, loading, executeReconstruction]);

  const activeModel = MODELS.find((m) => m.id === modelId) || MODELS[0];

  const getProbedSurfaceVal = (chanIdx: number): string => {
    if (!result || !result.surface_inputs || !result.surface_inputs[chanIdx]) return "—";
    const lats = result.lats;
    const lons = result.lons;
    const latI = lats.reduce(
      (best, cur, idx) => (Math.abs(cur - probeLat) < Math.abs(lats[best] - probeLat) ? idx : best),
      0
    );
    const lonI = lons.reduce(
      (best, cur, idx) => (Math.abs(cur - probeLon) < Math.abs(lons[best] - probeLon) ? idx : best),
      0
    );
    const val = result.surface_inputs[chanIdx]?.[latI]?.[lonI];
    return val !== null && val !== undefined ? Number(val).toFixed(2) : "—";
  };

  return (
    <div className="w-full min-h-[calc(100svh-64px)] bg-[#050505] text-neutral-100 antialiased font-sans">
      {/* ── Workstation Top Status Strip ────────────────────────────── */}
      <div className="w-full px-6 md:px-12 lg:px-16 py-4 rule-b flex flex-wrap items-center justify-between gap-4 font-mono text-xs">
        <div className="flex items-center gap-3">
          <span className="text-white font-medium">RECONSTRUCTION LAB</span>
          <span className="text-neutral-500">|</span>
          <span className="text-neutral-400">BAY OF BENGAL DOMAIN (5°–22°N, 80°–100°E)</span>
        </div>

        <div className="flex items-center gap-6 text-neutral-500 text-[11px]">
          <span>GRID: 69 × 81 (0.25°)</span>
          <span>DEPTH: 0–1000 m (15 LEVELS)</span>
          {inferenceLatency !== null && <span>LATENCY: {inferenceLatency} ms</span>}
        </div>
      </div>

      {errorMsg && (
        <div className="w-full px-6 md:px-12 py-3 bg-red-950/40 text-red-300 rule-b font-mono text-xs flex justify-between items-center">
          <span>{errorMsg}</span>
          <button onClick={() => setErrorMsg(null)} className="text-neutral-400 hover:text-white uppercase">
            Dismiss
          </button>
        </div>
      )}

      {/* ── Main Open Two-Column Scientific Workstation ──────────────── */}
      <div className="w-full px-6 md:px-12 lg:px-16 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-start">
          {/* ── LEFT COLUMN: Input Controls (Open Separators, No Cards) ── */}
          <div className="lg:col-span-4 space-y-8">
            <div className="space-y-6">
              <div className="pb-2 rule-b font-mono text-xs text-neutral-400 uppercase tracking-wider">
                Reconstruction Parameters
              </div>

              {/* Observation Date */}
              <div className="space-y-2">
                <label className="font-mono text-xs text-neutral-400 block">
                  Observation Date
                </label>
                <select
                  value={sampleIdx}
                  onChange={(e) => setSampleIdx(Number(e.target.value))}
                  className="w-full bg-[#0a0a0a] border border-white/10 rounded-none px-3 py-2 text-neutral-200 text-xs font-mono outline-none focus:border-white transition-colors"
                >
                  {dates.map((d, i) => (
                    <option key={d} value={i}>
                      {d} (#{i.toString().padStart(3, "0")})
                    </option>
                  ))}
                </select>
              </div>

              {/* Neural Architecture */}
              <div className="space-y-2">
                <label className="font-mono text-xs text-neutral-400 block">
                  Neural Architecture
                </label>
                <div className="divide-y divide-white/[0.08] rule-b rule-t">
                  {MODELS.map((m) => {
                    const isSelected = m.id === modelId;
                    return (
                      <button
                        key={m.id}
                        type="button"
                        onClick={() => setModelId(m.id)}
                        className={`w-full py-3 px-2 flex items-center justify-between text-left transition-colors cursor-pointer ${
                          isSelected ? "bg-white/[0.06] text-white" : "text-neutral-400 hover:text-neutral-200"
                        }`}
                      >
                        <div>
                          <div className="font-medium text-xs text-neutral-100">{m.name}</div>
                          <div className="text-[10px] text-neutral-500 font-mono">{m.type}</div>
                        </div>
                        <span className="font-mono text-[11px] text-neutral-400">{m.params}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Geographic Probe Sliders */}
              <div className="space-y-3">
                <div className="flex items-center justify-between font-mono text-xs">
                  <span className="text-neutral-400">Probe Coordinate</span>
                  <span className="text-white font-medium">
                    {probeLat.toFixed(2)}°N, {probeLon.toFixed(2)}°E
                  </span>
                </div>

                <div className="space-y-2 font-mono text-[11px] text-neutral-400">
                  <div className="flex justify-between">
                    <span>Latitude: {probeLat.toFixed(1)}°N</span>
                    <span className="text-neutral-600">5°N–22°N</span>
                  </div>
                  <input
                    type="range"
                    min={5.0}
                    max={22.0}
                    step={0.25}
                    value={probeLat}
                    onChange={(e) => setProbeLat(Number(e.target.value))}
                    className="w-full accent-white"
                  />

                  <div className="flex justify-between pt-1">
                    <span>Longitude: {probeLon.toFixed(1)}°E</span>
                    <span className="text-neutral-600">80°E–100°E</span>
                  </div>
                  <input
                    type="range"
                    min={80.0}
                    max={100.0}
                    step={0.25}
                    value={probeLon}
                    onChange={(e) => setProbeLon(Number(e.target.value))}
                    className="w-full accent-white"
                  />
                </div>
              </div>

              {/* Execution Action Button */}
              <div className="pt-2">
                <button
                  type="button"
                  onClick={executeReconstruction}
                  disabled={loading}
                  className="w-full btn-research-primary py-3 text-xs tracking-wider uppercase font-semibold cursor-pointer"
                >
                  {loading ? "INFERRING TEMPERATURE FIELD..." : "RUN RECONSTRUCTION →"}
                </button>
              </div>
            </div>

            {/* Surface Signatures Readout at Probe Point */}
            <div className="space-y-3 rule-t pt-6">
              <div className="font-mono text-xs text-neutral-400 uppercase tracking-wider flex justify-between">
                <span>Surface Signatures</span>
                <span className="text-neutral-500">@ Probe Point</span>
              </div>
              <div className="divide-y divide-white/[0.06] font-mono text-xs">
                {SURFACE_CHANNELS.map((ch, idx) => (
                  <div key={ch.id} className="py-2 flex items-center justify-between">
                    <span className="text-neutral-400">{ch.label}</span>
                    <span className="text-neutral-100 font-medium">
                      {getProbedSurfaceVal(idx)} <span className="text-neutral-500 text-[10px]">{ch.unit}</span>
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* ── RIGHT COLUMN: Primary Scientific Visualization (Dominant) ── */}
          <div className="lg:col-span-8 space-y-6">
            {/* View Mode & Field Controls Strip */}
            <div className="flex flex-wrap items-center justify-between gap-4 pb-3 rule-b font-mono text-xs">
              {/* Viz Mode Selector */}
              <div className="flex items-center gap-2">
                {[
                  { id: "volume3d", label: "3D Volume" },
                  { id: "depthSlice2d", label: "2D Depth Slice" },
                  { id: "transect", label: "Vertical Transect" },
                  { id: "profile", label: "Point Sounding" },
                ].map((m) => (
                  <button
                    key={m.id}
                    onClick={() => setVizMode(m.id as VizMode)}
                    className={`px-3 py-1.5 transition-colors cursor-pointer ${
                      vizMode === m.id
                        ? "bg-white/10 text-white font-medium"
                        : "text-neutral-400 hover:text-white"
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>

              {/* Field Mode Toggle (Hidden in 2D Depth Slice mode since all three are shown simultaneously!) */}
              {vizMode !== "depthSlice2d" ? (
                <div className="flex items-center gap-1">
                  {[
                    { id: "prediction", label: "Prediction" },
                    { id: "groundTruth", label: "GLORYS12" },
                    { id: "error", label: "Residual" },
                  ].map((f) => (
                    <button
                      key={f.id}
                      onClick={() => setFieldType(f.id as FieldType)}
                      className={`px-2.5 py-1 text-[11px] transition-colors cursor-pointer ${
                        fieldType === f.id
                          ? "text-white font-medium underline underline-offset-4"
                          : "text-neutral-500 hover:text-neutral-300"
                      }`}
                    >
                      {f.label}
                    </button>
                  ))}
                </div>
              ) : (
                <div className="flex items-center gap-2 text-[11px] text-cyan-400 font-mono">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                  <span>SIMULTANEOUS TRI-FIELD VIEW (PRED · REF · RESIDUAL)</span>
                </div>
              )}
            </div>

            {/* Depth Scrubber */}
            <div className="flex items-center justify-between gap-4 font-mono text-xs rule-b pb-3">
              <span className="text-neutral-500 uppercase tracking-wider shrink-0 text-[11px]">
                Depth Level:
              </span>
              <div className="flex-1 flex items-center gap-1.5 overflow-x-auto py-1">
                {TARGET_DEPTHS.map((d, idx) => (
                  <button
                    key={d}
                    onClick={() => setDepthIdx(idx)}
                    className={`px-2 py-0.5 text-xs font-mono shrink-0 transition-colors cursor-pointer ${
                      depthIdx === idx
                        ? "bg-white text-black font-semibold"
                        : "text-neutral-400 hover:text-white"
                    }`}
                  >
                    {d}m
                  </button>
                ))}
              </div>
              <span className="text-white font-semibold shrink-0">
                {TARGET_DEPTHS[depthIdx]} m
              </span>
            </div>

            {/* Visualizer Area */}
            <div className="w-full min-h-[640px] relative bg-[#050505]">
              {result ? (
                <>
                  {vizMode === "volume3d" && (
                    <OceanVolume3D
                      prediction={result.prediction as (number | null)[][][]}
                      groundTruth={result.ground_truth as (number | null)[][][]}
                      error={result.error as (number | null)[][][]}
                      depths={TARGET_DEPTHS}
                      lats={result.lats}
                      lons={result.lons}
                      fieldMode={fieldType}
                      activeDepthIndex={depthIdx}
                      onDepthSelect={setDepthIdx}
                      onProbePoint={(lat, lon) => {
                        setProbeLat(lat);
                        setProbeLon(lon);
                      }}
                    />
                  )}

                  {vizMode === "depthSlice2d" && (
                    <DepthSlice2D
                      prediction={result.prediction as (number | null)[][][]}
                      groundTruth={result.ground_truth as (number | null)[][][]}
                      error={result.error as (number | null)[][][]}
                      depthIdx={depthIdx}
                      depth_m={TARGET_DEPTHS[depthIdx]}
                      lats={result.lats}
                      lons={result.lons}
                      model_name={result.model_name}
                      onProbePoint={(lat, lon) => {
                        setProbeLat(lat);
                        setProbeLon(lon);
                      }}
                    />
                  )}

                  {vizMode === "transect" && (
                    <VerticalTransectPlotly
                      prediction={
                        fieldType === "groundTruth"
                          ? (result.ground_truth as (number | null)[][][])
                          : (result.prediction as (number | null)[][][])
                      }
                      depths={TARGET_DEPTHS}
                      lats={result.lats}
                      lons={result.lons}
                      model_name={result.model_name}
                    />
                  )}

                  {vizMode === "profile" && (
                    <SoundingProfilePlotly
                      prediction={result.prediction as (number | null)[][][]}
                      groundTruth={result.ground_truth as (number | null)[][][]}
                      depths={TARGET_DEPTHS}
                      lats={result.lats}
                      lons={result.lons}
                      probeLat={probeLat}
                      probeLon={probeLon}
                      model_name={result.model_name}
                    />
                  )}
                </>
              ) : (
                <div className="w-full h-[640px] flex flex-col items-center justify-center gap-3 font-mono text-xs text-neutral-500">
                  <div className="w-5 h-5 border-2 border-neutral-700 border-t-white rounded-full animate-spin" />
                  <span>INITIALIZING WORKSTATION...</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
