"use client";

import React, { useState } from "react";
import Link from "next/link";

interface LayerDetail {
  id: string;
  name: string;
  tensorIn: string;
  tensorOut: string;
  params: string;
  summary: string;
  operations: string[];
  mathEquation?: string;
}

const ARCHITECTURE_STAGES: LayerDetail[] = [
  {
    id: "obs",
    name: "7-Channel Surface Observation",
    tensorIn: "CMEMS Products",
    tensorOut: "[B, 7, 69, 81]",
    params: "Raw Observables",
    summary:
      "Seven daily satellite observable fields over the Bay of Bengal (5°–22°N, 80°–100°E). Normalised by empirical training statistics.",
    operations: [
      "Sea Surface Temperature (OSTIA L4, °C)",
      "Sea Surface Salinity (CMEMS L4, psu)",
      "Sea Level Anomaly (DUACS L4 altimetry, m)",
      "Geostrophic Currents U & V (derived from SLA, m/s)",
      "Wind Stress U & V at 10m (Scatterometer L4, m/s)",
    ],
  },
  {
    id: "encoder",
    name: "CNN Spatial Encoder",
    tensorIn: "[B, 7, 69, 81]",
    tensorOut: "[B, 128, 17, 20]",
    params: "182,400 parameters",
    summary:
      "Multiscale 2D convolutional layers extract spatial gradients, capturing eddy frontal boundaries and coastal boundary current shear.",
    operations: [
      "Conv2d(7 → 32, kernel=3×3, pad=1) + BatchNorm2d + LeakyReLU",
      "Conv2d(32 → 64, kernel=3×3, stride=2, pad=1) → [B, 64, 35, 41]",
      "Conv2d(64 → 128, kernel=3×3, stride=2, pad=1) → [B, 128, 18, 21]",
      "Spatial interpolation to latent mesh [B, 128, 17, 20]",
    ],
  },
  {
    id: "latent",
    name: "Latent Ocean Representation",
    tensorIn: "[B, 128, 17, 20]",
    tensorOut: "[B, 128, 17, 20]",
    params: "32,896 parameters",
    summary:
      "Learned latent manifold encoding non-local coupling between surface wind-stress curl, steric height, and subsurface baroclinic modes.",
    operations: [
      "1×1 Linear projection across feature channels",
      "Layer normalization across spatial dimensions",
      "Residual identity shortcut path",
    ],
  },
  {
    id: "fno",
    name: "Fourier Neural Operator (FNO2D)",
    tensorIn: "[B, 128, 17, 20]",
    tensorOut: "[B, 128, 17, 20]",
    params: "2,097,152 parameters",
    summary:
      "Four SpectralConv2d layers evaluate learned continuous integral kernels in Fourier frequency space, parameterizing the differential operator.",
    operations: [
      "Forward 2D Real FFT: x_ft = rfft2(x)",
      "Mode Truncation: 16 lowest Fourier modes along spatial dimensions",
      "Complex Tensor Multiplication: out_ft = compl_mul(x_ft, W_spectral)",
      "Inverse 2D Real FFT: x_out = irfft2(out_ft)",
      "Parallel 1×1 Conv2d local bypass + GELU non-linear activation",
    ],
    mathEquation: "𝒦(v)(x) = ℱ⁻¹( R · (ℱv) )(x) + W · v(x)",
  },
  {
    id: "depth",
    name: "15 Depth Levels Projection",
    tensorIn: "[B, 128, 17, 20]",
    tensorOut: "[B, 15, 69, 81]",
    params: "439,935 parameters",
    summary:
      "Bilinear spatial upsampling to the 0.25° grid conditioned with sinusoidal depth embeddings across the 15 standard depth horizons.",
    operations: [
      "Bilinear spatial upsampling: [17, 20] → [35, 41] → [69, 81]",
      "Sinusoidal depth positional encoding vector (dim=32)",
      "Target depths: 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m",
    ],
  },
  {
    id: "field",
    name: "3D Temperature Field",
    tensorIn: "[B, 15, 69, 81]",
    tensorOut: "Physical Grid (0–1000 m)",
    params: "Final Inferred Volume",
    summary:
      "Continuous 3D temperature volume over the Bay of Bengal, land-masked with physical coastal bathymetry.",
    operations: [
      "15 Target Depths across 69 × 81 geographic bins",
      "Land mask applied from GLORYS12 bathymetric mask",
      "Validated against in-situ autonomous Argo CTD floats",
    ],
  },
];

export default function ArchitecturePage() {
  const [expandedStage, setExpandedStage] = useState<string>("fno");

  return (
    <div className="w-full min-h-screen bg-[#050505] text-neutral-100 antialiased font-sans pb-36">
      {/* ── Editorial Header ────────────────────────────────────────── */}
      <div className="w-full px-6 md:px-16 lg:px-24 xl:px-32 py-24 rule-b">
        <div className="max-w-4xl space-y-4">
          <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest">
            TECHNICAL EXPLAINER / DEEP OPERATOR LEARNING
          </div>
          <h1 className="text-4xl sm:text-6xl font-light text-white tracking-tight uppercase">
            HOW OCEANEMBED RECONSTRUCTS THE WATER COLUMN
          </h1>
          <p className="text-lg sm:text-xl text-neutral-300 font-sans leading-relaxed pt-2 max-w-3xl">
            A visual explanation of the neural operator pipeline that inverts surface-only
            satellite observations into the continuous 3D subsurface temperature volume.
          </p>

          <div className="pt-8 border-t border-white/[0.08] flex flex-wrap items-center gap-12 font-mono text-xs text-neutral-400">
            <div>
              <span className="text-neutral-500 uppercase text-[10px] block">Total Trainable Parameters</span>
              <span className="text-white mt-0.5 block font-semibold">8,670,241 parameters</span>
            </div>
            <div>
              <span className="text-neutral-500 uppercase text-[10px] block">Operator Formulation</span>
              <span className="text-white mt-0.5 block">Fourier Neural Operator (FNO2D)</span>
            </div>
            <div>
              <span className="text-neutral-500 uppercase text-[10px] block">Input to Output</span>
              <span className="text-white mt-0.5 block">[7, 69, 81] → [15, 69, 81]</span>
            </div>
          </div>
        </div>
      </div>

      <div className="w-full px-6 md:px-16 lg:px-24 xl:px-32 mt-20 space-y-24">
        {/* ── LARGE OPEN ARCHITECTURE DIAGRAM ─────────────────────────── */}
        <section className="space-y-8">
          <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest pb-3 rule-b">
            NEURAL RECONSTRUCTION ARCHITECTURE
          </div>

          <div className="space-y-6">
            {ARCHITECTURE_STAGES.map((st, i) => {
              const isExpanded = expandedStage === st.id;
              return (
                <div key={st.id} className="rule-b pb-6 space-y-3">
                  <div
                    onClick={() => setExpandedStage(isExpanded ? "" : st.id)}
                    className="flex flex-col md:flex-row md:items-center justify-between gap-4 cursor-pointer py-2 group"
                  >
                    <div className="flex items-baseline gap-6">
                      <span className="font-mono text-xs text-neutral-500 group-hover:text-white transition-colors">
                        0{i + 1}
                      </span>
                      <h3 className="text-xl sm:text-2xl font-light text-white group-hover:text-neutral-200 transition-colors">
                        {st.name}
                      </h3>
                    </div>

                    <div className="flex items-center gap-6 font-mono text-xs text-neutral-400">
                      <span className="text-neutral-500">{st.tensorIn} →</span>
                      <span className="text-white font-medium">{st.tensorOut}</span>
                      <span className="text-neutral-600">{st.params}</span>
                      <span className="text-neutral-400 group-hover:text-white transition-colors">
                        {isExpanded ? "−" : "+"}
                      </span>
                    </div>
                  </div>

                  {/* Expandable Technical Details */}
                  {isExpanded && (
                    <div className="pl-0 md:pl-12 pt-4 space-y-4 text-sm font-sans text-neutral-300">
                      <p className="max-w-3xl leading-relaxed">{st.summary}</p>

                      {st.mathEquation && (
                        <div className="p-4 bg-[#0a0a0a] rule-b rule-t font-mono text-xs text-neutral-200">
                          <span className="text-neutral-500 uppercase text-[10px] block mb-1">
                            Mathematical Formulation
                          </span>
                          <span className="text-white font-semibold text-sm">{st.mathEquation}</span>
                        </div>
                      )}

                      <div className="font-mono text-xs text-neutral-400 space-y-1.5 pt-2">
                        {st.operations.map((op, idx) => (
                          <div key={idx} className="flex items-center gap-3">
                            <span className="text-neutral-600">—</span>
                            <span>{op}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </section>

        {/* ── IMPLEMENTED MODEL ABLATION COMPARISON ────────────────────── */}
        <section className="space-y-6">
          <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest pb-3 rule-b">
            ABLATION BENCHMARK & PARAMETER COUNTS
          </div>

          <table className="editorial-table font-mono text-xs">
            <thead>
              <tr>
                <th>Model</th>
                <th>Paradigm</th>
                <th>Parameters</th>
                <th>Checkpoint</th>
                <th>In-Situ Argo RMSE</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-l-2 border-white">
                <td className="text-white font-semibold font-sans">OceanEmbed (CNN+FNO2D)</td>
                <td className="text-neutral-300 font-sans">Fourier Neural Operator</td>
                <td className="text-white font-bold">8,670,241</td>
                <td>oceanembed_best.pt (~205 MB)</td>
                <td className="text-white font-bold">1.3777 °C</td>
              </tr>
              <tr>
                <td className="text-neutral-200 font-sans">CBAM-CNN (Attention)</td>
                <td className="text-neutral-400 font-sans">Channel &amp; Spatial Attention</td>
                <td>195,705</td>
                <td>cbam_cnn_best.pt (~2.4 MB)</td>
                <td>1.7113 °C</td>
              </tr>
            </tbody>
          </table>
        </section>

        {/* ── CTA ─────────────────────────────────────────────────────── */}
        <div className="pt-12 rule-t flex flex-wrap items-center justify-between gap-6">
          <div>
            <div className="text-xl font-light text-white">Evaluate the reconstruction live</div>
            <div className="text-sm text-neutral-400 font-sans mt-1">
              Select any test date to inspect the 3D inferred temperature field.
            </div>
          </div>
          <Link href="/demo" className="btn-research-primary">
            <span>Launch Reconstruction Lab</span>
            <span>→</span>
          </Link>
        </div>
      </div>
    </div>
  );
}
