"use client";

import React, { useState } from "react";

interface PipelineStage {
  id: string;
  step: string;
  title: string;
  tensor: string;
  subtitle: string;
  description: string;
}

const STAGES: PipelineStage[] = [
  {
    id: "obs",
    step: "01",
    title: "Satellite Observations",
    tensor: "[7, 69, 81]",
    subtitle: "Multi-Sensor Surface Products",
    description: "Daily L4 gridded observations of SST, SSS, SLA, geostrophic current vectors (U, V), and scatterometer wind stress fields at 0.25° resolution.",
  },
  {
    id: "repr",
    step: "02",
    title: "Surface Representation",
    tensor: "[128, 35, 41]",
    subtitle: "Convolutional Gradient Extraction",
    description: "Hierarchical 2D convolutional layers extract multiscale spatial gradients, capturing eddy boundaries, frontal zones, and regional divergence patterns.",
  },
  {
    id: "embed",
    step: "03",
    title: "Ocean Embedding",
    tensor: "[256, 18, 21]",
    subtitle: "Coupled Spatial Manifold",
    description: "A learned latent manifold encoding the dynamic coupling between surface fluxes and subsurface baroclinic pressure modes.",
  },
  {
    id: "recon",
    step: "04",
    title: "Deep Reconstruction",
    tensor: "FNO2D (16 Modes)",
    subtitle: "Fourier Neural Operator",
    description: "Continuous spectral convolution in Fourier frequency space: (K v)(x) = F^-1(R · F(v))(x), learning the global differential operator.",
  },
  {
    id: "field",
    step: "05",
    title: "3D Temperature Field",
    tensor: "[15, 69, 81]",
    subtitle: "Continuous Water Column (0–1000 m)",
    description: "Reconstructed full 3D temperature tensor across 15 standard depth horizons, resolving the mixed layer and sharp thermocline.",
  },
  {
    id: "argo",
    step: "06",
    title: "Argo In-Situ Validation",
    tensor: "3,509 Soundings",
    subtitle: "CORA v1.3 Quality-Controlled Floats",
    description: "Rigorous empirical validation against 253 independent autonomous profiling CTD floats, achieving 1.38 °C RMSE and R = 0.987.",
  },
];

export default function ResearchPipelineDiagram() {
  const [activeStageId, setActiveStageId] = useState<string>("recon");
  const active = STAGES.find((s) => s.id === activeStageId) || STAGES[3];

  return (
    <div className="w-full space-y-10">
      {/* 6-Stage Open Horizontal Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 divide-y md:divide-y-0 md:divide-x divide-white/[0.08] rule-b rule-t">
        {STAGES.map((st) => {
          const isSelected = st.id === activeStageId;
          return (
            <button
              key={st.id}
              onClick={() => setActiveStageId(st.id)}
              className={`p-6 text-left transition-colors cursor-pointer ${
                isSelected ? "bg-white/[0.05]" : "hover:bg-white/[0.02]"
              }`}
            >
              <div className="font-mono text-[11px] text-neutral-500 mb-2">
                STAGE {st.step}
              </div>
              <div
                className={`text-sm font-medium leading-snug ${
                  isSelected ? "text-white" : "text-neutral-300"
                }`}
              >
                {st.title}
              </div>
              <div className="font-mono text-xs text-neutral-500 mt-4 truncate">
                {st.tensor}
              </div>
            </button>
          );
        })}
      </div>

      {/* Selected Stage Detail Inspector */}
      <div className="py-6 px-8 bg-[#0a0a0a] rule-b rule-t flex flex-col md:flex-row items-start justify-between gap-8">
        <div className="max-w-2xl space-y-2">
          <div className="font-mono text-xs text-cyan-400 uppercase tracking-wider">
            Stage {active.step}: {active.title} — {active.subtitle}
          </div>
          <p className="text-neutral-300 text-sm font-sans leading-relaxed">
            {active.description}
          </p>
        </div>
        <div className="font-mono text-xs text-neutral-400 shrink-0 border-t md:border-t-0 md:border-l border-white/[0.08] pt-4 md:pt-0 md:pl-8">
          <div className="text-neutral-500 uppercase text-[9px]">Active Tensor Dimension</div>
          <div className="text-neutral-100 text-sm font-semibold mt-1">{active.tensor}</div>
        </div>
      </div>
    </div>
  );
}
