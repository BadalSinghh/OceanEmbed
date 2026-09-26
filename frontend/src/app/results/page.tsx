"use client";

import React, { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import {
  getEvaluation,
  getPerDepthMetrics,
  getTrainingHistory,
  getArgoAggregate,
  getArgoProfiles,
  getArgoObservations,
} from "@/lib/api";
import type {
  EvaluationSummary,
  PerDepthRow,
  TrainingHistory,
  ArgoAggregate,
  ArgoProfile,
  ArgoObservation,
} from "@/types";

const DepthMetricsChart = dynamic(
  () => import("@/components/results/DepthMetricsChart"),
  { ssr: false }
);
const TrainingHistoryChart = dynamic(
  () => import("@/components/results/TrainingHistoryChart"),
  { ssr: false }
);
const ArgoScatterChart = dynamic(
  () => import("@/components/results/ArgoScatterChart"),
  { ssr: false }
);
const ArgoProfileChart = dynamic(
  () => import("@/components/results/ArgoProfileChart"),
  { ssr: false }
);

export default function ResultsPage() {
  const [evalData, setEvalData] = useState<EvaluationSummary | null>(null);
  const [perDepth, setPerDepth] = useState<PerDepthRow[]>([]);
  const [histories, setHistories] = useState<Record<string, TrainingHistory>>({});
  const [argoAgg, setArgoAgg] = useState<ArgoAggregate | null>(null);
  const [argoProfiles, setArgoProfiles] = useState<ArgoProfile[]>([]);
  const [selectedProfile, setSelectedProfile] = useState<string>("");
  const [argoObs, setArgoObs] = useState<ArgoObservation[]>([]);
  const [loadingObs, setLoadingObs] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      getEvaluation().catch(() => null),
      getPerDepthMetrics().catch(() => []),
      getTrainingHistory().catch(() => ({})),
      getArgoAggregate().catch(() => null),
      getArgoProfiles().catch(() => ({ profiles: [], count: 0 })),
    ])
      .then(([ev, pd, hist, agg, profilesRes]) => {
        if (ev) setEvalData(ev);
        if (pd && pd.length > 0) setPerDepth(pd);
        if (hist) setHistories(hist);
        if (agg) setArgoAgg(agg);
        if (profilesRes && profilesRes.profiles.length > 0) {
          setArgoProfiles(profilesRes.profiles);
          setSelectedProfile(profilesRes.profiles[0].profile_id);
        }
      })
      .catch((err) => {
        console.warn("API loading issue:", err);
        setErrorMsg("Failed to synchronize with backend results API.");
      });
  }, []);

  useEffect(() => {
    if (!selectedProfile) return;
    setLoadingObs(true);
    getArgoObservations(selectedProfile, 200)
      .then((r) => setArgoObs(r.data))
      .catch((err) => console.error("Failed to load profile obs:", err))
      .finally(() => setLoadingObs(false));
  }, [selectedProfile]);

  return (
    <div className="w-full min-h-screen bg-[#050505] text-neutral-100 antialiased font-sans pb-36">
      {/* ── Editorial Header ────────────────────────────────────────── */}
      <div className="w-full px-6 md:px-16 lg:px-24 xl:px-32 py-24 rule-b">
        <div className="max-w-4xl space-y-4">
          <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest">
            EMPIRICAL EVALUATION & VALIDATION
          </div>
          <h1 className="text-5xl sm:text-7xl font-light text-white tracking-tight">
            RESULTS
          </h1>
          <p className="text-lg sm:text-xl text-neutral-300 font-sans leading-relaxed pt-2 max-w-3xl">
            Quantitative evaluation of OceanEmbed against the implemented baseline architectures
            on the out-of-sample GLORYS12 test split and independent in-situ Argo profiling floats.
          </p>

          <div className="pt-8 border-t border-white/[0.08] flex flex-wrap items-center gap-12 font-mono text-xs text-neutral-400">
            <div>
              <span className="text-neutral-500 uppercase text-[10px] block">Test Split</span>
              <span className="text-neutral-200 mt-0.5 block">2023-09-14 to 2023-12-31 (109 days)</span>
            </div>
            <div>
              <span className="text-neutral-500 uppercase text-[10px] block">In-Situ Float Corpus</span>
              <span className="text-neutral-200 mt-0.5 block">253 Floats (3,509 Matched CTD Soundings)</span>
            </div>
            <div>
              <span className="text-neutral-500 uppercase text-[10px] block">Vertical Scope</span>
              <span className="text-neutral-200 mt-0.5 block">0 to 1000 m (15 Depth Horizons)</span>
            </div>
          </div>
        </div>
      </div>

      <div className="w-full px-6 md:px-16 lg:px-24 xl:px-32 mt-20 space-y-32">
        {errorMsg && (
          <div className="p-4 bg-red-950/40 text-red-300 font-mono text-xs rule-b">
            {errorMsg}
          </div>
        )}

        {/* ── SECTION 1: OVERALL PERFORMANCE ─────────────────────────── */}
        <section id="overall-performance" className="space-y-6">
          <div className="space-y-2">
            <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest">
              01 / BENCHMARK LEADERBOARD
            </div>
            <h2 className="text-3xl sm:text-4xl font-light text-white tracking-tight">
              Overall Model Performance
            </h2>
          </div>

          <div className="w-full overflow-x-auto">
            <table className="editorial-table font-mono text-xs">
              <thead>
                <tr>
                  <th>Architecture</th>
                  <th>Parameters</th>
                  <th>Overall RMSE (°C)</th>
                  <th>Overall MAE (°C)</th>
                  <th>Mean Bias (°C)</th>
                  <th>Pearson R</th>
                  <th>R²</th>
                </tr>
              </thead>
              <tbody>
                <tr className="bg-white/[0.02]">
                  <td className="text-white font-medium font-sans">GLORYS12 Reanalysis</td>
                  <td className="text-neutral-500">—</td>
                  <td className="text-white font-bold">0.8061</td>
                  <td className="text-white font-bold">0.4565</td>
                  <td className="text-white font-bold">+0.2629</td>
                  <td className="text-white font-bold">0.9958</td>
                  <td className="text-white font-bold">0.9904</td>
                </tr>

                <tr className="border-l-2 border-white">
                  <td className="text-white font-semibold font-sans">OceanEmbed (CNN+FNO2D)</td>
                  <td className="text-neutral-300">8,670,241</td>
                  <td className="text-white font-bold">1.3777</td>
                  <td className="text-white font-bold">0.7905</td>
                  <td className="text-white font-bold">+0.3735</td>
                  <td className="text-white font-bold">0.9874</td>
                  <td className="text-white font-bold">0.9720</td>
                </tr>

                <tr>
                  <td className="text-neutral-200 font-sans">CBAM-CNN (Attention)</td>
                  <td className="text-neutral-400">195,705</td>
                  <td>1.7113</td>
                  <td>1.1496</td>
                  <td>+0.6032</td>
                  <td>0.9811</td>
                  <td>0.9568</td>
                </tr>

                <tr className="text-neutral-500">
                  <td className="text-neutral-500 font-sans">Climatology Baseline</td>
                  <td className="text-neutral-500">0</td>
                  <td>2.0253</td>
                  <td>1.2297</td>
                  <td>+0.9484</td>
                  <td>0.9766</td>
                  <td>0.9394</td>
                </tr>
              </tbody>
            </table>
          </div>

          <p className="text-neutral-400 text-sm font-sans pt-2">
            OceanEmbed (CNN+FNO2D) achieves an overall RMSE of 1.3777 °C and Pearson R of 0.9874,
            outperforming the CBAM-CNN attention baseline (1.7113 °C) and monthly climatology (2.0253 °C).
          </p>
        </section>

        {/* ── SECTION 2: DEPTH-DEPENDENT ERROR ────────────────────────── */}
        <section id="depth-dependent-error" className="space-y-6">
          <div className="space-y-2">
            <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest">
              02 / VERTICAL WATER COLUMN
            </div>
            <h2 className="text-3xl sm:text-4xl font-light text-white tracking-tight">
              Depth-Dependent Reconstruction Error
            </h2>
            <p className="text-neutral-400 text-sm font-sans max-w-3xl leading-relaxed">
              In physical oceanography, vertical reconstruction error is highly depth-dependent.
              Error peaks in the sharp pycnocline layer (75–150 m) where internal waves and baroclinic
              eddy heaving are strongest, and reduces significantly below 300 m.
            </p>
          </div>

          <DepthMetricsChart data={perDepth} />
        </section>

        {/* ── SECTION 3: ARGO VALIDATION ──────────────────────────────── */}
        <section id="argo-validation" className="space-y-6">
          <div className="space-y-2">
            <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest">
              03 / IN-SITU EMPIRICAL BENCHMARK
            </div>
            <h2 className="text-3xl sm:text-4xl font-light text-white tracking-tight">
              Argo Float Ground-Truth Validation
            </h2>
            <p className="text-neutral-400 text-sm font-sans max-w-3xl leading-relaxed">
              Scatter distribution of predicted temperature against autonomous profiling Argo floats
              from the CORA v1.3 delayed-mode QC-passed archive. No Argo float data was used during training.
            </p>
          </div>

          <ArgoScatterChart profileIds={argoProfiles.map((p) => p.profile_id)} />
        </section>

        {/* ── SECTION 4: REPRESENTATIVE PROFILES ──────────────────────── */}
        <section id="representative-profiles" className="space-y-6">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div className="space-y-2">
              <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest">
                04 / FLOAT PROFILE SOUNDING
              </div>
              <h2 className="text-3xl sm:text-4xl font-light text-white tracking-tight">
                Representative Vertical Profiles
              </h2>
            </div>

            <div className="flex items-center gap-3 font-mono text-xs">
              <span className="text-neutral-500">Select Float:</span>
              <select
                value={selectedProfile}
                onChange={(e) => setSelectedProfile(e.target.value)}
                className="bg-[#0a0a0a] border border-white/10 rounded-none px-3 py-1.5 text-neutral-200 text-xs font-mono outline-none focus:border-white"
              >
                {argoProfiles.map((p) => (
                  <option key={p.profile_id} value={p.profile_id}>
                    {p.profile_id} — {p.date?.slice(0, 10)} ({p.lat?.toFixed(1)}°N, {p.lon?.toFixed(1)}°E)
                  </option>
                ))}
              </select>
            </div>
          </div>

          {loadingObs ? (
            <div className="w-full h-80 flex items-center justify-center font-mono text-xs text-neutral-500">
              <div className="w-5 h-5 border-2 border-neutral-700 border-t-white rounded-full animate-spin" />
            </div>
          ) : argoObs.length > 0 ? (
            <ArgoProfileChart observations={argoObs} />
          ) : (
            <div className="p-12 text-center text-neutral-500 font-mono text-xs">
              No observation records found for the selected float.
            </div>
          )}
        </section>

        {/* ── SECTION 5: TRAINING DYNAMICS ────────────────────────────── */}
        <section id="training-dynamics" className="space-y-6">
          <div className="space-y-2">
            <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest">
              05 / OPTIMIZATION CONVERGENCE
            </div>
            <h2 className="text-3xl sm:text-4xl font-light text-white tracking-tight">
              Training & Validation Loss Dynamics
            </h2>
            <p className="text-neutral-400 text-sm font-sans max-w-3xl leading-relaxed">
              Convergence curves evaluated per epoch using masked mean squared error over ocean grid cells.
              Fourier spectral layers maintain continuous spatial gradient propagation.
            </p>
          </div>

          <TrainingHistoryChart histories={histories} />
        </section>
      </div>
    </div>
  );
}
