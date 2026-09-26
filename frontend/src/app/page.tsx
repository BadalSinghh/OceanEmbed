import type { Metadata } from "next";
import Link from "next/link";
import CinematicOpening from "@/components/common/CinematicOpening";
import HeroOceanScene from "@/components/home/HeroOceanScene";
import SurfaceChannelsDiagram from "@/components/home/SurfaceChannelsDiagram";
import VerticalColumnExplorer from "@/components/home/VerticalColumnExplorer";
import ResearchPipelineDiagram from "@/components/home/ResearchPipelineDiagram";
import HomeResultsPreview from "@/components/home/HomeResultsPreview";

export const metadata: Metadata = {
  title: "OceanEmbed — Reconstructing the Ocean Beneath the Surface",
  description:
    "A satellite-embedding deep learning framework for reconstructing the 3D vertical temperature structure of the ocean from multi-satellite surface observations over the Bay of Bengal.",
};

export default function HomePage() {
  return (
    <div className="w-full min-h-screen bg-[#050505] text-neutral-100 antialiased selection:bg-neutral-800 selection:text-white">
      {/* ── Cinematic Opening Sequence (1-2s pure black title reveal) ── */}
      <CinematicOpening />

      {/* ── HERO VIEWPORT: Full 100svh Screen with 3D Ocean Depth Volume ─ */}
      <section className="relative w-full min-h-[100svh] flex items-center overflow-hidden rule-b">
        {/* Subtle oceanographic coordinate atmosphere & soft radial glow in background */}
        <div className="absolute right-0 top-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-[radial-gradient(ellipse_at_center,rgba(6,182,212,0.06)_0%,rgba(5,5,5,0)_75%)] pointer-events-none z-0" />
        <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(to_bottom,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:64px_64px] pointer-events-none z-0 opacity-40" />

        {/* 2-Column Responsive Editorial Composition */}
        <div className="relative z-10 w-full px-6 md:px-12 lg:px-16 xl:px-24 py-16 grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-8 items-center min-h-[100svh]">
          {/* Left Column: Typographic Editorial Hero */}
          <div className="lg:col-span-6 xl:col-span-5 space-y-8 pointer-events-auto">
            {/* Technical Eyebrow with Restrained Cyan Indicator */}
            <div className="flex items-center gap-2.5 font-mono text-xs text-neutral-400 tracking-widest uppercase">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
              <span>OCEANEMBED / BAY OF BENGAL / 5°–22°N, 80°–100°E</span>
            </div>

            {/* Display Headline (72–96px desktop) */}
            <h1 className="text-4xl sm:text-6xl md:text-7xl lg:text-[76px] font-light text-white tracking-tight leading-[1.04]">
              Reconstructing <br />
              the ocean <br />
              beneath the surface.
            </h1>

            {/* Scientific Subheading */}
            <p className="text-base sm:text-lg text-neutral-300 font-sans leading-relaxed max-w-xl font-normal">
              A satellite-embedding deep learning framework for reconstructing the 3D vertical
              temperature structure of the ocean from multi-satellite surface observations.
            </p>

            {/* Interactive Primary Actions */}
            <div className="pt-2 flex flex-wrap items-center gap-5">
              <Link href="/demo" className="btn-research-primary">
                <span>Explore the Reconstruction</span>
                <span>→</span>
              </Link>
              <Link href="/research" className="btn-research-secondary">
                <span>Read the Research</span>
              </Link>
            </div>

            {/* Domain Telemetry Footnote */}
            <div className="pt-8 border-t border-white/[0.08] grid grid-cols-3 gap-6 font-mono text-xs text-neutral-400">
              <div>
                <span className="text-[10px] text-neutral-500 uppercase block">Domain Mesh</span>
                <span className="text-neutral-200 mt-1 block">69 × 81 at 0.25°</span>
              </div>
              <div>
                <span className="text-[10px] text-neutral-500 uppercase block">Water Column</span>
                <span className="text-neutral-200 mt-1 block">0–1000 m (15 lvls)</span>
              </div>
              <div>
                <span className="text-[10px] text-neutral-500 uppercase block">Validation</span>
                <span className="text-neutral-200 mt-1 block">253 Argo Floats</span>
              </div>
            </div>
          </div>

          {/* Right Column: Dominant 3D Ocean Depth Volume */}
          <div className="lg:col-span-6 xl:col-span-7 w-full h-[540px] sm:h-[620px] lg:h-[700px] xl:h-[760px] relative flex items-center justify-center">
            <HeroOceanScene className="w-full h-full" />
          </div>
        </div>
      </section>

      {/* ── SECTION 1: The Surface is Only the Beginning ──────────────── */}
      <section className="w-full py-28 px-6 md:px-12 lg:px-16 xl:px-24 rule-b">
        <div className="max-w-4xl mb-16 space-y-4">
          <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest flex items-center gap-2">
            <span className="text-cyan-400 font-semibold">01</span>
            <span>/ OBSERVATIONAL SURFACE BOUNDARY</span>
          </div>
          <h2 className="text-3xl sm:text-5xl font-light text-white tracking-tight">
            Observational surface boundary.
          </h2>
          <p className="text-neutral-300 text-base sm:text-lg leading-relaxed font-sans pt-2">
            Spaceborne remote sensing satellites observe the continuous two-dimensional boundary
            skin of the ocean. Seven daily physical channels—sea surface temperature, sea surface salinity,
            sea level anomaly, geostrophic current vectors, and scatterometer wind stress—carry the
            dynamical imprint of subsurface circulation.
          </p>
        </div>

        {/* 7-Channel Horizontal Composition (Open Layout, No Cards) */}
        <SurfaceChannelsDiagram />
      </section>

      {/* ── SECTION 2: The Hidden Ocean (Full-Screen Water Column) ────── */}
      <section className="w-full py-28 px-6 md:px-12 lg:px-16 xl:px-24 rule-b bg-[#080808]">
        <div className="max-w-4xl mb-16 space-y-4">
          <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest flex items-center gap-2">
            <span className="text-cyan-400 font-semibold">02</span>
            <span>/ VERTICAL WATER COLUMN</span>
          </div>
          <h2 className="text-3xl sm:text-5xl font-light text-white tracking-tight">
            The ocean beneath.
          </h2>
          <p className="text-neutral-300 text-base sm:text-lg leading-relaxed font-sans pt-2">
            Beneath the surface mixed layer lies the sharp, steep thermocline—a barrier layer
            where temperature plunges over 14 °C in less than 150 meters. Resolving this vertical
            stratification across all 15 physical depth horizons from 0 to 1000 meters is vital for
            quantifying ocean heat content and baroclinic transport.
          </p>
        </div>

        {/* Vertical Column Explorer */}
        <VerticalColumnExplorer />
      </section>

      {/* ── SECTION 3: The Research Pipeline ──────────────────────────── */}
      <section className="w-full py-28 px-6 md:px-12 lg:px-16 xl:px-24 rule-b">
        <div className="max-w-4xl mb-16 space-y-4">
          <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest flex items-center gap-2">
            <span className="text-cyan-400 font-semibold">03</span>
            <span>/ METHODOLOGY & PIPELINE</span>
          </div>
          <h2 className="text-3xl sm:text-5xl font-light text-white tracking-tight">
            The neural reconstruction pipeline.
          </h2>
          <p className="text-neutral-300 text-base sm:text-lg leading-relaxed font-sans pt-2">
            From multi-satellite surface observables to continuous 3D vertical fields:
            OceanEmbed couples multiscale convolutional feature extraction with Fourier
            Neural Operator spectral integral convolutions to solve the ill-posed inverse problem.
          </p>
        </div>

        {/* Open Research Pipeline Diagram */}
        <ResearchPipelineDiagram />
      </section>

      {/* ── SECTION 4: Results Preview ────────────────────────────────── */}
      <section className="w-full py-28 px-6 md:px-12 lg:px-16 xl:px-24 rule-b bg-[#080808]">
        <div className="max-w-4xl mb-16 space-y-4">
          <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest flex items-center gap-2">
            <span className="text-cyan-400 font-semibold">04</span>
            <span>/ EMPIRICAL BENCHMARKS</span>
          </div>
          <h2 className="text-3xl sm:text-5xl font-light text-white tracking-tight">
            Quantitative benchmark results.
          </h2>
          <p className="text-neutral-300 text-base sm:text-lg leading-relaxed font-sans pt-2">
            Evaluated on 109 held-out test dates against GLORYS12 reanalysis and validated
            against 253 independent in-situ Argo profiling floats (3,509 matched soundings)
            across the Bay of Bengal basin.
          </p>
        </div>

        {/* Large Plotly Results Preview */}
        <HomeResultsPreview />
      </section>

      {/* ── SECTION 5: Workstation CTA ────────────────────────────────── */}
      <section className="w-full py-32 px-6 md:px-12 lg:px-16 xl:px-24 rule-b text-center relative overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(6,182,212,0.04)_0%,rgba(5,5,5,0)_70%)] pointer-events-none" />
        <div className="relative z-10 max-w-3xl mx-auto space-y-6">
          <div className="font-mono text-xs text-neutral-500 uppercase tracking-widest">
            COMPUTATIONAL WORKSTATION
          </div>
          <h2 className="text-4xl sm:text-6xl font-light text-white tracking-tight">
            See beneath the surface.
          </h2>
          <p className="text-neutral-300 text-base sm:text-lg leading-relaxed font-sans max-w-xl mx-auto">
            Run the reconstruction framework across any test date. Inspect the continuous
            3D temperature volume, slice through depth horizons, and evaluate sounding profiles.
          </p>
          <div className="pt-6">
            <Link href="/demo" className="btn-research-primary px-8 py-3.5 text-base">
              <span>Open Interactive Reconstruction</span>
              <span>→</span>
            </Link>
          </div>
        </div>
      </section>

      {/* ── SECTION 6: Colophon Footer ────────────────────────────────── */}
      <footer className="w-full py-16 px-6 md:px-12 lg:px-16 xl:px-24 text-xs font-mono text-neutral-500">
        <div className="w-full flex flex-col md:flex-row items-start md:items-center justify-between gap-8">
          <div>
            <div className="text-neutral-200 font-semibold text-sm">OCEANEMBED</div>
            <div className="text-neutral-500 mt-1 font-sans text-xs">
              Satellite Embedding-Based Deep Learning Framework for Subsurface Ocean Temperature Reconstruction.
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-8 text-neutral-400">
            <Link href="/" className="hover:text-white transition-colors">Home</Link>
            <Link href="/demo" className="hover:text-white transition-colors">Explore</Link>
            <Link href="/results" className="hover:text-white transition-colors">Results</Link>
            <Link href="/architecture" className="hover:text-white transition-colors">Architecture</Link>
            <Link href="/research" className="hover:text-white transition-colors">Research</Link>
          </div>

          <div className="text-neutral-600 text-[11px]">
            PyTorch · FNO2D · CBAM · CMEMS L4 · CORA v1.3 Argo
          </div>
        </div>
      </footer>
    </div>
  );
}
