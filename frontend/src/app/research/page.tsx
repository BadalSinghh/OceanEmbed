import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Research Methodology",
  description:
    "Scientific methodology, satellite data provenance, and in-situ validation protocols behind the OceanEmbed subsurface ocean temperature reconstruction project.",
};

export default function ResearchPage() {
  return (
    <div className="w-full min-h-screen bg-[#050505] text-neutral-100 antialiased font-sans pb-36">
      {/* ── Editorial Header ────────────────────────────────────────── */}
      <div className="w-full px-6 md:px-16 lg:px-24 xl:px-32 py-24 rule-b">
        <div className="max-w-4xl space-y-4">
          <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest">
            SCIENTIFIC REPORT & METHODOLOGY
          </div>
          <h1 className="text-4xl sm:text-6xl font-light text-white tracking-tight">
            RESEARCH & PROTOCOL
          </h1>
          <p className="text-lg sm:text-xl text-neutral-300 font-sans leading-relaxed pt-2 max-w-3xl">
            Theoretical foundations, spaceborne observational data sources, and empirical
            validation protocols for deep representation learning in computational oceanography.
          </p>

          <div className="pt-8 border-t border-white/[0.08] flex flex-wrap items-center gap-12 font-mono text-xs text-neutral-400">
            <div>
              <span className="text-neutral-500 uppercase text-[10px] block">Target Domain</span>
              <span className="text-neutral-200 mt-0.5 block">Bay of Bengal (5°–22°N, 80°–100°E)</span>
            </div>
            <div>
              <span className="text-neutral-500 uppercase text-[10px] block">Satellite Constellation</span>
              <span className="text-neutral-200 mt-0.5 block">Copernicus Marine (CMEMS) L4 Products</span>
            </div>
            <div>
              <span className="text-neutral-500 uppercase text-[10px] block">Validation Corpus</span>
              <span className="text-neutral-200 mt-0.5 block">CORA v1.3 Delayed-Mode Argo CTD Profiles</span>
            </div>
          </div>
        </div>
      </div>

      <div className="w-full px-6 md:px-16 lg:px-24 xl:px-32 mt-20 space-y-24">
        {/* Section 1: Problem Formulation */}
        <section className="space-y-4 max-w-4xl">
          <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest">
            01 / PROBLEM FORMULATION
          </div>
          <h2 className="text-2xl sm:text-3xl font-light text-white tracking-tight">
            The Inverse Surface-to-Subsurface Mapping
          </h2>
          <div className="space-y-4 text-neutral-300 text-base leading-relaxed font-sans pt-2">
            <p>
              Spaceborne remote sensing satellites continuously sample the two-dimensional surface
              boundary condition of the world ocean. However, key oceanic processes—such as tropical
              cyclone intensification, heat uptake, internal wave generation, and biological productivity—are
              governed by the three-dimensional vertical structure of the water column.
            </p>
            <p>
              Autonomous profiling floats (e.g. Argo) and moored buoy arrays provide high-precision
              in-situ vertical soundings, but remain fundamentally sparse in space and time.
              OceanEmbed addresses this fundamental observation gap by learning a parameterised
              continuous operator mapping 2D multi-satellite surface fields to the full 3D temperature
              column (0 to 1000 m).
            </p>
          </div>
        </section>

        {/* Section 2: Study Domain */}
        <section className="space-y-4 max-w-4xl">
          <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest">
            02 / STUDY DOMAIN
          </div>
          <h2 className="text-2xl sm:text-3xl font-light text-white tracking-tight">
            The Bay of Bengal Basin
          </h2>
          <div className="space-y-4 text-neutral-300 text-base leading-relaxed font-sans pt-2">
            <p>
              The target geographic domain is bounded by <strong>5.0°N–22.0°N</strong> and{" "}
              <strong>80.0°E–100.0°E</strong> at a uniform <strong>0.25° grid resolution</strong>,
              yielding an active ocean matrix of 69 × 81 cells across 15 standard depth horizons.
            </p>
            <p>
              The Bay of Bengal represents one of the most dynamically challenging ocean basins
              on Earth: massive freshwater runoff from the Ganges-Brahmaputra river systems forms a
              persistent low-salinity surface lens. This creates a strong barrier layer that decouples
              the surface mixed layer from the underlying thermocline, rendering classical linear
              projections ineffective and necessitating nonlinear deep representation learning.
            </p>
          </div>
        </section>

        {/* Section 3: Observational Provenance */}
        <section className="space-y-6">
          <div className="space-y-2">
            <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest">
              03 / OBSERVATIONAL PROVENANCE
            </div>
            <h2 className="text-2xl sm:text-3xl font-light text-white tracking-tight">
              Multi-Satellite Surface Channels (7 Observables)
            </h2>
          </div>

          <div className="w-full overflow-x-auto">
            <table className="editorial-table font-mono text-xs">
              <thead>
                <tr>
                  <th>Channel</th>
                  <th>Observable</th>
                  <th>Unit</th>
                  <th>Dataset Identifier</th>
                  <th>Satellite Sensor Class</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="text-white font-medium">SST</td>
                  <td className="text-neutral-200 font-sans">Sea Surface Temperature</td>
                  <td>°C</td>
                  <td className="text-neutral-400">METOFFICE-GLO-SST-L4-REP-OBS-SST</td>
                  <td className="text-neutral-400 font-sans">Infrared + Microwave radiometry (OSTIA)</td>
                </tr>
                <tr>
                  <td className="text-white font-medium">SSS</td>
                  <td className="text-neutral-200 font-sans">Sea Surface Salinity</td>
                  <td>psu</td>
                  <td className="text-neutral-400">cmems_obs-mob_glo_phy-sss_my_multi_P1D</td>
                  <td className="text-neutral-400 font-sans">SMOS / SMAP L-band microwave</td>
                </tr>
                <tr>
                  <td className="text-white font-medium">SLA</td>
                  <td className="text-neutral-200 font-sans">Sea Level Anomaly</td>
                  <td>m</td>
                  <td className="text-neutral-400">cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs</td>
                  <td className="text-neutral-400 font-sans">Merged multi-mission radar altimetry</td>
                </tr>
                <tr>
                  <td className="text-white font-medium">Current U</td>
                  <td className="text-neutral-200 font-sans">Zonal Surface Current</td>
                  <td>m/s</td>
                  <td className="text-neutral-400">DUACS Geostrophic Component</td>
                  <td className="text-neutral-400 font-sans">Derived via geostrophic balance from SLA</td>
                </tr>
                <tr>
                  <td className="text-white font-medium">Current V</td>
                  <td className="text-neutral-200 font-sans">Meridional Surface Current</td>
                  <td>m/s</td>
                  <td className="text-neutral-400">DUACS Geostrophic Component</td>
                  <td className="text-neutral-400 font-sans">Derived via geostrophic balance from SLA</td>
                </tr>
                <tr>
                  <td className="text-white font-medium">Wind U</td>
                  <td className="text-neutral-200 font-sans">Zonal 10m Wind Stress</td>
                  <td>m/s</td>
                  <td className="text-neutral-400">cmems_obs-wind_glo_phy_my_l4</td>
                  <td className="text-neutral-400 font-sans">MetOp ASCAT scatterometer constellation</td>
                </tr>
                <tr>
                  <td className="text-white font-medium">Wind V</td>
                  <td className="text-neutral-200 font-sans">Meridional 10m Wind Stress</td>
                  <td>m/s</td>
                  <td className="text-neutral-400">cmems_obs-wind_glo_phy_my_l4</td>
                  <td className="text-neutral-400 font-sans">MetOp ASCAT scatterometer constellation</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Section 4: Ground-Truth Protocol */}
        <section className="space-y-4 max-w-4xl">
          <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest">
            04 / GROUND-TRUTH PROTOCOL
          </div>
          <h2 className="text-2xl sm:text-3xl font-light text-white tracking-tight">
            Independent In-Situ Argo Float Validation
          </h2>
          <div className="space-y-4 text-neutral-300 text-base leading-relaxed font-sans pt-2">
            <p>
              To ensure rigorous validation free of circular data leakage, OceanEmbed evaluates
              model performance against two strictly segregated references:
            </p>
            <ol className="list-decimal list-inside space-y-2 text-neutral-300">
              <li>
                <strong>Held-Out GLORYS12 Reanalysis Split:</strong> 109 consecutive test dates
                (2023-09-14 to 2023-12-31) completely unseen during model training.
              </li>
              <li>
                <strong>CORA v1.3 Argo In-Situ CTD Observations:</strong> 253 independent autonomous
                profiling floats yielding 3,509 depth-matched in-situ soundings across the basin.
                Argo data was never provided during model optimization or hyperparameter tuning.
              </li>
            </ol>
          </div>
        </section>

        {/* CTA */}
        <div className="pt-12 rule-t flex flex-wrap items-center justify-between gap-6">
          <div>
            <div className="text-xl font-light text-white">Experience the reconstruction workstation</div>
            <div className="text-sm text-neutral-400 font-sans mt-1">
              Test live model inference on any observation date in the held-out split.
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
