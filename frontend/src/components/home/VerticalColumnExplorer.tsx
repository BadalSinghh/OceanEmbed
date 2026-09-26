"use client";

import React, { useState, useEffect, useRef } from "react";
import * as THREE from "three";
import Link from "next/link";

interface DepthLevelInfo {
  depth: number;
  temp: number;
  rmse: number;
  layer: string;
  zone: string;
  description: string;
}

const DEPTH_LEVELS: DepthLevelInfo[] = [
  { depth: 0, temp: 28.6, rmse: 0.81, layer: "Surface Skin", zone: "Epipelagic", description: "Direct satellite observational horizon. SST from OSTIA, SLA from DUACS." },
  { depth: 5, temp: 28.5, rmse: 0.82, layer: "Near Surface", zone: "Epipelagic", description: "Wind-stirred layer coupled directly to surface shear." },
  { depth: 10, temp: 28.4, rmse: 0.84, layer: "Upper Mixed Layer", zone: "Epipelagic", description: "Isothermal upper column in post-monsoon Bay of Bengal." },
  { depth: 20, temp: 28.2, rmse: 0.88, layer: "Mixed Layer Core", zone: "Epipelagic", description: "Freshwater lens dynamics driven by river runoff." },
  { depth: 30, temp: 27.8, rmse: 0.94, layer: "Mixed Layer Base", zone: "Epipelagic", description: "Transition zone where wind mixing ceases." },
  { depth: 50, temp: 25.4, rmse: 1.12, layer: "Upper Thermocline", zone: "Thermocline", description: "Onset of steep vertical temperature gradient." },
  { depth: 75, temp: 22.1, rmse: 1.34, layer: "Sharp Thermocline", zone: "Thermocline", description: "High vertical shear. Temperature plunges >3°C over 25m." },
  { depth: 100, temp: 19.4, rmse: 1.52, layer: "Core Thermocline", zone: "Thermocline", description: "Peak stratification and maximum baroclinic heaving." },
  { depth: 125, temp: 17.1, rmse: 1.58, layer: "Mid Thermocline", zone: "Thermocline", description: "Subsurface counter-currents and mesoscale eddy displacements." },
  { depth: 150, temp: 15.2, rmse: 1.48, layer: "Lower Thermocline", zone: "Thermocline", description: "Weakening vertical thermal gradient." },
  { depth: 200, temp: 13.1, rmse: 1.32, layer: "Thermocline Base", zone: "Thermocline", description: "Transition from steep thermocline to intermediate ocean." },
  { depth: 300, temp: 10.8, rmse: 1.05, layer: "Upper Intermediate", zone: "Mesopelagic", description: "Aphotic water mass below direct wind or solar influence." },
  { depth: 500, temp: 8.2, rmse: 0.76, layer: "Mid Intermediate", zone: "Mesopelagic", description: "Intrusion depths of Arabian Sea High Salinity Water." },
  { depth: 700, temp: 6.5, rmse: 0.54, layer: "Deep Intermediate", zone: "Mesopelagic", description: "Highly stable thermal structure with diminished temporal variance." },
  { depth: 1000, temp: 5.3, rmse: 0.38, layer: "Abyssal Horizon", zone: "Bathypelagic", description: "Cold, dense bottom-flowing water mass. Lowest model error." },
];

export default function VerticalColumnExplorer() {
  const [selectedIdx, setSelectedIdx] = useState<number>(7); // 100m default
  const active = DEPTH_LEVELS[selectedIdx];

  const canvasContainerRef = useRef<HTMLDivElement>(null);

  // Mini 3D Three.js column visualizer for the water column
  useEffect(() => {
    const container = canvasContainerRef.current;
    if (!container) return;

    let width = container.clientWidth;
    let height = container.clientHeight;

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x050505, 0.05);

    const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 100);
    camera.position.set(4.5, 0, 5.5);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height, false);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const colGroup = new THREE.Group();
    scene.add(colGroup);

    // Build 15 horizontal depth disks
    const diskMeshes: THREE.Mesh[] = [];
    DEPTH_LEVELS.forEach((lvl, i) => {
      // Map 0 to 1000m to Y range 2.0 to -2.0
      const y = 2.0 - (i / (DEPTH_LEVELS.length - 1)) * 4.0;
      const radius = 1.4 - (i / DEPTH_LEVELS.length) * 0.3;
      const diskGeo = new THREE.RingGeometry(0.1, radius, 32);

      // Color from warm amber (28°C) to cold cyan/blue (5°C)
      const norm = Math.max(0, Math.min(1, (lvl.temp - 4) / 25));
      const col = new THREE.Color();
      if (norm > 0.6) {
        col.setRGB(0.9, 0.4 + norm * 0.3, 0.1);
      } else if (norm > 0.3) {
        col.setRGB(0.1 + norm * 0.5, 0.7, 0.6);
      } else {
        col.setRGB(0.05, 0.2 + norm * 0.4, 0.7);
      }

      const diskMat = new THREE.MeshBasicMaterial({
        color: col,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: i === selectedIdx ? 0.9 : 0.25,
      });

      const mesh = new THREE.Mesh(diskGeo, diskMat);
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.y = y;
      colGroup.add(mesh);
      diskMeshes.push(mesh);
    });

    // Central core line
    const lineGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 2.2, 0),
      new THREE.Vector3(0, -2.2, 0),
    ]);
    const lineMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.15 });
    colGroup.add(new THREE.Line(lineGeo, lineMat));

    let animId: number;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      colGroup.rotation.y += 0.005;
      renderer.render(scene, camera);
    };
    animate();

    const onResize = () => {
      if (!container) return;
      width = container.clientWidth;
      height = container.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };
    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", onResize);
      lineGeo.dispose();
      lineMat.dispose();
      renderer.dispose();
      if (renderer.domElement && container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [selectedIdx]);

  return (
    <div className="w-full">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-start">
        {/* Left: 15-Depth Interactive Scale (Open Editorial List) */}
        <div className="lg:col-span-6 space-y-2">
          <div className="flex items-center justify-between pb-3 rule-b font-mono text-xs text-neutral-400">
            <span className="uppercase tracking-wider">Depth Horizon (m)</span>
            <span className="uppercase tracking-wider">Thermal State & Error</span>
          </div>

          <div className="divide-y divide-white/[0.06]">
            {DEPTH_LEVELS.map((item, idx) => {
              const isSelected = idx === selectedIdx;
              return (
                <button
                  key={item.depth}
                  onClick={() => setSelectedIdx(idx)}
                  className={`w-full py-2.5 px-3 flex items-center justify-between text-left transition-colors font-mono cursor-pointer ${
                    isSelected ? "bg-white/[0.06] text-white" : "text-neutral-400 hover:text-neutral-200"
                  }`}
                >
                  <div className="flex items-center gap-4">
                    <span
                      className={`text-sm font-semibold w-16 ${
                        isSelected ? "text-cyan-400" : "text-neutral-200"
                      }`}
                    >
                      {item.depth.toString().padStart(4, " ")} m
                    </span>
                    <span className="text-xs text-neutral-400 font-sans">{item.layer}</span>
                  </div>

                  <div className="flex items-center gap-6 text-xs">
                    <span className="text-neutral-300 font-medium">{item.temp.toFixed(1)} °C</span>
                    <span className="text-neutral-500 w-16 text-right">±{item.rmse.toFixed(2)} °C</span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Right: 3D Water Column Visualizer & Layer Insight */}
        <div className="lg:col-span-6 space-y-8">
          {/* Mini 3D Column Canvas */}
          <div className="w-full h-80 relative select-none">
            <div ref={canvasContainerRef} className="w-full h-full" />
            <div className="absolute top-2 left-2 font-mono text-[11px] text-neutral-500 uppercase">
              Vertical Thermal Structure (0 to 1000 m)
            </div>
          </div>

          {/* Active Layer Deep Insight */}
          <div className="rule-t pt-6 space-y-4">
            <div className="flex items-baseline justify-between font-mono">
              <span className="text-xs uppercase text-cyan-400 tracking-wider font-semibold">
                {active.zone} · {active.layer}
              </span>
              <span className="text-xs text-neutral-500">
                OceanEmbed RMSE: ±{active.rmse.toFixed(2)} °C
              </span>
            </div>

            <div className="text-3xl sm:text-4xl font-medium text-white font-sans">
              {active.depth} meters <span className="text-xl text-neutral-400">({active.temp.toFixed(1)} °C)</span>
            </div>

            <p className="text-neutral-300 text-sm leading-relaxed font-sans">
              {active.description} In the central Bay of Bengal, the thermocline (50–200 m) creates a
              strong density barrier separating the solar-heated freshwater skin from the cold intermediate
              basin water.
            </p>

            <div className="pt-2">
              <Link
                href={`/demo?depth=${active.depth}`}
                className="font-mono text-xs text-neutral-300 hover:text-white flex items-center gap-2"
              >
                <span>Inspect in Reconstruction Lab</span>
                <span>→</span>
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
