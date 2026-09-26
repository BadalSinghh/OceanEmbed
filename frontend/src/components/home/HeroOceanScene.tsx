"use client";

import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

interface HeroOceanSceneProps {
  className?: string;
}

export default function HeroOceanScene({ className = "" }: HeroOceanSceneProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [activeDepth, setActiveDepth] = useState<number>(100);
  const [activeTemp, setActiveTemp] = useState<number>(21.0);
  const [activeLayerName, setActiveLayerName] = useState<string>("Thermocline Layer");

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let width = container.clientWidth || 600;
    let height = container.clientHeight || 600;

    // ── Three.js Scene Setup ──────────────────────────────────────────
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x050505, 0.032);

    const camera = new THREE.PerspectiveCamera(38, width / height, 0.1, 100);
    // Elevated angle looking at the 3D water column
    camera.position.set(7.2, 4.4, 8.8);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: "high-performance",
    });
    renderer.setSize(width, height, false);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x050505, 0);
    renderer.domElement.style.display = "block";
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.enablePan = false;
    controls.minDistance = 4.5;
    controls.maxDistance = 18;
    controls.target.set(0, -0.4, 0);
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.35;

    const oceanGroup = new THREE.Group();
    scene.add(oceanGroup);

    // ── Physical Depth Levels (OceanEmbed Bay of Bengal Stratification) ──
    const depthLevels = [
      { depth: 0, y: 1.5, temp: 28.8, name: "Surface Mixed Layer (SST)" },
      { depth: 5, y: 1.38, temp: 28.7, name: "Near-Surface" },
      { depth: 10, y: 1.26, temp: 28.5, name: "Upper Mixed Layer" },
      { depth: 20, y: 1.1, temp: 28.2, name: "Mixed Layer Interior" },
      { depth: 30, y: 0.92, temp: 27.8, name: "Mixed Layer Base" },
      { depth: 50, y: 0.65, temp: 26.0, name: "Barrier Layer / Upper Pycnocline" },
      { depth: 75, y: 0.32, temp: 23.4, name: "Upper Thermocline" },
      { depth: 100, y: -0.05, temp: 21.0, name: "Thermocline Core (Max Gradient)" },
      { depth: 125, y: -0.4, temp: 18.5, name: "Main Thermocline" },
      { depth: 150, y: -0.75, temp: 16.2, name: "Lower Thermocline" },
      { depth: 200, y: -1.15, temp: 13.8, name: "Thermocline Base" },
      { depth: 300, y: -1.6, temp: 11.2, name: "Sub-Thermocline Water" },
      { depth: 500, y: -2.1, temp: 8.5, name: "Indian Ocean Intermediate Water" },
      { depth: 700, y: -2.55, temp: 6.8, name: "Deep Water Boundary" },
      { depth: 1000, y: -2.95, temp: 5.4, name: "Abyssal Layer Horizon" },
    ];

    // Temperature to scientific color function (cmocean thermal scientific palette)
    // 29°C -> warm golden amber, 22°C -> gold-teal, 15°C -> cyan-teal, 5°C -> deep navy
    function tempToColor(t: number): THREE.Color {
      const norm = Math.max(0, Math.min(1, (t - 4) / 26));
      const col = new THREE.Color();
      if (norm > 0.75) {
        // 23.5 - 30 °C: solar heated warm amber / coral
        const s = (norm - 0.75) / 0.25;
        col.setRGB(0.94 + 0.06 * s, 0.48 + 0.35 * s, 0.12 + 0.4 * s);
      } else if (norm > 0.45) {
        // 15.7 - 23.5 °C: teal-cyan to warm gold
        const s = (norm - 0.45) / 0.3;
        col.setRGB(0.08 + 0.8 * s, 0.68 - 0.12 * s, 0.62 - 0.45 * s);
      } else if (norm > 0.15) {
        // 7.9 - 15.7 °C: deep cyan to teal
        const s = (norm - 0.15) / 0.3;
        col.setRGB(0.02 + 0.12 * s, 0.28 + 0.4 * s, 0.65 + 0.1 * s);
      } else {
        // 4 - 7.9 °C: deep cobalt navy
        const s = norm / 0.15;
        col.setRGB(0.01 + 0.04 * s, 0.08 + 0.16 * s, 0.32 + 0.32 * s);
      }
      return col;
    }

    // ── Build Volumetric Water Column ──
    const boxWidth = 5.4; // Longitude (80°E to 100°E)
    const boxDepth = 4.6; // Latitude (5°N to 22°N)

    // 1. Horizontal Layer Outlines & Translucent Bounding Planes
    depthLevels.forEach((lvl, idx) => {
      const edges = new THREE.EdgesGeometry(new THREE.PlaneGeometry(boxWidth, boxDepth));
      const isKeyDepth = [0, 50, 100, 200, 500, 1000].includes(lvl.depth);
      const wireMat = new THREE.LineBasicMaterial({
        color: tempToColor(lvl.temp),
        transparent: true,
        opacity: lvl.depth === 0 ? 0.7 : isKeyDepth ? 0.3 : 0.12,
      });
      const wire = new THREE.LineSegments(edges, wireMat);
      wire.rotation.x = -Math.PI / 2;
      wire.position.y = lvl.y;
      oceanGroup.add(wire);

      // Subtle surface and thermocline translucent skin
      if (lvl.depth === 0 || lvl.depth === 100 || lvl.depth === 1000) {
        const planeGeo = new THREE.PlaneGeometry(boxWidth, boxDepth);
        const planeMat = new THREE.MeshBasicMaterial({
          color: tempToColor(lvl.temp),
          transparent: true,
          opacity: lvl.depth === 0 ? 0.08 : 0.025,
          side: THREE.DoubleSide,
        });
        const plane = new THREE.Mesh(planeGeo, planeMat);
        plane.rotation.x = -Math.PI / 2;
        plane.position.y = lvl.y;
        oceanGroup.add(plane);
      }
    });

    // 2. Corner Vertical Bounding Columns (0 to 1000m)
    const corners = [
      [-boxWidth / 2, -boxDepth / 2],
      [boxWidth / 2, -boxDepth / 2],
      [boxWidth / 2, boxDepth / 2],
      [-boxWidth / 2, boxDepth / 2],
    ];

    corners.forEach(([cx, cz]) => {
      const railGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(cx, depthLevels[0].y, cz),
        new THREE.Vector3(cx, depthLevels[depthLevels.length - 1].y, cz),
      ]);
      const railMat = new THREE.LineBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.18,
      });
      oceanGroup.add(new THREE.Line(railGeo, railMat));
    });

    // 3. Volumetric 3D Data Particle Lattice (69x81 Subsampled Grid)
    const nx = 22; // Longitude samples
    const nz = 18; // Latitude samples
    const numPoints = nx * nz * depthLevels.length;
    const positions = new Float32Array(numPoints * 3);
    const colors = new Float32Array(numPoints * 3);

    let pIdx = 0;
    depthLevels.forEach((lvl) => {
      for (let i = 0; i < nx; i++) {
        for (let j = 0; j < nz; j++) {
          const x = -boxWidth / 2 + (i / (nx - 1)) * boxWidth;
          const z = -boxDepth / 2 + (j / (nz - 1)) * boxDepth;

          // Realistic Bay of Bengal temperature distribution:
          // Warmer in central/south basin, fresh river runoff cooling in north, eddy meanders
          const lonNorm = i / (nx - 1);
          const latNorm = j / (nz - 1);
          const eddyPerturb =
            Math.sin(lonNorm * Math.PI * 2) * Math.cos(latNorm * Math.PI * 1.5) * 1.1;
          const northCooling = -1.4 * latNorm;
          const localTemp = lvl.temp + eddyPerturb + northCooling;

          const col = tempToColor(localTemp);

          positions[pIdx * 3] = x;
          positions[pIdx * 3 + 1] = lvl.y;
          positions[pIdx * 3 + 2] = z;

          colors[pIdx * 3] = col.r;
          colors[pIdx * 3 + 1] = col.g;
          colors[pIdx * 3 + 2] = col.b;

          pIdx++;
        }
      }
    });

    const pointGeo = new THREE.BufferGeometry();
    pointGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    pointGeo.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const pointMat = new THREE.PointsMaterial({
      size: 0.052,
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
      sizeAttenuation: true,
    });
    const pointCloud = new THREE.Points(pointGeo, pointMat);
    oceanGroup.add(pointCloud);

    // 4. Interactive Depth Scanner Plane
    const scannerGeo = new THREE.PlaneGeometry(boxWidth * 1.04, boxDepth * 1.04);
    const scannerMat = new THREE.MeshBasicMaterial({
      color: 0x06b6d4,
      transparent: true,
      opacity: 0.12,
      side: THREE.DoubleSide,
    });
    const scannerPlane = new THREE.Mesh(scannerGeo, scannerMat);
    scannerPlane.rotation.x = -Math.PI / 2;
    scannerPlane.position.y = depthLevels[7].y; // 100m default
    oceanGroup.add(scannerPlane);

    const scannerBorder = new THREE.LineSegments(
      new THREE.EdgesGeometry(scannerGeo),
      new THREE.LineBasicMaterial({ color: 0x06b6d4, transparent: true, opacity: 0.85 })
    );
    scannerBorder.rotation.x = -Math.PI / 2;
    scannerBorder.position.y = depthLevels[7].y;
    oceanGroup.add(scannerBorder);

    // ── Mouse & Hover Depth Probing ──
    let scannerTargetY = depthLevels[7].y;

    const onPointerMove = (e: MouseEvent) => {
      const rect = container.getBoundingClientRect();
      const mouseYRel = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));

      // Map vertical mouse position across 15 depth tiers
      const depthIdx = Math.floor(mouseYRel * depthLevels.length);
      const selected = depthLevels[Math.min(depthLevels.length - 1, depthIdx)];
      scannerTargetY = selected.y;
      setActiveDepth(selected.depth);
      setActiveTemp(selected.temp);
      setActiveLayerName(selected.name);
    };

    container.addEventListener("mousemove", onPointerMove);

    // ── Resize Observer (Bulletproof Sizing, Prevents Overflows) ──
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width: newW, height: newH } = entry.contentRect;
        if (newW > 0 && newH > 0) {
          camera.aspect = newW / newH;
          camera.updateProjectionMatrix();
          renderer.setSize(newW, newH, false);
        }
      }
    });
    resizeObserver.observe(container);

    // ── Animation Loop ──
    let animId: number;
    let clock = new THREE.Clock();

    const animate = () => {
      animId = requestAnimationFrame(animate);
      const delta = clock.getDelta();
      const time = clock.getElapsedTime();

      // Smooth scanner plane descent/ascent
      scannerPlane.position.y += (scannerTargetY - scannerPlane.position.y) * 0.12;
      scannerBorder.position.y = scannerPlane.position.y;

      // Subtle slow internal oscillation (internal baroclinic wave effect)
      pointCloud.rotation.y = Math.sin(time * 0.25) * 0.015;

      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(animId);
      resizeObserver.disconnect();
      container.removeEventListener("mousemove", onPointerMove);
      controls.dispose();
      pointGeo.dispose();
      pointMat.dispose();
      scannerGeo.dispose();
      scannerMat.dispose();
      renderer.dispose();
      if (renderer.domElement && container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  return (
    <div className={`relative w-full h-full select-none ${className}`}>
      {/* Three.js WebGL Canvas */}
      <div ref={containerRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

      {/* Top Left: Scientific Metadata Telemetry */}
      <div className="absolute top-4 left-4 pointer-events-none flex flex-col gap-1 font-mono text-[11px] bg-black/75 backdrop-blur-md border border-white/10 px-3.5 py-2.5 rounded">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          <span className="font-semibold text-neutral-100 uppercase tracking-wider">
            3D WATER COLUMN RECONSTRUCTION
          </span>
        </div>
        <div className="text-[10px] text-neutral-400">
          Bay of Bengal (5°–22°N, 80°–100°E) · 69 × 81 Mesh · 15 Depths
        </div>
      </div>

      {/* Bottom Left: Active Depth Horizon Telemetry Badge */}
      <div className="absolute bottom-4 left-4 pointer-events-none font-mono text-xs bg-black/80 backdrop-blur-md border border-white/10 px-4 py-3 rounded flex items-center gap-6">
        <div>
          <div className="text-[10px] uppercase tracking-wider text-neutral-500">Active Horizon</div>
          <div className="text-sm font-semibold text-neutral-100">{activeDepth} m depth</div>
        </div>
        <div className="border-l border-white/10 pl-6">
          <div className="text-[10px] uppercase tracking-wider text-neutral-500">Mean Temp</div>
          <div className="text-sm font-semibold text-amber-400">~{activeTemp.toFixed(1)} °C</div>
        </div>
        <div className="hidden sm:block border-l border-white/10 pl-6 text-[11px] text-neutral-400 max-w-xs truncate">
          {activeLayerName}
        </div>
      </div>

      {/* Right: Vertical Depth Scale (0m down to 1000m) */}
      <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none font-mono text-[10px] text-neutral-400 flex flex-col items-end gap-2.5 bg-black/50 backdrop-blur-sm px-2.5 py-3 rounded border border-white/10">
        <span className="text-amber-400 font-semibold">0 m (SST)</span>
        <span className="text-neutral-500">— 50 m</span>
        <span className="text-cyan-400 font-semibold">100 m (Thermocline)</span>
        <span className="text-neutral-500">— 200 m</span>
        <span className="text-neutral-500">— 500 m</span>
        <span className="text-indigo-400 font-semibold">1000 m (Abyss)</span>
      </div>

      {/* Top Right: Subtle Interaction Hint */}
      <div className="absolute top-4 right-4 pointer-events-none font-mono text-[10px] text-neutral-500 bg-black/40 px-2.5 py-1 rounded border border-white/5 uppercase tracking-widest">
        Drag to Orbit · Hover to Slice
      </div>
    </div>
  );
}
