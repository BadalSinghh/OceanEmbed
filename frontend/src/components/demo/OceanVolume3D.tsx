"use client";

import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

interface OceanVolume3DProps {
  prediction: (number | null)[][][];      // [15 depths, 69 lats, 81 lons]
  groundTruth?: (number | null)[][][];
  error?: (number | null)[][][];
  depths: number[];                        // 15 depths [0, 5, 10, ... 1000]
  lats: number[];                          // 69 lats [5.0 ... 22.0]
  lons: number[];                          // 81 lons [80.0 ... 100.0]
  fieldMode?: "prediction" | "groundTruth" | "error";
  activeDepthIndex?: number;
  onDepthSelect?: (depthIdx: number) => void;
  onProbePoint?: (lat: number, lon: number, depth: number, value: number) => void;
}

export default function OceanVolume3D({
  prediction,
  groundTruth,
  error,
  depths,
  lats,
  lons,
  fieldMode = "prediction",
  activeDepthIndex = 7, // 100m default
  onDepthSelect,
  onProbePoint,
}: OceanVolume3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredData, setHoveredData] = useState<{
    lat: number;
    lon: number;
    depth: number;
    value: number;
  } | null>(null);

  // References to keep in sync without full rebuilds
  const activeDepthRef = useRef(activeDepthIndex);
  activeDepthRef.current = activeDepthIndex;

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let width = container.clientWidth;
    let height = container.clientHeight;

    // ── Three.js Scene Setup ──────────────────────────────────────────
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x07080a, 0.035);

    const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 100);
    camera.position.set(7.8, 5.2, 9.2);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: "high-performance",
    });
    renderer.setSize(width, height, false);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x07080a, 0);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 22;
    controls.minDistance = 3;
    controls.target.set(0, -0.6, 0);

    const volumeGroup = new THREE.Group();
    scene.add(volumeGroup);

    // ── Dimensions & Scaling ──────────────────────────────────────────
    const boxW = 5.6; // Longitude (80°E to 100°E)
    const boxD = 4.8; // Latitude (5°N to 22°N)
    const totalY = 3.6; // Depth (0 to 1000m)

    // Select active 3D tensor based on fieldMode
    let activeTensor = prediction;
    if (fieldMode === "groundTruth" && groundTruth) activeTensor = groundTruth;
    if (fieldMode === "error" && error) activeTensor = error;

    // Depth Y position mapping (nonlinear to visually expand thermocline)
    const getYForDepth = (d: number) => {
      // Map 0 -> 1.5, 100 -> 0.4, 200 -> -0.4, 500 -> -1.4, 1000 -> -2.1
      const logNorm = Math.log10(d + 10) / Math.log10(1010); // 0 to 1
      return 1.4 - logNorm * totalY;
    };

    // Scientific Color Mapping
    const getColorForVal = (val: number | null): THREE.Color => {
      if (val === null || isNaN(val)) return new THREE.Color(0x181c24); // land/nan

      if (fieldMode === "error") {
        // Divergent RdBu / Cyan-Amber for error (-2 to +2 °C)
        const norm = Math.max(-1, Math.min(1, val / 2.0));
        const col = new THREE.Color();
        if (norm < 0) {
          // Negative error (underestimation) - cyan/blue
          col.setRGB(0.1, 0.4 + 0.5 * (1 + norm), 0.9);
        } else {
          // Positive error (overestimation) - amber/orange
          col.setRGB(0.9, 0.5 * (1 - norm), 0.1);
        }
        return col;
      }

      // cmocean thermal color scale for temperature (4°C to 30°C)
      const tNorm = Math.max(0, Math.min(1, (val - 4) / 26));
      const col = new THREE.Color();
      if (tNorm > 0.75) {
        // 23.5 - 30 °C: amber to golden white
        const s = (tNorm - 0.75) / 0.25;
        col.setRGB(0.92 + 0.08 * s, 0.55 + 0.4 * s, 0.15 + 0.4 * s);
      } else if (tNorm > 0.45) {
        // 15.7 - 23.5 °C: teal-cyan to amber
        const s = (tNorm - 0.45) / 0.3;
        col.setRGB(0.1 + 0.8 * s, 0.7 - 0.15 * s, 0.65 - 0.45 * s);
      } else if (tNorm > 0.15) {
        // 8 - 15.7 °C: deep cyan to teal
        const s = (tNorm - 0.15) / 0.3;
        col.setRGB(0.02 + 0.1 * s, 0.3 + 0.4 * s, 0.65 + 0.05 * s);
      } else {
        // 4 - 8 °C: deep cobalt navy
        const s = tNorm / 0.15;
        col.setRGB(0.01 + 0.04 * s, 0.08 + 0.15 * s, 0.3 + 0.35 * s);
      }
      return col;
    };

    // ── Build Depth Planes & 3D Sample Lattice ────────────────────────
    const numDepths = depths.length;
    const nLat = lats.length;
    const nLon = lons.length;

    // Subsample grid for interactive WebGL performance (step 2)
    const latStep = 2;
    const lonStep = 2;
    const sLat = Math.floor(nLat / latStep);
    const sLon = Math.floor(nLon / lonStep);

    // 1. Horizontal Depth Layer Outlines & Slices
    const depthMeshes: THREE.Group[] = [];

    depths.forEach((d, dIdx) => {
      const depthGroup = new THREE.Group();
      const yPos = getYForDepth(d);

      // Wireframe border for this depth level
      const edges = new THREE.EdgesGeometry(new THREE.PlaneGeometry(boxW, boxD));
      const wireMat = new THREE.LineBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: dIdx === activeDepthRef.current ? 0.7 : 0.12,
      });
      const wire = new THREE.LineSegments(edges, wireMat);
      wire.rotation.x = -Math.PI / 2;
      wire.position.y = yPos;
      depthGroup.add(wire);

      // Sample points on this depth layer
      if (activeTensor && activeTensor[dIdx]) {
        const tensorSlice = activeTensor[dIdx];
        const numPts = sLat * sLon;
        const positions = new Float32Array(numPts * 3);
        const colors = new Float32Array(numPts * 3);

        let pIdx = 0;
        for (let i = 0; i < nLat; i += latStep) {
          for (let j = 0; j < nLon; j += lonStep) {
            const x = -boxW / 2 + (j / (nLon - 1)) * boxW;
            const z = boxD / 2 - (i / (nLat - 1)) * boxD; // Lat inverted for North up
            const val = tensorSlice[i]?.[j] ?? null;
            const col = getColorForVal(val);

            positions[pIdx * 3] = x;
            positions[pIdx * 3 + 1] = yPos;
            positions[pIdx * 3 + 2] = z;

            colors[pIdx * 3] = col.r;
            colors[pIdx * 3 + 1] = col.g;
            colors[pIdx * 3 + 2] = col.b;
            pIdx++;
          }
        }

        const ptGeo = new THREE.BufferGeometry();
        ptGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        ptGeo.setAttribute("color", new THREE.BufferAttribute(colors, 3));

        const ptMat = new THREE.PointsMaterial({
          size: dIdx === activeDepthRef.current ? 0.075 : 0.045,
          vertexColors: true,
          transparent: true,
          opacity: dIdx === activeDepthRef.current ? 0.95 : 0.45,
          sizeAttenuation: true,
        });

        const pts = new THREE.Points(ptGeo, ptMat);
        depthGroup.add(pts);
      }

      volumeGroup.add(depthGroup);
      depthMeshes.push(depthGroup);
    });

    // 2. 3D Bounding Box Rails (4 corners 0 to 1000m)
    const corners = [
      [-boxW / 2, -boxD / 2],
      [boxW / 2, -boxD / 2],
      [boxW / 2, boxD / 2],
      [-boxW / 2, boxD / 2],
    ];
    const topY = getYForDepth(0);
    const bottomY = getYForDepth(1000);

    corners.forEach(([cx, cz]) => {
      const railGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(cx, topY, cz),
        new THREE.Vector3(cx, bottomY, cz),
      ]);
      const railMat = new THREE.LineBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.2,
      });
      volumeGroup.add(new THREE.Line(railGeo, railMat));
    });

    // 3. Active Depth Slice Highlight Plane
    const sliceGeo = new THREE.PlaneGeometry(boxW * 1.04, boxD * 1.04);
    const sliceMat = new THREE.MeshBasicMaterial({
      color: 0x00e5ff,
      transparent: true,
      opacity: 0.1,
      side: THREE.DoubleSide,
    });
    const slicePlane = new THREE.Mesh(sliceGeo, sliceMat);
    slicePlane.rotation.x = -Math.PI / 2;
    slicePlane.position.y = getYForDepth(depths[activeDepthRef.current] || 100);
    volumeGroup.add(slicePlane);

    const sliceBorder = new THREE.LineSegments(
      new THREE.EdgesGeometry(sliceGeo),
      new THREE.LineBasicMaterial({ color: 0x00e5ff, transparent: true, opacity: 0.85 })
    );
    sliceBorder.rotation.x = -Math.PI / 2;
    sliceBorder.position.y = slicePlane.position.y;
    volumeGroup.add(sliceBorder);

    // 4. Raycaster & Pointer Interaction for Coordinate Probing
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const onPointerMove = (e: MouseEvent) => {
      const rect = container.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObject(slicePlane);

      if (intersects.length > 0) {
        const pt = intersects[0].point;
        // Map local point x, z back to lon, lat
        const normLon = Math.max(0, Math.min(1, (pt.x + boxW / 2) / boxW));
        const normLat = Math.max(0, Math.min(1, (boxD / 2 - pt.z) / boxD));

        const lonIdx = Math.floor(normLon * (nLon - 1));
        const latIdx = Math.floor(normLat * (nLat - 1));

        const latVal = lats[latIdx] ?? 13.5;
        const lonVal = lons[lonIdx] ?? 90.0;
        const depthVal = depths[activeDepthRef.current] ?? 100;
        const rawVal = activeTensor?.[activeDepthRef.current]?.[latIdx]?.[lonIdx] ?? null;

        if (rawVal !== null) {
          const info = {
            lat: latVal,
            lon: lonVal,
            depth: depthVal,
            value: rawVal,
          };
          setHoveredData(info);
          if (onProbePoint) {
            onProbePoint(latVal, lonVal, depthVal, rawVal);
          }
        }
      }
    };

    container.addEventListener("mousemove", onPointerMove);

    // Resize
    const onResize = () => {
      if (!container) return;
      width = container.clientWidth;
      height = container.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };
    window.addEventListener("resize", onResize);

    // Animation Loop
    let animId: number;
    const animate = () => {
      animId = requestAnimationFrame(animate);

      // Smooth target Y update for active slice
      const targetY = getYForDepth(depths[activeDepthRef.current] || 100);
      slicePlane.position.y += (targetY - slicePlane.position.y) * 0.12;
      sliceBorder.position.y = slicePlane.position.y;

      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", onResize);
      container.removeEventListener("mousemove", onPointerMove);
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement && container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [prediction, groundTruth, error, fieldMode, depths, lats, lons]);

  return (
    <div className="relative w-full h-[540px] lg:h-[640px] bg-[#07080a] select-none rounded overflow-hidden">
      {/* 3D Canvas */}
      <div ref={containerRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

      {/* Top Left: Active Mode & Model Telemetry */}
      <div className="absolute top-4 left-4 pointer-events-none flex flex-col gap-1 font-mono text-xs bg-black/75 backdrop-blur-md border border-white/10 px-3.5 py-2.5 rounded">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-cyan-400" />
          <span className="font-semibold text-neutral-100 uppercase tracking-wider">
            3D TEMPERATURE VOLUME
          </span>
        </div>
        <div className="text-[11px] text-neutral-400">
          Field: <span className="text-cyan-400 uppercase font-semibold">{fieldMode}</span> · 15 Depths
        </div>
      </div>

      {/* Top Right: Probe Coordinate Hover Readout */}
      {hoveredData ? (
        <div className="absolute top-4 right-4 pointer-events-none font-mono text-xs bg-black/80 backdrop-blur-md border border-cyan-400/40 px-4 py-2.5 rounded shadow-[0_0_15px_rgba(0,229,255,0.2)]">
          <div className="text-[10px] uppercase text-cyan-400 font-semibold tracking-wider">
            Active Coordinate Probe
          </div>
          <div className="text-neutral-100 font-medium mt-0.5">
            {hoveredData.lat.toFixed(2)}°N, {hoveredData.lon.toFixed(2)}°E @ {hoveredData.depth} m
          </div>
          <div className="text-amber-400 text-sm font-bold mt-1">
            {hoveredData.value.toFixed(2)} °C
          </div>
        </div>
      ) : (
        <div className="absolute top-4 right-4 pointer-events-none font-mono text-[10px] text-neutral-500 bg-black/50 px-3 py-1.5 rounded border border-white/5">
          ROTATE: LEFT CLICK · PAN: RIGHT CLICK · ZOOM: WHEEL
        </div>
      )}

      {/* Bottom Floating Bar: Active Depth Scrub & Thermal Scale */}
      <div className="absolute bottom-4 left-4 right-4 pointer-events-auto bg-[#0d0f14]/90 backdrop-blur-md border border-white/10 p-3 rounded flex flex-col sm:flex-row items-center justify-between gap-4 font-mono text-xs">
        {/* Depth Slice Selector */}
        <div className="flex items-center gap-3 w-full sm:w-auto">
          <span className="text-[11px] text-neutral-400 uppercase shrink-0">Depth Slice:</span>
          <select
            value={activeDepthIndex}
            onChange={(e) => onDepthSelect && onDepthSelect(Number(e.target.value))}
            className="bg-[#15181f] text-neutral-200 border border-white/10 rounded px-2.5 py-1 text-xs outline-none focus:border-cyan-400 font-mono"
          >
            {depths.map((d, idx) => (
              <option key={d} value={idx}>
                {d} m {idx === 0 ? "(Surface SST)" : idx === 7 ? "(Thermocline 100m)" : idx === 14 ? "(Abyss 1000m)" : ""}
              </option>
            ))}
          </select>
          <span className="text-cyan-400 font-semibold text-xs">
            {depths[activeDepthIndex] ?? 0} meters
          </span>
        </div>

        {/* Scientific Colorbar Legend */}
        <div className="flex items-center gap-2 w-full sm:w-auto justify-end">
          <span className="text-[10px] text-neutral-500">4 °C</span>
          <div className="w-36 h-2.5 rounded-full overflow-hidden bg-gradient-to-r from-blue-700 via-cyan-400 via-yellow-400 to-amber-500" />
          <span className="text-[10px] text-neutral-500">30 °C</span>
        </div>
      </div>
    </div>
  );
}
