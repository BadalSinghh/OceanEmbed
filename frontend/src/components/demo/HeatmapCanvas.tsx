"use client";

import { useEffect, useRef, useMemo } from "react";

interface HeatmapCanvasProps {
  data: (number | null)[][];
  lats: number[];
  lons: number[];
  diverging?: boolean;
  height?: number;
}

// ── Color scales ─────────────────────────────────────────────────────────

function thermalColor(t: number): [number, number, number] {
  // Thermal colorscale: dark blue -> cyan -> yellow -> red
  const stops: [number, [number, number, number]][] = [
    [0.0, [4, 10, 60]],
    [0.2, [20, 60, 140]],
    [0.4, [60, 170, 170]],
    [0.6, [200, 200, 60]],
    [0.8, [220, 120, 30]],
    [1.0, [180, 30, 20]],
  ];
  return interpolateStops(stops, t);
}

function rdBuColor(t: number): [number, number, number] {
  // Diverging RdBu: red(0) -> white(0.5) -> blue(1)
  const stops: [number, [number, number, number]][] = [
    [0.0, [165, 30, 20]],
    [0.25, [220, 120, 80]],
    [0.5, [250, 248, 246]],
    [0.75, [80, 140, 210]],
    [1.0, [20, 60, 160]],
  ];
  return interpolateStops(stops, t);
}

function interpolateStops(
  stops: [number, [number, number, number]][],
  t: number
): [number, number, number] {
  t = Math.max(0, Math.min(1, t));
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i];
    const [t1, c1] = stops[i + 1];
    if (t >= t0 && t <= t1) {
      const f = (t - t0) / (t1 - t0);
      return [
        Math.round(c0[0] + f * (c1[0] - c0[0])),
        Math.round(c0[1] + f * (c1[1] - c0[1])),
        Math.round(c0[2] + f * (c1[2] - c0[2])),
      ];
    }
  }
  return stops[stops.length - 1][1];
}

export default function HeatmapCanvas({
  data,
  lats,
  lons,
  diverging = false,
  height = 280,
}: HeatmapCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Compute data extents
  const { vmin, vmax } = useMemo(() => {
    let mn = Infinity;
    let mx = -Infinity;
    for (const row of data) {
      for (const v of row) {
        if (v != null && isFinite(v)) {
          if (v < mn) mn = v;
          if (v > mx) mx = v;
        }
      }
    }
    if (!isFinite(mn)) return { vmin: 0, vmax: 1 };
    if (diverging) {
      const maxAbs = Math.max(Math.abs(mn), Math.abs(mx));
      return { vmin: -maxAbs, vmax: maxAbs };
    }
    return { vmin: mn, vmax: mx };
  }, [data, diverging]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data || data.length === 0) return;

    const nRows = data.length;
    const nCols = data[0]?.length ?? 0;
    if (nRows === 0 || nCols === 0) return;

    canvas.width = nCols;
    canvas.height = nRows;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const imgData = ctx.createImageData(nCols, nRows);
    const buf = imgData.data;
    const range = vmax - vmin || 1;
    const getColor = diverging ? rdBuColor : thermalColor;

    for (let r = 0; r < nRows; r++) {
      // Flip vertically so lat=min is at bottom
      const srcRow = nRows - 1 - r;
      for (let c = 0; c < nCols; c++) {
        const v = data[srcRow]?.[c];
        const idx = (r * nCols + c) * 4;
        if (v == null || !isFinite(v)) {
          buf[idx] = 6;
          buf[idx + 1] = 14;
          buf[idx + 2] = 27;
          buf[idx + 3] = 255;
        } else {
          const t = (v - vmin) / range;
          const [R, G, B] = getColor(t);
          buf[idx] = R;
          buf[idx + 1] = G;
          buf[idx + 2] = B;
          buf[idx + 3] = 255;
        }
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }, [data, vmin, vmax, diverging]);

  return (
    <div style={{ position: "relative" }}>
      <canvas
        ref={canvasRef}
        style={{
          display: "block",
          width: "100%",
          height: `${height}px`,
          imageRendering: "pixelated",
        }}
      />
      {/* Color-scale bar */}
      <div
        style={{
          position: "absolute",
          bottom: 6,
          right: 6,
          width: 10,
          height: 80,
          borderRadius: 3,
          background: diverging
            ? "linear-gradient(to top, #a51e14, #f8f8f6, #143ca0)"
            : "linear-gradient(to top, #04063c, #14aaa0, #dcc83c, #b41e14)",
          border: "1px solid rgba(255,255,255,0.15)",
        }}
      />
      <div
        style={{
          position: "absolute",
          bottom: 4,
          right: 18,
          fontSize: "0.6rem",
          color: "rgba(255,255,255,0.5)",
          fontFamily: "var(--font-mono)",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          height: 84,
        }}
      >
        <span>{vmax.toFixed(1)}</span>
        <span>{vmin.toFixed(1)}</span>
      </div>
    </div>
  );
}
