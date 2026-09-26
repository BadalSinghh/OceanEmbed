"use client";

import React, { useEffect, useRef, useState } from "react";

interface PlotlyChartProps {
  data: any[];
  layout?: Record<string, any>;
  config?: Record<string, any>;
  style?: React.CSSProperties;
  className?: string;
  onHover?: (event: any) => void;
  onClick?: (event: any) => void;
}

export default function PlotlyChart({
  data,
  layout = {},
  config = {},
  style,
  className = "",
  onHover,
  onClick,
}: PlotlyChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const plotlyRef = useRef<any>(null);
  const [isLoaded, setIsLoaded] = useState(false);

  // Strict layout defaults eliminating overlapping titles and cramped margins
  const buildMergedLayout = (userLayout: Record<string, any>) => {
    const defaultLayout = {
      // NEVER set title inside Plotly layout by default — titles belong to HTML headers!
      title: undefined,
      autosize: true,
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: {
        family: "IBM Plex Sans, -apple-system, sans-serif",
        color: "#a1a1aa",
        size: 12,
      },
      // Generous margins preventing collision of axis labels and legends
      margin: {
        l: 80,
        r: 48,
        t: 24,
        b: 64,
        pad: 4,
      },
      hovermode: "closest" as const,
      hoverlabel: {
        bgcolor: "#111111",
        bordercolor: "rgba(255,255,255,0.18)",
        font: { family: "IBM Plex Mono, monospace", size: 12, color: "#f4f4f5" },
      },
      xaxis: {
        gridcolor: "rgba(255, 255, 255, 0.05)",
        linecolor: "rgba(255, 255, 255, 0.12)",
        tickcolor: "rgba(255, 255, 255, 0.12)",
        zerolinecolor: "rgba(255, 255, 255, 0.12)",
        tickfont: { family: "IBM Plex Mono, monospace", size: 11, color: "#a1a1aa" },
        title: {
          font: { family: "IBM Plex Sans, sans-serif", size: 12, color: "#d4d4d8" },
          standoff: 14,
        },
      },
      yaxis: {
        gridcolor: "rgba(255, 255, 255, 0.05)",
        linecolor: "rgba(255, 255, 255, 0.12)",
        tickcolor: "rgba(255, 255, 255, 0.12)",
        zerolinecolor: "rgba(255, 255, 255, 0.12)",
        tickfont: { family: "IBM Plex Mono, monospace", size: 11, color: "#a1a1aa" },
        title: {
          font: { family: "IBM Plex Sans, sans-serif", size: 12, color: "#d4d4d8" },
          standoff: 14,
        },
      },
      legend: {
        font: { family: "IBM Plex Mono, monospace", size: 11, color: "#a1a1aa" },
        bgcolor: "rgba(10, 10, 10, 0.85)",
        bordercolor: "rgba(255, 255, 255, 0.08)",
        borderwidth: 1,
      },
    };

    return {
      ...defaultLayout,
      ...userLayout,
      // Ensure layout.title is strictly undefined unless explicitly passed
      title: userLayout.title ?? undefined,
      margin: { ...defaultLayout.margin, ...(userLayout.margin || {}) },
      xaxis: { ...defaultLayout.xaxis, ...(userLayout.xaxis || {}) },
      yaxis: { ...defaultLayout.yaxis, ...(userLayout.yaxis || {}) },
      legend: { ...defaultLayout.legend, ...(userLayout.legend || {}) },
    };
  };

  useEffect(() => {
    let isMounted = true;

    import("plotly.js-dist-min")
      .then((PlotlyModule) => {
        if (!isMounted || !containerRef.current) return;
        const Plotly = PlotlyModule.default || PlotlyModule;
        plotlyRef.current = Plotly;

        const mergedLayout = buildMergedLayout(layout);
        const defaultConfig = {
          responsive: true,
          displayModeBar: false,
          displaylogo: false,
          scrollZoom: false,
          ...config,
        };

        Plotly.newPlot(containerRef.current, data, mergedLayout, defaultConfig).then(() => {
          if (isMounted) setIsLoaded(true);

          if (onClick && containerRef.current) {
            (containerRef.current as any).on("plotly_click", onClick);
          }
          if (onHover && containerRef.current) {
            (containerRef.current as any).on("plotly_hover", onHover);
          }
        });
      })
      .catch((err) => {
        console.error("Failed to load Plotly:", err);
      });

    return () => {
      isMounted = false;
      if (containerRef.current && plotlyRef.current) {
        try {
          plotlyRef.current.purge(containerRef.current);
        } catch {
          // ignore cleanup errors
        }
      }
    };
  }, []);

  // Update plot when data or layout changes
  useEffect(() => {
    if (!isLoaded || !containerRef.current || !plotlyRef.current) return;

    const mergedLayout = buildMergedLayout(layout);
    plotlyRef.current.react(containerRef.current, data, mergedLayout, {
      responsive: true,
      displayModeBar: false,
      displaylogo: false,
      ...config,
    });
  }, [data, layout, config, isLoaded]);

  // Handle window resize cleanly
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current && plotlyRef.current && isLoaded) {
        plotlyRef.current.Plots.resize(containerRef.current);
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [isLoaded]);

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        minHeight: layout?.height ?? 520,
        ...style,
      }}
      className={`plotly-chart-container ${className}`}
    />
  );
}
