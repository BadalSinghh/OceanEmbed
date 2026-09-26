"use client";

import React, { useEffect, useState } from "react";

interface CinematicOpeningProps {
  onComplete?: () => void;
}

export default function CinematicOpening({ onComplete }: CinematicOpeningProps) {
  const [visible, setVisible] = useState(true);
  const [fadingOut, setFadingOut] = useState(false);

  useEffect(() => {
    // Respect reduced motion
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReducedMotion) {
      setVisible(false);
      if (onComplete) onComplete();
      return;
    }

    // Check if already shown in this tab session
    const hasSeen = sessionStorage.getItem("oceanembed_intro_seen");
    if (hasSeen) {
      setVisible(false);
      if (onComplete) onComplete();
      return;
    }

    // Start fade out after 1.8 seconds
    const timer = setTimeout(() => {
      setFadingOut(true);
      setTimeout(() => {
        setVisible(false);
        sessionStorage.setItem("oceanembed_intro_seen", "true");
        if (onComplete) onComplete();
      }, 1200);
    }, 1800);

    // Skip on user interaction (click, scroll, keypress)
    const handleSkip = () => {
      setFadingOut(true);
      setTimeout(() => {
        setVisible(false);
        sessionStorage.setItem("oceanembed_intro_seen", "true");
        if (onComplete) onComplete();
      }, 400);
    };

    window.addEventListener("click", handleSkip, { once: true });
    window.addEventListener("wheel", handleSkip, { once: true });
    window.addEventListener("keydown", handleSkip, { once: true });
    window.addEventListener("touchstart", handleSkip, { once: true });

    return () => {
      clearTimeout(timer);
      window.removeEventListener("click", handleSkip);
      window.removeEventListener("wheel", handleSkip);
      window.removeEventListener("keydown", handleSkip);
      window.removeEventListener("touchstart", handleSkip);
    };
  }, [onComplete]);

  if (!visible) return null;

  return (
    <div
      className={`fixed inset-0 z-[99999] bg-[#050505] flex flex-col items-center justify-center select-none pointer-events-auto transition-opacity duration-1000 ease-out ${
        fadingOut ? "opacity-0 pointer-events-none" : "opacity-100"
      }`}
    >
      <div className="text-center px-6 space-y-4">
        <h1
          className="text-white text-3xl sm:text-5xl md:text-6xl font-light tracking-[0.35em] uppercase font-sans"
          style={{ letterSpacing: "0.35em" }}
        >
          O C E A N E M B E D
        </h1>
        <div className="pt-2">
          <p className="text-neutral-400 font-mono text-xs sm:text-sm tracking-[0.25em] uppercase">
            BAY OF BENGAL
          </p>
          <p className="text-neutral-500 font-mono text-[11px] sm:text-xs tracking-[0.2em] uppercase mt-1">
            RESEARCH SYSTEM
          </p>
        </div>
      </div>

      <div className="absolute bottom-8 text-neutral-600 font-mono text-[10px] tracking-widest uppercase">
        Click or scroll to continue
      </div>
    </div>
  );
}
