"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_LINKS = [
  { href: "/", label: "Home" },
  { href: "/demo", label: "Explore" },
  { href: "/results", label: "Results" },
  { href: "/architecture", label: "Architecture" },
  { href: "/research", label: "Research" },
];

export function Nav() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 w-full bg-[#050505]/90 backdrop-blur-md rule-b">
      <div className="w-full px-6 md:px-12 lg:px-16 h-16 flex items-center justify-between">
        {/* Left: Scientific Brand Identifier */}
        <Link href="/" className="flex items-baseline gap-4 group">
          <span className="font-mono text-sm tracking-wider font-semibold text-neutral-100 group-hover:text-white transition-colors">
            OCEANEMBED
          </span>
          <span className="hidden sm:inline font-mono text-[11px] text-neutral-500 uppercase tracking-widest pl-3 border-l border-white/10">
            RESEARCH / BAY OF BENGAL
          </span>
        </Link>

        {/* Center: Editorial Navigation Links */}
        <nav className="flex items-center gap-6 md:gap-8">
          {NAV_LINKS.map(({ href, label }) => {
            const isActive = href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={`text-xs md:text-[13px] tracking-wide transition-colors py-1 ${
                  isActive
                    ? "text-white font-medium border-b border-white"
                    : "text-neutral-400 hover:text-neutral-200"
                }`}
              >
                {label}
              </Link>
            );
          })}
        </nav>

        {/* Right: Direct Action Link */}
        <div className="flex items-center">
          <Link
            href="/demo"
            className="text-xs md:text-[13px] font-mono tracking-wide text-neutral-300 hover:text-white flex items-center gap-1.5 transition-colors"
          >
            <span>Launch Reconstruction</span>
            <span className="text-neutral-500 group-hover:text-white">→</span>
          </Link>
        </div>
      </div>
    </header>
  );
}
