import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || "https://oceanembed-1-555s.onrender.com",
  },
  // Allow importing Three.js and related packages
  transpilePackages: [],
};

export default nextConfig;
