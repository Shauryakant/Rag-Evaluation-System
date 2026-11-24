import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  eslint: {
    // Prevent non-fatal ESLint warnings/formatting rules from blocking Vercel deployments
    ignoreDuringBuilds: true,
  },
  typescript: {
    // Keep strict type-checking enabled
    ignoreBuildErrors: false,
  },
};

export default nextConfig;
