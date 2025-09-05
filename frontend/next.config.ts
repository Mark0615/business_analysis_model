// next.config.ts
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  eslint: { ignoreDuringBuilds: true },     // ← 忽略 ESLint 錯誤
  typescript: { ignoreBuildErrors: true },  // ← 忽略 TS 型別錯誤
};

export default nextConfig;
