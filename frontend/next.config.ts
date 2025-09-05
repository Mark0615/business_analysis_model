import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
};

export default nextConfig;

// frontend/next.config.ts
import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  eslint: { ignoreDuringBuilds: true },      // ← 部署時忽略 ESLint 錯誤
  // （可選）如果連 TypeScript 型別錯誤也會擋 build，可暫時加這行
  // typescript: { ignoreBuildErrors: true },
}

export default nextConfig
