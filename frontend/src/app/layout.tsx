import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const notoTC = Noto_Sans_TC({
  variable: "--font-noto-tc",
  weight: ["400", "500", "700"],
  subsets: ["latin"], // 不用特別填 "chinese-traditional"，Noto 會自帶 CJK glyphs
  display: "swap",
});

export const metadata: Metadata = {
  title: "Business Analytics Model",
  description: "Business Product Analytics Model",
  icons: {
  icon: '/rocket.png',
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-Hant">
      <head>
        <meta charSet="utf-8" /> {/* 明確宣告 UTF-8 */}
      </head>
      <body
        className={[
          geistSans.variable,
          geistMono.variable,
          notoTC.variable,     // ⬅ 把 Noto TC 帶進來
          "antialiased",
        ].join(" ")}
      >
        {children}
      </body>
    </html>
  );
}
