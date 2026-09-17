// Fonts: warm editorial serif (Fraunces) + tech mono (IBM Plex Mono) + Inter.
// No purple, no neon, no blue — "paper & ember" print-lab aesthetic.

import type { Metadata } from "next";
import { Fraunces, IBM_Plex_Mono, Inter } from "next/font/google";
import "./globals.css";

const fraunces = Fraunces({
  subsets: ["latin"],
  variable: "--font-fraunces",
  weight: ["400", "600"],
  style: ["normal", "italic"],
});

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  weight: ["400", "500", "600"],
});

const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  variable: "--font-plex",
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "MiraLM — a 47M machine, built by hand",
  description:
    "MiraLM-47M: a from-scratch 47M-parameter hybrid language model — GQA + RoPE + SwiGLU, Mamba SSM, and a sparse top-2 Mixture of Experts with a semantically-seeded router. Built with zero pretrained weights for GIBC V2.",
  openGraph: {
    title: "MiraLM — a 47M machine, built by hand",
    description:
      "From-scratch 47M-parameter hybrid LLM for structured reasoning. No pretrained weights, one consumer GPU, a continuously enforced 50M budget.",
    type: "website",
  },
  robots: { index: true, follow: true },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${fraunces.variable} ${inter.variable} ${plexMono.variable} h-full`}
    >
      <body className="min-h-full antialiased">{children}</body>
    </html>
  );
}