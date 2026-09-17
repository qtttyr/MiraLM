import { SmoothScrollProvider } from "@/components/providers";
import { Nav } from "@/components/nav";
import { Footer } from "@/components/footer";
import { ProgressRuler } from "@/components/progress-ruler";
import { Hero } from "@/components/hero";
import { Ticker } from "@/components/ticker";
import { Architecture } from "@/components/architecture";
import { RouterMatrix } from "@/components/router-matrix";
import { Benchmarks } from "@/components/benchmarks";
import { TryConsole } from "@/components/try-console";
import { Principles } from "@/components/principles";
import { BuildLog } from "@/components/buildlog";

export default function Home() {
  return (
    <SmoothScrollProvider>
      <div className="sheet-noise min-h-screen bg-paper text-ink">
        <ProgressRuler />
        <Nav />
        <main>
          <Hero />
          <Ticker />
          <Architecture />
          <RouterMatrix />
          <Benchmarks />
          <TryConsole />
          <Principles />
          <BuildLog />
        </main>
        <Footer />
      </div>
    </SmoothScrollProvider>
  );
}