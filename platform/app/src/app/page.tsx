"use client";

import { ConnectButton, useWallet } from "@/components/ConnectButton";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

export default function Home() {
  const { connected } = useWallet();
  const router = useRouter();

  useEffect(() => {
    if (connected) router.push("/dashboard");
  }, [connected, router]);

  return (
    <div className="min-h-screen flex flex-col">
      <nav className="border-b border-dark-700 px-6 py-4 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <span className="text-2xl">📈</span>
          <span className="text-xl font-bold text-white">TrendRider</span>
        </div>
        <ConnectButton />
      </nav>

      <main className="flex-1 flex flex-col items-center justify-center px-6">
        <div className="max-w-2xl text-center space-y-6">
          <h1 className="text-5xl font-bold text-white leading-tight">
            Automated Crypto Trading
            <br />
            <span className="text-accent">On Your Terms</span>
          </h1>
          <p className="text-lg text-gray-400">
            Run a proven trend-following strategy on your MEXC account.
            You keep your keys. You keep your funds. We provide the edge.
          </p>
          <div className="grid grid-cols-3 gap-4 pt-6">
            <div className="bg-dark-800 rounded-xl p-4">
              <div className="text-2xl font-bold text-accent">$10K+</div>
              <div className="text-sm text-gray-500">Volume Traded</div>
            </div>
            <div className="bg-dark-800 rounded-xl p-4">
              <div className="text-2xl font-bold text-accent">1.98</div>
              <div className="text-sm text-gray-500">Profit Factor</div>
            </div>
            <div className="bg-dark-800 rounded-xl p-4">
              <div className="text-2xl font-bold text-accent">24/7</div>
              <div className="text-sm text-gray-500">Automated</div>
            </div>
          </div>
          <div className="pt-4">
            <ConnectButton />
          </div>
        </div>
      </main>

      <footer className="border-t border-dark-700 px-6 py-8 text-center text-sm text-gray-600 max-w-3xl mx-auto">
        <p className="font-semibold text-gray-400 mb-2">⚠️ Disclaimer</p>
        <p>
          TrendRider is experimental software in active development. Trading
          cryptocurrency involves substantial risk of loss. Use at your own risk.{" "}
          <a href="/terms" className="text-accent underline">Terms & Conditions</a>
        </p>
      </footer>
    </div>
  );
}
