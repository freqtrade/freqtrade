"use client";

import { ConnectButton } from "@/components/ConnectButton";
import { AuthGuard } from "@/components/AuthGuard";

function ConfigContent() {
  return (
    <div className="min-h-screen">
      <nav className="border-b border-dark-700 px-6 py-4 flex justify-between items-center">
        <a href="/dashboard" className="flex items-center gap-2">
          <span className="text-2xl">📈</span>
          <span className="text-xl font-bold text-white">TrendRider</span>
        </a>
        <ConnectButton />
      </nav>

      <div className="max-w-2xl mx-auto px-6 py-8 space-y-8">
        <h1 className="text-3xl font-bold text-white">Bot Configuration</h1>

        <div className="bg-dark-800 rounded-xl p-6 space-y-4">
          <h2 className="text-lg font-semibold text-white">MEXC API Keys</h2>
          <p className="text-sm text-gray-500">
            Your keys are encrypted with AES-256 and never visible after saving.
            Create keys at{" "}
            <a href="https://www.mexc.com/user/openapi" target="_blank" className="text-accent underline">
              MEXC API Management
            </a>
            . Enable Spot Trading only — disable Withdrawal.
          </p>
          <div className="space-y-3">
            <div>
              <label className="block text-sm text-gray-400 mb-1">API Key</label>
              <input type="password" placeholder="mx0vgl..."
                className="w-full bg-dark-900 border border-dark-700 rounded-lg px-4 py-3 text-white focus:border-accent focus:outline-none" />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">API Secret</label>
              <input type="password" placeholder="••••••••••••••••"
                className="w-full bg-dark-900 border border-dark-700 rounded-lg px-4 py-3 text-white focus:border-accent focus:outline-none" />
            </div>
          </div>
        </div>

        <div className="bg-dark-800 rounded-xl p-6 space-y-4">
          <h2 className="text-lg font-semibold text-white">Trading Settings</h2>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Stake per Trade (USDT)</label>
              <input type="number" defaultValue="1" min="1" max="100"
                className="w-full bg-dark-900 border border-dark-700 rounded-lg px-4 py-3 text-white focus:border-accent focus:outline-none" />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Max Open Trades</label>
              <input type="number" defaultValue="10" min="1" max="20"
                className="w-full bg-dark-900 border border-dark-700 rounded-lg px-4 py-3 text-white focus:border-accent focus:outline-none" />
            </div>
          </div>
        </div>

        <div className="bg-dark-800 rounded-xl p-6 space-y-3 opacity-75">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-white">Strategy</h2>
            <span className="bg-dark-700 text-gray-400 px-3 py-1 rounded text-xs">🔒 Locked</span>
          </div>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div><span className="text-gray-500">Name:</span> <span className="text-white">TrendRider v2</span></div>
            <div><span className="text-gray-500">Timeframe:</span> <span className="text-white">5m</span></div>
            <div><span className="text-gray-500">Stop Loss:</span> <span className="text-white">-0.9%</span></div>
            <div><span className="text-gray-500">Trailing:</span> <span className="text-white">0.3% @ 1.2%</span></div>
          </div>
          <p className="text-xs text-gray-600">Strategy is managed by the TrendRider team and cannot be modified.</p>
        </div>

        <div className="flex gap-4">
          <button className="flex-1 bg-accent hover:bg-accent-dark text-dark-900 font-bold py-3 rounded-lg transition">
            Save & Start Bot
          </button>
          <button className="px-6 bg-red-500/20 hover:bg-red-500/30 text-red-400 font-bold py-3 rounded-lg transition">
            Stop Bot
          </button>
        </div>

        <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-xl p-4 text-sm text-yellow-200/70">
          ⚠️ Your API keys grant trading access to your MEXC account. Only enable
          Spot Trading permissions. Never enable Withdrawal.
        </div>
      </div>
    </div>
  );
}

export default function Config() {
  return (
    <AuthGuard>
      <ConfigContent />
    </AuthGuard>
  );
}
