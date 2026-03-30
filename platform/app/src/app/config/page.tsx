"use client";

import { ConnectButton } from "@/components/ConnectButton";
import { AuthGuard } from "@/components/AuthGuard";
import { useAccount } from "wagmi";
import { useState } from "react";
import { apiPost } from "@/lib/api";

function ConfigContent() {
  const { address } = useAccount();
  const [apiKey, setApiKey] = useState("");
  const [apiSecret, setApiSecret] = useState("");
  const [stake, setStake] = useState("1");
  const [maxTrades, setMaxTrades] = useState("10");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  const handleSave = async () => {
    if (!address) return;
    setSaving(true);
    const result = await apiPost("/api/config", address, {
      apiKey: apiKey || undefined,
      apiSecret: apiSecret || undefined,
      stakeAmount: parseFloat(stake),
      maxOpenTrades: parseInt(maxTrades),
    });
    setSaving(false);
    if (result.ok) {
      setSaved(true);
      setApiKey("");
      setApiSecret("");
      setTimeout(() => setSaved(false), 3000);
    }
  };

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

        {saved && (
          <div className="bg-green-500/10 border border-green-500/20 rounded-xl p-4 text-green-400 text-sm">
            ✅ Configuration saved successfully
          </div>
        )}

        <div className="bg-dark-800 rounded-xl p-6 space-y-4">
          <h2 className="text-lg font-semibold text-white">MEXC API Keys</h2>
          <p className="text-sm text-gray-500">
            Encrypted with AES-256. Create keys at{" "}
            <a href="https://www.mexc.com/user/openapi" target="_blank" className="text-accent underline">MEXC API Management</a>.
            Enable Spot Trading only.
          </p>
          <div className="space-y-3">
            <div>
              <label className="block text-sm text-gray-400 mb-1">API Key</label>
              <input type="password" placeholder="mx0vgl..." value={apiKey} onChange={(e) => setApiKey(e.target.value)}
                className="w-full bg-dark-900 border border-dark-700 rounded-lg px-4 py-3 text-white focus:border-accent focus:outline-none" />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">API Secret</label>
              <input type="password" placeholder="••••••••" value={apiSecret} onChange={(e) => setApiSecret(e.target.value)}
                className="w-full bg-dark-900 border border-dark-700 rounded-lg px-4 py-3 text-white focus:border-accent focus:outline-none" />
            </div>
          </div>
        </div>

        <div className="bg-dark-800 rounded-xl p-6 space-y-4">
          <h2 className="text-lg font-semibold text-white">Trading Settings</h2>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Stake per Trade (USDT)</label>
              <input type="number" value={stake} onChange={(e) => setStake(e.target.value)} min="1" max="100"
                className="w-full bg-dark-900 border border-dark-700 rounded-lg px-4 py-3 text-white focus:border-accent focus:outline-none" />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Max Open Trades</label>
              <input type="number" value={maxTrades} onChange={(e) => setMaxTrades(e.target.value)} min="1" max="20"
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
        </div>

        <button
          onClick={handleSave}
          disabled={saving}
          className="w-full bg-accent hover:bg-accent-dark text-dark-900 font-bold py-3 rounded-lg transition disabled:opacity-50"
        >
          {saving ? "Saving..." : "Save Configuration"}
        </button>

        <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-xl p-4 text-sm text-yellow-200/70">
          ⚠️ Only enable Spot Trading permissions on your MEXC API key. Never enable Withdrawal.
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
