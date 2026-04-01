"use client";

import { ConnectButton } from "@/components/ConnectButton";
import { AuthGuard } from "@/components/AuthGuard";
import { useWallet } from "@/components/ConnectButton";
import { useEffect, useState } from "react";
import { apiGet, apiPost } from "@/lib/api";

function DashboardContent() {
  const { address } = useWallet();
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!address) return;
    apiPost("/api/user", address, {});
    apiGet("/api/user", address).then((d) => {
      setData(d);
      setLoading(false);
    });
  }, [address]);

  const user = data?.user;
  const config = data?.config;
  const botStatus = config?.botStatus || "stopped";
  const hasKey = config?.hasApiKey;

  return (
    <div className="min-h-screen">
      <nav className="border-b border-dark-700 px-6 py-4 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <span className="text-2xl">📈</span>
          <span className="text-xl font-bold text-white">TrendRider</span>
        </div>
        <div className="flex items-center gap-4">
          <a href="/dashboard" className="text-accent font-medium">Dashboard</a>
          <a href="/config" className="text-gray-400 hover:text-white transition">Config</a>
          {user?.isAdmin && <a href="/admin" className="text-gray-400 hover:text-white transition">Admin</a>}
          <ConnectButton />
        </div>
      </nav>

      <div className="max-w-6xl mx-auto px-6 py-8 space-y-8">
        {!user?.isWhitelisted && (
          <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-xl p-6 text-center">
            <span className="text-3xl">⏳</span>
            <h2 className="text-xl font-bold text-white mt-2">Awaiting Whitelist</h2>
            <p className="text-gray-400 mt-1">Your wallet is registered. Contact admin for access.</p>
            <p className="text-xs text-gray-600 font-mono mt-2">{address}</p>
          </div>
        )}

        {user?.isWhitelisted && !hasKey && (
          <div className="bg-dark-800 rounded-xl p-6 flex items-center justify-between">
            <div>
              <div className="text-sm text-gray-500">Bot Status</div>
              <div className="text-2xl font-bold text-gray-500 flex items-center gap-2">
                <span className="w-3 h-3 bg-gray-500 rounded-full" />
                Not Configured
              </div>
            </div>
            <a href="/config" className="bg-accent hover:bg-accent-dark text-dark-900 font-semibold px-6 py-2 rounded-lg transition">
              Set Up Bot →
            </a>
          </div>
        )}

        {user?.isWhitelisted && hasKey && (
          <div className="bg-dark-800 rounded-xl p-6 flex items-center justify-between">
            <div>
              <div className="text-sm text-gray-500">Bot Status</div>
              <div className={`text-2xl font-bold flex items-center gap-2 ${botStatus === "running" ? "text-accent" : "text-gray-500"}`}>
                <span className={`w-3 h-3 rounded-full ${botStatus === "running" ? "bg-accent animate-pulse" : "bg-gray-500"}`} />
                {botStatus === "running" ? "Running" : "Stopped"}
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm text-gray-500">Strategy</div>
              <div className="text-lg font-semibold text-white">TrendRider v2</div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-dark-800 rounded-xl p-5">
            <div className="text-sm text-gray-500">Stake / Trade</div>
            <div className="text-2xl font-bold text-white">${config?.stakeAmount || "—"}</div>
          </div>
          <div className="bg-dark-800 rounded-xl p-5">
            <div className="text-sm text-gray-500">Max Open</div>
            <div className="text-2xl font-bold text-white">{config?.maxOpenTrades || "—"}</div>
          </div>
          <div className="bg-dark-800 rounded-xl p-5">
            <div className="text-sm text-gray-500">API Key</div>
            <div className="text-2xl font-bold text-white">{hasKey ? "✅" : "❌"}</div>
          </div>
          <div className="bg-dark-800 rounded-xl p-5">
            <div className="text-sm text-gray-500">Whitelisted</div>
            <div className="text-2xl font-bold text-white">{user?.isWhitelisted ? "✅" : "⏳"}</div>
          </div>
        </div>

        <div className="bg-dark-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Trade History</h2>
          <div className="text-gray-500 text-center py-8">
            {hasKey ? "Bot is running — trades will appear here" : "Configure your MEXC API keys to start trading"}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function Dashboard() {
  return (
    <AuthGuard>
      <DashboardContent />
    </AuthGuard>
  );
}
