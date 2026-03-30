"use client";

import { ConnectButton } from "@/components/ConnectButton";
import { AuthGuard } from "@/components/AuthGuard";
import { useAccount } from "wagmi";

function DashboardContent() {
  const { address } = useAccount();

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
          <a href="/admin" className="text-gray-400 hover:text-white transition">Admin</a>
          <ConnectButton />
        </div>
      </nav>

      <div className="max-w-6xl mx-auto px-6 py-8 space-y-8">
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

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-dark-800 rounded-xl p-5">
            <div className="text-sm text-gray-500">Balance</div>
            <div className="text-2xl font-bold text-white">—</div>
          </div>
          <div className="bg-dark-800 rounded-xl p-5">
            <div className="text-sm text-gray-500">Total P&L</div>
            <div className="text-2xl font-bold text-gray-500">—</div>
          </div>
          <div className="bg-dark-800 rounded-xl p-5">
            <div className="text-sm text-gray-500">Win Rate</div>
            <div className="text-2xl font-bold text-white">—</div>
          </div>
          <div className="bg-dark-800 rounded-xl p-5">
            <div className="text-sm text-gray-500">Trades</div>
            <div className="text-2xl font-bold text-white">0</div>
          </div>
        </div>

        <div className="bg-dark-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Open Trades</h2>
          <div className="text-gray-500 text-center py-8">
            Configure your MEXC API keys to start trading
          </div>
        </div>

        <div className="bg-dark-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Trade History</h2>
          <div className="text-gray-500 text-center py-8">
            No trades yet
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
