"use client";

import { ConnectButton } from "@/components/ConnectButton";
import { AuthGuard, useIsAdmin } from "@/components/AuthGuard";

function AdminContent() {
  const isAdmin = useIsAdmin();

  if (!isAdmin) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-4">
        <span className="text-5xl">⛔</span>
        <h1 className="text-2xl font-bold text-white">Admin Only</h1>
        <p className="text-gray-500">Your wallet is not an admin wallet.</p>
        <a href="/dashboard" className="text-accent underline">Back to Dashboard</a>
      </div>
    );
  }

  return (
    <div className="min-h-screen">
      <nav className="border-b border-dark-700 px-6 py-4 flex justify-between items-center">
        <a href="/dashboard" className="flex items-center gap-2">
          <span className="text-2xl">📈</span>
          <span className="text-xl font-bold text-white">TrendRider</span>
        </a>
        <div className="flex items-center gap-4">
          <span className="bg-red-500/20 text-red-400 px-3 py-1 rounded text-xs font-bold">ADMIN</span>
          <ConnectButton />
        </div>
      </nav>

      <div className="max-w-4xl mx-auto px-6 py-8 space-y-8">
        <h1 className="text-3xl font-bold text-white">Whitelist Management</h1>

        <div className="bg-dark-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Add Wallet</h2>
          <div className="flex gap-3">
            <input type="text" placeholder="0x..."
              className="flex-1 bg-dark-900 border border-dark-700 rounded-lg px-4 py-3 text-white font-mono text-sm focus:border-accent focus:outline-none" />
            <button className="bg-accent hover:bg-accent-dark text-dark-900 font-bold px-6 py-3 rounded-lg transition">
              Whitelist
            </button>
          </div>
        </div>

        <div className="bg-dark-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Whitelisted Users</h2>
          <div className="text-gray-500 text-center py-8">
            No users whitelisted yet
          </div>
        </div>

        <div className="bg-dark-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Platform Stats</h2>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <div className="text-2xl font-bold text-white">0</div>
              <div className="text-sm text-gray-500">Active Users</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-white">0</div>
              <div className="text-sm text-gray-500">Running Bots</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-accent">0</div>
              <div className="text-sm text-gray-500">Total Trades</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function Admin() {
  return (
    <AuthGuard>
      <AdminContent />
    </AuthGuard>
  );
}
