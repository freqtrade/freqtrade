"use client";

import { ConnectButton } from "@/components/ConnectButton";
import { AuthGuard, useIsAdmin } from "@/components/AuthGuard";
import { useAccount } from "wagmi";
import { useEffect, useState } from "react";
import { apiGet, apiPost, apiDelete } from "@/lib/api";

function AdminContent() {
  const isAdmin = useIsAdmin();
  const { address } = useAccount();
  const [users, setUsers] = useState<any[]>([]);
  const [newWallet, setNewWallet] = useState("");
  const [loading, setLoading] = useState(true);

  const loadUsers = () => {
    if (!address) return;
    apiGet("/api/admin/whitelist", address).then((d) => {
      setUsers(d.users || []);
      setLoading(false);
    });
  };

  useEffect(() => { loadUsers(); }, [address]);

  const addWallet = async () => {
    if (!address || !newWallet) return;
    await apiPost("/api/admin/whitelist", address, { targetWallet: newWallet, whitelist: true });
    setNewWallet("");
    loadUsers();
  };

  const removeWallet = async (target: string) => {
    if (!address) return;
    await apiDelete("/api/admin/whitelist", address, { targetWallet: target });
    loadUsers();
  };

  if (!isAdmin) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-4">
        <span className="text-5xl">⛔</span>
        <h1 className="text-2xl font-bold text-white">Admin Only</h1>
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
            <input type="text" placeholder="0x..." value={newWallet} onChange={(e) => setNewWallet(e.target.value)}
              className="flex-1 bg-dark-900 border border-dark-700 rounded-lg px-4 py-3 text-white font-mono text-sm focus:border-accent focus:outline-none" />
            <button onClick={addWallet}
              className="bg-accent hover:bg-accent-dark text-dark-900 font-bold px-6 py-3 rounded-lg transition">
              Whitelist
            </button>
          </div>
        </div>

        <div className="bg-dark-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Users ({users.length})</h2>
          {loading ? (
            <div className="text-gray-500 text-center py-4">Loading...</div>
          ) : users.length === 0 ? (
            <div className="text-gray-500 text-center py-4">No users yet</div>
          ) : (
            <div className="space-y-3">
              {users.map((u) => (
                <div key={u.id} className="flex items-center justify-between bg-dark-900 rounded-lg p-4">
                  <div className="flex items-center gap-3">
                    <span className={`w-2 h-2 rounded-full ${u.isWhitelisted ? "bg-green-400" : "bg-gray-600"}`} />
                    <span className="font-mono text-sm text-gray-300">
                      {u.walletAddress?.slice(0, 6)}...{u.walletAddress?.slice(-4)}
                    </span>
                    {u.isAdmin && <span className="bg-red-500/20 text-red-400 px-2 py-0.5 rounded text-xs">admin</span>}
                    {u.hasApiKey && <span className="bg-green-500/20 text-green-400 px-2 py-0.5 rounded text-xs">API set</span>}
                  </div>
                  <div className="flex items-center gap-4 text-sm">
                    <span className={u.isWhitelisted ? "text-green-400" : "text-gray-600"}>
                      {u.isWhitelisted ? "whitelisted" : "pending"}
                    </span>
                    {!u.isAdmin && (
                      <button onClick={() => removeWallet(u.walletAddress)}
                        className="text-red-400 hover:text-red-300 text-xs">
                        Remove
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="bg-dark-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Platform Stats</h2>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <div className="text-2xl font-bold text-white">{users.filter((u) => u.isWhitelisted).length}</div>
              <div className="text-sm text-gray-500">Whitelisted</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-white">{users.filter((u) => u.hasApiKey).length}</div>
              <div className="text-sm text-gray-500">Configured</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-accent">{users.length}</div>
              <div className="text-sm text-gray-500">Total Users</div>
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
