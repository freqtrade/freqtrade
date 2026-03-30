export default function Admin() {
  return (
    <div className="min-h-screen">
      <nav className="border-b border-dark-700 px-6 py-4 flex justify-between items-center">
        <a href="/dashboard" className="flex items-center gap-2">
          <span className="text-2xl">📈</span>
          <span className="text-xl font-bold text-white">TrendRider</span>
        </a>
        <span className="bg-red-500/20 text-red-400 px-3 py-1 rounded text-xs font-bold">ADMIN</span>
      </nav>

      <div className="max-w-4xl mx-auto px-6 py-8 space-y-8">
        <h1 className="text-3xl font-bold text-white">Admin — Whitelist Management</h1>

        {/* Add wallet */}
        <div className="bg-dark-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Add Wallet to Whitelist</h2>
          <div className="flex gap-3">
            <input
              type="text"
              placeholder="0x..."
              className="flex-1 bg-dark-900 border border-dark-700 rounded-lg px-4 py-3 text-white font-mono text-sm focus:border-accent focus:outline-none"
            />
            <button className="bg-accent hover:bg-accent-dark text-dark-900 font-bold px-6 py-3 rounded-lg transition">
              Whitelist
            </button>
          </div>
        </div>

        {/* Whitelisted users */}
        <div className="bg-dark-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Whitelisted Users</h2>
          <div className="space-y-3">
            {[
              { wallet: "0x1234...abcd", status: "running", trades: 32, pnl: "+$0.012" },
              { wallet: "0x5678...efgh", status: "stopped", trades: 0, pnl: "$0.000" },
            ].map((user, i) => (
              <div key={i} className="flex items-center justify-between bg-dark-900 rounded-lg p-4">
                <div className="flex items-center gap-3">
                  <span className={`w-2 h-2 rounded-full ${user.status === "running" ? "bg-green-400" : "bg-gray-600"}`} />
                  <span className="font-mono text-sm text-gray-300">{user.wallet}</span>
                </div>
                <div className="flex items-center gap-6 text-sm">
                  <span className="text-gray-500">{user.trades} trades</span>
                  <span className="text-green-400">{user.pnl}</span>
                  <button className="text-red-400 hover:text-red-300 text-xs">Remove</button>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Platform stats */}
        <div className="bg-dark-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Platform Stats</h2>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <div className="text-2xl font-bold text-white">2</div>
              <div className="text-sm text-gray-500">Active Users</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-white">1</div>
              <div className="text-sm text-gray-500">Running Bots</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-accent">32</div>
              <div className="text-sm text-gray-500">Total Trades</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
