export default function Dashboard() {
  return (
    <div className="min-h-screen">
      {/* Nav */}
      <nav className="border-b border-dark-700 px-6 py-4 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <span className="text-2xl">📈</span>
          <span className="text-xl font-bold text-white">TrendRider</span>
        </div>
        <div className="flex items-center gap-4">
          <a href="/config" className="text-gray-400 hover:text-white transition">Config</a>
          <a href="/admin" className="text-gray-400 hover:text-white transition">Admin</a>
          <div className="bg-dark-700 px-4 py-2 rounded-lg text-sm font-mono text-gray-300">
            0x...connect
          </div>
        </div>
      </nav>

      <div className="max-w-6xl mx-auto px-6 py-8 space-y-8">
        {/* Status Banner */}
        <div className="bg-dark-800 rounded-xl p-6 flex items-center justify-between">
          <div>
            <div className="text-sm text-gray-500">Bot Status</div>
            <div className="text-2xl font-bold text-accent flex items-center gap-2">
              <span className="w-3 h-3 bg-accent rounded-full animate-pulse" />
              Running
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-500">Strategy</div>
            <div className="text-lg font-semibold text-white">TrendRider v2</div>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-dark-800 rounded-xl p-5">
            <div className="text-sm text-gray-500">Balance</div>
            <div className="text-2xl font-bold text-white">$25.65</div>
          </div>
          <div className="bg-dark-800 rounded-xl p-5">
            <div className="text-sm text-gray-500">Total P&L</div>
            <div className="text-2xl font-bold text-green-400">+$0.012</div>
          </div>
          <div className="bg-dark-800 rounded-xl p-5">
            <div className="text-sm text-gray-500">Win Rate</div>
            <div className="text-2xl font-bold text-white">66%</div>
          </div>
          <div className="bg-dark-800 rounded-xl p-5">
            <div className="text-sm text-gray-500">Trades</div>
            <div className="text-2xl font-bold text-white">32</div>
          </div>
        </div>

        {/* Open Trades */}
        <div className="bg-dark-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Open Trades</h2>
          <div className="text-gray-500 text-center py-8">
            No open trades — waiting for entry signal
          </div>
        </div>

        {/* Trade History */}
        <div className="bg-dark-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Recent Trades</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500 border-b border-dark-700">
                  <th className="text-left py-3 px-2">Pair</th>
                  <th className="text-right py-3 px-2">Entry</th>
                  <th className="text-right py-3 px-2">Exit</th>
                  <th className="text-right py-3 px-2">P&L</th>
                  <th className="text-right py-3 px-2">Duration</th>
                  <th className="text-left py-3 px-2">Reason</th>
                </tr>
              </thead>
              <tbody className="text-gray-300">
                <tr className="border-b border-dark-700/50">
                  <td className="py-3 px-2 font-medium">ETH/USDT</td>
                  <td className="text-right py-3 px-2">$1,932.00</td>
                  <td className="text-right py-3 px-2">$1,946.00</td>
                  <td className="text-right py-3 px-2 text-green-400">+0.72%</td>
                  <td className="text-right py-3 px-2">56m</td>
                  <td className="py-3 px-2"><span className="bg-green-500/20 text-green-400 px-2 py-0.5 rounded text-xs">roi</span></td>
                </tr>
                <tr className="border-b border-dark-700/50">
                  <td className="py-3 px-2 font-medium">BTC/USDT</td>
                  <td className="text-right py-3 px-2">$67,200</td>
                  <td className="text-right py-3 px-2">$66,560</td>
                  <td className="text-right py-3 px-2 text-red-400">-0.95%</td>
                  <td className="text-right py-3 px-2">20m</td>
                  <td className="py-3 px-2"><span className="bg-red-500/20 text-red-400 px-2 py-0.5 rounded text-xs">stop_loss</span></td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
