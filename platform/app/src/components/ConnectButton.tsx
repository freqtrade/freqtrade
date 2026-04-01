"use client";

import { useAppKit } from "@reown/appkit/react";
import { useAccount, useDisconnect } from "wagmi";

export function ConnectButton() {
  const { open } = useAppKit();
  const { address, isConnected } = useAccount();
  const { disconnect } = useDisconnect();

  if (isConnected && address) {
    return (
      <div className="flex items-center gap-3">
        <div className="bg-dark-700 px-4 py-2 rounded-lg text-sm font-mono text-gray-300">
          {address.slice(0, 6)}...{address.slice(-4)}
        </div>
        <button
          onClick={() => disconnect()}
          className="text-gray-500 hover:text-red-400 text-sm transition"
        >
          Disconnect
        </button>
      </div>
    );
  }

  return (
    <button
      onClick={() => open({ view: "Connect" })}
      className="bg-accent hover:bg-accent-dark text-dark-900 font-semibold px-6 py-2 rounded-lg transition"
    >
      Connect Wallet
    </button>
  );
}
