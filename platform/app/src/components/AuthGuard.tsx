"use client";

import { useAccount } from "wagmi";
import { ConnectButton } from "./ConnectButton";

// Hardcoded admin wallets — add yours here
const ADMIN_WALLETS = [
  "0xb6c0b51dcb745a1bc41d503d3a1c959e857200f2",
];

// For now, all connected wallets are whitelisted (MVP)
// Later: check against database whitelist
const WHITELISTED = true;

export function AuthGuard({ children }: { children: React.ReactNode }) {
  const { isConnected } = useAccount();

  if (!isConnected) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-6">
        <div className="text-center space-y-3">
          <span className="text-5xl">🔒</span>
          <h1 className="text-2xl font-bold text-white">Connect Your Wallet</h1>
          <p className="text-gray-500">You need a whitelisted wallet to access TrendRider</p>
        </div>
        <ConnectButton />
      </div>
    );
  }

  if (!WHITELISTED) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-4">
        <span className="text-5xl">⛔</span>
        <h1 className="text-2xl font-bold text-white">Access Denied</h1>
        <p className="text-gray-500">Your wallet is not whitelisted. Contact admin.</p>
      </div>
    );
  }

  return <>{children}</>;
}

export function useIsAdmin() {
  const { address } = useAccount();
  return address ? ADMIN_WALLETS.includes(address.toLowerCase()) : false;
}
