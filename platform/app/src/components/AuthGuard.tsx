"use client";

import { ConnectButton, useWallet } from "./ConnectButton";

const ADMIN_WALLETS = [
  "0xaedb312d90fa956775ea8abed298ea3b085abbd9",
];

export function AuthGuard({ children }: { children: React.ReactNode }) {
  const { connected } = useWallet();

  if (!connected) {
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

  return <>{children}</>;
}

export function useIsAdmin() {
  const { address } = useWallet();
  return address ? ADMIN_WALLETS.includes(address.toLowerCase()) : false;
}
