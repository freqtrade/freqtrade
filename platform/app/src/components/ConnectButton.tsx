"use client";

import { useAccount, useDisconnect } from "wagmi";

export function useWallet() {
  const { address, isConnected } = useAccount();
  const { disconnect } = useDisconnect();

  return {
    address: address ? address.toLowerCase() : null,
    connected: isConnected,
    disconnect,
  };
}

export function ConnectButton() {
  const { address, connected, disconnect } = useWallet();

  if (connected && address) {
    return (
      <div className="flex items-center gap-3">
        <appkit-button />
      </div>
    );
  }

  return <appkit-button />;
}
