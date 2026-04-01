"use client";

import { useState, useEffect, useCallback } from "react";

declare global {
  interface Window {
    ethereum?: any;
  }
}

export function useWallet() {
  const [address, setAddress] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const saved = localStorage.getItem("wallet_address");
    if (saved && window.ethereum) {
      setAddress(saved);
      setConnected(true);
    }
  }, []);

  const connect = useCallback(async () => {
    if (!window.ethereum) {
      alert("Please install MetaMask");
      return;
    }
    try {
      const accounts = await window.ethereum.request({
        method: "eth_requestAccounts",
      });
      if (accounts[0]) {
        setAddress(accounts[0]);
        setConnected(true);
        localStorage.setItem("wallet_address", accounts[0]);
      }
    } catch (err) {
      console.error("Connect failed:", err);
    }
  }, []);

  const disconnect = useCallback(() => {
    setAddress(null);
    setConnected(false);
    localStorage.removeItem("wallet_address");
  }, []);

  return { address, connected, connect, disconnect };
}

export function ConnectButton() {
  const { address, connected, connect, disconnect } = useWallet();

  if (connected && address) {
    return (
      <div className="flex items-center gap-3">
        <div className="bg-dark-700 px-4 py-2 rounded-lg text-sm font-mono text-gray-300">
          {address.slice(0, 6)}...{address.slice(-4)}
        </div>
        <button
          onClick={disconnect}
          className="text-gray-500 hover:text-red-400 text-sm transition"
        >
          Disconnect
        </button>
      </div>
    );
  }

  return (
    <button
      onClick={connect}
      className="bg-accent hover:bg-accent-dark text-dark-900 font-semibold px-6 py-2 rounded-lg transition"
    >
      Connect Wallet
    </button>
  );
}
