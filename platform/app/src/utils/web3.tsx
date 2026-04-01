"use client";

import { createAppKit } from "@reown/appkit/react";
import { WagmiProvider } from "wagmi";
import { mainnet, arbitrum, polygon, base } from "@reown/appkit/networks";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { WagmiAdapter } from "@reown/appkit-adapter-wagmi";
import { type ReactNode, useState, useEffect } from "react";

const projectId = "29d1dbf9d19b903551f69499bcbe825e";

const metadata = {
  name: "TrendRider",
  description: "Automated Crypto Trading Platform",
  url: typeof window !== "undefined" ? window.location.origin : "https://trendrider-platform-production.up.railway.app",
  icons: ["https://avatars.githubusercontent.com/u/37784886"],
};

const networks = [mainnet, arbitrum, polygon, base];

const wagmiAdapter = new WagmiAdapter({
  networks,
  projectId,
  ssr: true,
});

let appKitInitialized = false;

const queryClient = new QueryClient();

export function Web3Provider({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!appKitInitialized) {
      createAppKit({
        adapters: [wagmiAdapter] as any,
        networks: [mainnet, arbitrum, polygon, base] as any,
        projectId,
        metadata,
        themeMode: "dark" as const,
        themeVariables: {
          "--w3m-accent": "#00d4aa",
        },
        features: {
          analytics: false,
        },
      });
      appKitInitialized = true;
    }
    setReady(true);
  }, []);

  if (!ready) return null;

  return (
    <WagmiProvider config={wagmiAdapter.wagmiConfig}>
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    </WagmiProvider>
  );
}
