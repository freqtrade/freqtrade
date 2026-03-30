import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TrendRider — Automated Crypto Trading",
  description: "Run a proven trading strategy on your MEXC account",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-dark-900">{children}</body>
    </html>
  );
}
