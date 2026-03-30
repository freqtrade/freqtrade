# TrendRider Platform — Architecture

## Overview

A SaaS platform where whitelisted users can run the TrendRider strategy on their own MEXC accounts. Users bring their API keys, we provide the proven strategy.

## Components

```
┌─────────────────────────────────────────────┐
│                  Frontend                     │
│          (Next.js + TailwindCSS)             │
│                                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
│  │  Auth     │ │ Dashboard│ │   Config     │ │
│  │ (Wallet)  │ │ (Trades) │ │ (MEXC Keys)  │ │
│  └──────────┘ └──────────┘ └──────────────┘ │
│  ┌──────────┐ ┌──────────────────────────┐   │
│  │  Admin   │ │  T&C / Disclaimers       │   │
│  │ (Whitelist│ │                          │   │
│  └──────────┘ └──────────────────────────┘   │
└───────────────────┬─────────────────────────┘
                    │ API
┌───────────────────▼─────────────────────────┐
│                  Backend                      │
│            (Node.js / Express)                │
│                                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
│  │  Auth    │ │  Bot     │ │   Admin      │ │
│  │  Service │ │  Manager │ │   Service    │ │
│  └──────────┘ └──────────┘ └──────────────┘ │
└───────────────────┬─────────────────────────┘
                    │
┌───────────────────▼─────────────────────────┐
│              Infrastructure                   │
│                                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ PostgreSQL│ │ Railway  │ │  Freqtrade   │ │
│  │ (Users,  │ │ (Deploy  │ │  (1 per user)│ │
│  │  Keys)   │ │  API)    │ │              │ │
│  └──────────┘ └──────────┘ └──────────────┘ │
└─────────────────────────────────────────────┘
```

## Security

### API Key Handling
- User MEXC API keys encrypted with AES-256 before storage
- Decrypted only in memory when starting a bot instance
- Keys never logged, never sent to frontend after submission
- Each bot runs in isolated container

### Authentication
- Wallet-gated: user connects MetaMask/WalletConnect
- Signs a message to prove ownership
- Admin whitelist: only approved wallet addresses can access
- JWT session tokens after wallet verification

### Admin
- Admin wallet addresses hardcoded in env
- Can add/remove wallet addresses from whitelist
- Can view all users, start/stop bots
- Cannot see user API keys (encrypted)

## Database Schema

### users
- id, wallet_address, display_name, is_whitelisted, is_admin
- created_at, last_login

### user_configs  
- id, user_id, mexc_api_key_encrypted, mexc_api_secret_encrypted
- stake_amount, max_open_trades, pairs (JSON)
- bot_status (stopped/running/error)
- created_at, updated_at

### user_trades
- id, user_id, trade_id, pair, open_rate, close_rate
- profit_pct, profit_abs, exit_reason
- open_date, close_date

### user_bot_instances
- id, user_id, railway_service_id, status
- started_at, stopped_at, last_heartbeat

## Pages

1. **/** — Landing page with T&C, disclaimers
2. **/connect** — Wallet connection
3. **/dashboard** — Trade overview (after auth)
4. **/config** — MEXC API key setup
5. **/admin** — Whitelist management (admin only)

## Tech Stack
- Frontend: Next.js 14, TailwindCSS, wagmi (wallet connect)
- Backend: Next.js API routes
- Database: PostgreSQL (Railway)
- Auth: SIWE (Sign-In with Ethereum)
- Encryption: AES-256-GCM for API keys
- Hosting: Railway or Vercel

## Deployment
- Frontend + API: single Next.js app on Railway
- Database: existing Railway PostgreSQL
- Bot per user: Railway service via API
- Domain: custom domain on Railway
