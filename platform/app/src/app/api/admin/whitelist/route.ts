import { NextRequest, NextResponse } from "next/server";
import { query } from "@/utils/db";

const ADMIN_WALLETS = ["0xaedb312d90fa956775ea8abed298ea3b085abbd9"];

function isAdmin(wallet: string): boolean {
  return ADMIN_WALLETS.includes(wallet.toLowerCase());
}

// GET: list all users
export async function GET(req: NextRequest) {
  const wallet = req.headers.get("x-wallet-address")?.toLowerCase();
  if (!wallet || !isAdmin(wallet)) {
    return NextResponse.json({ error: "Admin only" }, { status: 403 });
  }

  const result = await query(`
    SELECT u.*, c."botStatus", c."stakeAmount",
           c."mexcApiKeyEncrypted" IS NOT NULL as "hasApiKey",
           b.status as "botRunning"
    FROM platform_users u
    LEFT JOIN platform_configs c ON u.id = c."userId"
    LEFT JOIN platform_bots b ON u.id = b."userId"
    ORDER BY u."createdAt" DESC
  `);

  return NextResponse.json({ users: result.rows });
}

// POST: whitelist/unwhitelist a wallet
export async function POST(req: NextRequest) {
  const wallet = req.headers.get("x-wallet-address")?.toLowerCase();
  if (!wallet || !isAdmin(wallet)) {
    return NextResponse.json({ error: "Admin only" }, { status: 403 });
  }

  const { targetWallet, whitelist } = await req.json();

  if (!targetWallet) {
    return NextResponse.json({ error: "No wallet specified" }, { status: 400 });
  }

  const target = targetWallet.toLowerCase();

  // Create user if doesn't exist
  await query(
    `INSERT INTO platform_users ("walletAddress", "isWhitelisted")
     VALUES ($1, $2)
     ON CONFLICT ("walletAddress") DO UPDATE SET "isWhitelisted" = $2`,
    [target, whitelist !== false]
  );

  return NextResponse.json({ ok: true, wallet: target, whitelisted: whitelist !== false });
}

// DELETE: remove from whitelist
export async function DELETE(req: NextRequest) {
  const wallet = req.headers.get("x-wallet-address")?.toLowerCase();
  if (!wallet || !isAdmin(wallet)) {
    return NextResponse.json({ error: "Admin only" }, { status: 403 });
  }

  const { targetWallet } = await req.json();
  if (!targetWallet) {
    return NextResponse.json({ error: "No wallet specified" }, { status: 400 });
  }

  await query(
    `UPDATE platform_users SET "isWhitelisted" = false WHERE LOWER("walletAddress") = $1`,
    [targetWallet.toLowerCase()]
  );

  return NextResponse.json({ ok: true, removed: targetWallet });
}
