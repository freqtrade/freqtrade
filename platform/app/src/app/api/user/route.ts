import { NextRequest, NextResponse } from "next/server";
import { query } from "@/utils/db";

export async function GET(req: NextRequest) {
  const wallet = req.headers.get("x-wallet-address")?.toLowerCase();
  if (!wallet) return NextResponse.json({ error: "No wallet" }, { status: 401 });

  const result = await query(
    `SELECT u.*, c."botStatus", c."stakeAmount", c."maxOpenTrades",
            c."mexcApiKeyEncrypted" IS NOT NULL as "hasApiKey"
     FROM platform_users u
     LEFT JOIN platform_configs c ON u.id = c."userId"
     WHERE LOWER(u."walletAddress") = $1`,
    [wallet]
  );

  if (result.rows.length === 0) {
    // Auto-register new user (not whitelisted by default)
    const insert = await query(
      `INSERT INTO platform_users ("walletAddress") VALUES ($1) RETURNING *`,
      [wallet]
    );
    return NextResponse.json({ user: insert.rows[0], config: null, trades: [] });
  }

  const user = result.rows[0];

  // Get user's trades from their bot instance
  let trades: any[] = [];
  const botResult = await query(
    `SELECT "railwayServiceId" FROM platform_bots WHERE "userId" = $1`,
    [user.id]
  );

  return NextResponse.json({
    user: {
      id: user.id,
      walletAddress: user.walletAddress,
      isWhitelisted: user.isWhitelisted,
      isAdmin: user.isAdmin,
      lastLogin: user.lastLogin,
    },
    config: {
      botStatus: user.botStatus || "stopped",
      stakeAmount: user.stakeAmount || 1,
      maxOpenTrades: user.maxOpenTrades || 10,
      hasApiKey: user.hasApiKey || false,
    },
    trades,
  });
}

export async function POST(req: NextRequest) {
  const wallet = req.headers.get("x-wallet-address")?.toLowerCase();
  if (!wallet) return NextResponse.json({ error: "No wallet" }, { status: 401 });

  // Update last login
  await query(
    `UPDATE platform_users SET "lastLogin" = NOW() WHERE LOWER("walletAddress") = $1`,
    [wallet]
  );

  return NextResponse.json({ ok: true });
}
