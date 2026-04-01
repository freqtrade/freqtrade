import { NextRequest, NextResponse } from "next/server";
import { query } from "@/utils/db";
import { encrypt } from "@/utils/encryption";

export async function POST(req: NextRequest) {
  const wallet = req.headers.get("x-wallet-address")?.toLowerCase();
  if (!wallet) return NextResponse.json({ error: "No wallet" }, { status: 401 });

  const body = await req.json();
  const { apiKey, apiSecret, stakeAmount, maxOpenTrades } = body;

  // Get user
  const userResult = await query(
    `SELECT id, "isWhitelisted" FROM platform_users WHERE LOWER("walletAddress") = $1`,
    [wallet]
  );

  if (userResult.rows.length === 0) {
    return NextResponse.json({ error: "User not found" }, { status: 404 });
  }

  const user = userResult.rows[0];

  if (!user.isWhitelisted) {
    return NextResponse.json({ error: "Not whitelisted" }, { status: 403 });
  }

  // Encrypt API keys
  const encryptedKey = apiKey ? encrypt(apiKey) : undefined;
  const encryptedSecret = apiSecret ? encrypt(apiSecret) : undefined;

  // Upsert config
  const existing = await query(
    `SELECT id FROM platform_configs WHERE "userId" = $1`,
    [user.id]
  );

  if (existing.rows.length > 0) {
    const updates: string[] = [];
    const values: any[] = [];
    let idx = 1;

    if (encryptedKey) {
      updates.push(`"mexcApiKeyEncrypted" = $${idx++}`);
      values.push(encryptedKey);
    }
    if (encryptedSecret) {
      updates.push(`"mexcApiSecretEncrypted" = $${idx++}`);
      values.push(encryptedSecret);
    }
    if (stakeAmount) {
      updates.push(`"stakeAmount" = $${idx++}`);
      values.push(stakeAmount);
    }
    if (maxOpenTrades) {
      updates.push(`"maxOpenTrades" = $${idx++}`);
      values.push(maxOpenTrades);
    }
    updates.push(`"updatedAt" = NOW()`);

    if (updates.length > 1) {
      values.push(user.id);
      await query(
        `UPDATE platform_configs SET ${updates.join(", ")} WHERE "userId" = $${idx}`,
        values
      );
    }
  } else {
    await query(
      `INSERT INTO platform_configs ("userId", "mexcApiKeyEncrypted", "mexcApiSecretEncrypted", "stakeAmount", "maxOpenTrades")
       VALUES ($1, $2, $3, $4, $5)`,
      [user.id, encryptedKey || null, encryptedSecret || null, stakeAmount || 1, maxOpenTrades || 10]
    );
  }

  return NextResponse.json({ ok: true, message: "Config saved" });
}
