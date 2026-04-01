const BASE = "";

export async function apiGet(path: string, wallet: string) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "x-wallet-address": wallet },
  });
  return res.json();
}

export async function apiPost(path: string, wallet: string, body: any) {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: {
      "x-wallet-address": wallet,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  return res.json();
}

export async function apiDelete(path: string, wallet: string, body: any) {
  const res = await fetch(`${BASE}${path}`, {
    method: "DELETE",
    headers: {
      "x-wallet-address": wallet,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  return res.json();
}
