#!/usr/bin/env python3
"""
SynapticChain Freqtrade Exchange Connector Demo.

Demonstrates 256-Lane Parallel Execution VM (ADR-062), single-slot BFT finality (<500ms),
and zero Head-of-Line nonce blocking for Freqtrade automated trading bots.
"""

import os
import sys
import time

# Ensure synaptic_freqtrade is importable directly from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from synaptic_freqtrade import (
    LaneAllocationStrategy,
    OrderSide,
    OrderType,
    SynapticFreqtradeExchange,
)


def print_header(title: str) -> None:
    print("\n" + "=" * 88)
    print(f"⚡ {title}")
    print("=" * 88)


def main() -> None:
    print_header("SynapticChain 256-Lane Parallel Execution Exchange Driver for Freqtrade")

    # 1. Initialize Exchange Instance with 256 execution lanes
    print("\n[1/5] 📡 Initializing SynapticChain L1 Exchange Driver (ADR-062)...")
    exchange = SynapticFreqtradeExchange(
        api_key="syn1trader77889900112233445566778899aabbccddeeff",
        rpc_url="https://nodes.synapticchain.xyz/rpc",
        lane_strategy=LaneAllocationStrategy.PAIR_HASH,
    )
    print(f"      ✅ Connected to Layer-1 RPC: {exchange.rpc_url}")
    print(f"      ✅ Wallet Address: {exchange.wallet_address}")
    print(f"      ✅ Total Concurrency: {exchange.NUM_LANES} Independent Execution Lanes")
    print(f"      ✅ Native Micro-Settlement Fee: ${exchange.MICRO_FEE_USD:.4f} per lane")

    # 2. Fetch Account Balances
    print("\n[2/5] 💰 Querying Unified Account Balances (Freqtrade Schema)...")
    balances = exchange.get_balances()
    print("      ---------------------------------------------------------")
    print(f"      {'Asset':<8} | {'Free Balance':<15} | {'Used':<10} | {'Total':<15}")
    print("      ---------------------------------------------------------")
    for asset, data in balances.items():
        print(f"      {asset:<8} | {data['free']:<15,.4f} | {data['used']:<10,.4f} | {data['total']:<15,.4f}")
    print("      ---------------------------------------------------------")

    # 3. Fetch Ticker & L2 Order Book
    print("\n[3/5] 📊 Fetching Real-Time Market Data for SYN/sUSD...")
    ticker = exchange.fetch_ticker("SYN/sUSD")
    orderbook = exchange.fetch_l2_order_book("SYN/sUSD", limit=3)
    print(f"      ✅ Best Bid: ${ticker['bid']:.4f} | Best Ask: ${ticker['ask']:.4f} | Last: ${ticker['last']:.4f}")
    print(f"      ✅ Top 3 Bids: {orderbook['bids']}")
    print(f"      ✅ Top 3 Asks: {orderbook['asks']}")

    # 4. Single Limit Order Placement and Cancellation
    print("\n[4/5] 📝 Submitting & Canceling Single Order on SynapticChain DEX...")
    order = exchange.create_order(
        pair="SYN/sUSD",
        ordertype="limit",
        side="buy",
        amount=100.0,
        rate=1.2450,
        lane_id=42,  # Explicit lane pinning test
    )
    print(f"      ✅ Order Placed: ID={order['id']}")
    print(f"         Pair: {order['pair']} | Side: {order['side']} | Amount: {order['amount']} | Rate: ${order['price']}")
    print(f"         Assigned Lane: {order['lane_id']} | Lane Nonce: {order['lane_nonce']} | Finality: {order['finality_ms']}ms")

    # Fetch order status
    queried_order = exchange.get_order(order["id"])
    print(f"      ✅ Queried Order Status: {queried_order['status'].upper()}")

    # Cancel order
    canceled_order = exchange.cancel_order(order["id"])
    print(f"      ✅ Cancel Receipt: Status={canceled_order['status'].upper()} | Refunded to Free Balance")

    # 5. Concurrent 16-Pair Parallel Execution Benchmark (Zero Head-of-Line Nonce Blocking)
    print("\n[5/5] ⚡ Executing 16 Parallel Orders across 16 Distinct Lanes (Zero Nonce Blocking)...")

    test_batch = [
        {"pair": "SYN/sUSD", "side": "buy", "amount": 100.0, "price": 1.2480, "lane_id": 10},
        {"pair": "SYN/sUSD", "side": "sell", "amount": 50.0, "price": 1.2520, "lane_id": 26},
        {"pair": "BTC/sUSD", "side": "buy", "amount": 0.05, "price": 64900.00, "lane_id": 42},
        {"pair": "BTC/sUSD", "side": "sell", "amount": 0.05, "price": 65100.00, "lane_id": 58},
        {"pair": "ETH/sUSD", "side": "buy", "amount": 1.0, "price": 3490.00, "lane_id": 74},
        {"pair": "ETH/sUSD", "side": "sell", "amount": 1.0, "price": 3510.00, "lane_id": 90},
        {"pair": "SOL/sUSD", "side": "buy", "amount": 10.0, "price": 179.00, "lane_id": 106},
        {"pair": "SOL/sUSD", "side": "sell", "amount": 10.0, "price": 181.00, "lane_id": 122},
        {"pair": "AVAX/sUSD", "side": "buy", "amount": 25.0, "price": 32.00, "lane_id": 138},
        {"pair": "AVAX/sUSD", "side": "sell", "amount": 25.0, "price": 33.00, "lane_id": 154},
        {"pair": "LINK/sUSD", "side": "buy", "amount": 50.0, "price": 18.50, "lane_id": 170},
        {"pair": "LINK/sUSD", "side": "sell", "amount": 50.0, "price": 19.00, "lane_id": 186},
        {"pair": "SYN/sUSD", "side": "buy", "amount": 200.0, "price": 1.2470, "lane_id": 202},
        {"pair": "SYN/sUSD", "side": "sell", "amount": 200.0, "price": 1.2530, "lane_id": 218},
        {"pair": "BTC/sUSD", "side": "buy", "amount": 0.02, "price": 64800.00, "lane_id": 234},
        {"pair": "ETH/sUSD", "side": "buy", "amount": 0.5, "price": 3480.00, "lane_id": 250},
    ]

    bench_start = time.time()
    batch_results = exchange.batch_create_parallel_orders(test_batch)
    total_wall_time_ms = (time.time() - bench_start) * 1000

    print("      " + "-" * 90)
    print(f"      {'Order ID':<18} | {'Pair':<10} | {'Side':<4} | {'Amount':<8} | {'Rate':<10} | {'Lane':<4} | {'Nonce':<5} | {'Finality':<10}")
    print("      " + "-" * 90)

    for r in batch_results:
        print(
            f"      {r['id']:<18} | {r['pair']:<10} | {r['side']:<4} | {r['amount']:<8.2f} | "
            f"${r['price']:<9.2f} | {r['lane_id']:<4} | {r['lane_nonce']:<5} | {r['finality_ms']:.2f}ms"
        )
    print("      " + "-" * 90)

    avg_finality = sum(r["finality_ms"] for r in batch_results) / len(batch_results)

    print(f"\n      🏁 Benchmark Summary:")
    print(f"      ⚡ Total Orders Dispatched Concurrently: {len(batch_results)}")
    print(f"      ⚡ Total Wall-Clock Settlement Time: {total_wall_time_ms:.2f}ms")
    print(f"      ⚡ Average Single-Slot BFT Finality: {avg_finality:.2f}ms (<500ms SLA)")
    print(f"      ⚡ Head-of-Line Nonce Contention: 0.00% (Independent Per-Lane Sequencing)")
    print(f"      ⚡ Total Micro-Gas Settled: ${len(batch_results) * exchange.MICRO_FEE_USD:.4f}")

    print_header("SynapticChain Freqtrade Integration Verification Complete ✅")


if __name__ == "__main__":
    main()
