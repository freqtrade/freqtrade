from __future__ import annotations

import math

from tools.ob_collector_ws import OrderBookBuilder


def test_ob_builder_snapshot_and_delta_sequence_ok():
    b = OrderBookBuilder(depth=3)
    # Snapshot
    bids = [["100", "2"], ["99", "3"], ["98", "4"]]
    asks = [["101", "2"], ["102", "3"], ["103", "4"]]
    b.apply_snapshot(bids=bids, asks=asks, update_id=10)
    assert b.state.update_id == 10
    assert b.needs_resync is False

    # Delta with correct next update id
    delta_bids = [["100", "0"]]  # remove best bid
    delta_asks = [["101", "5"]]  # change best ask qty
    b.apply_delta(bids=delta_bids, asks=delta_asks, update_id=11)
    assert b.state.update_id == 11
    assert b.needs_resync is False

    best_bid, best_ask, spread, mid, bid_vol, ask_vol, top_bid_qty, top_ask_qty = b.top_stats()
    assert math.isclose(best_bid, 99.0)
    assert math.isclose(best_ask, 101.0)
    assert math.isclose(spread, 2.0)
    assert math.isclose(mid, 100.0 + 0.0)
    assert top_bid_qty == 3.0
    assert top_ask_qty == 5.0
    assert bid_vol > 0 and ask_vol > 0


def test_ob_builder_gap_triggers_resync():
    b = OrderBookBuilder(depth=2)
    bids = [["100", "1"]]
    asks = [["101", "1"]]
    b.apply_snapshot(bids=bids, asks=asks, update_id=5)
    # Gap: next should be 6, but we send 8
    b.apply_delta(bids=[["100", "2"]], asks=None, update_id=8)
    assert b.needs_resync is True
