from adapters.ccxt_shim.breeze_ccxt import BreezeCCXT


def test_mock_ohlcv_has_bounded_moves() -> None:
    exchange = BreezeCCXT({"mode": "mock"})
    candles = exchange._generate_mock_ohlcv("RELIANCE/INR", "5m", since=0, limit=20)

    assert len(candles) > 2

    max_move = exchange._MOCK_MAX_PCT_MOVE
    previous_close = candles[0][4]
    for candle in candles[1:]:
        open_price, high_price, low_price, close_price = candle[1:5]
        assert abs((close_price - previous_close) / previous_close) <= max_move
        assert high_price >= max(open_price, close_price)
        assert low_price <= min(open_price, close_price)
        previous_close = close_price
