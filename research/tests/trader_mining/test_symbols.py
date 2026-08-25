from research.trader_mining.symbols import base_asset_of


def test_parses_base_from_slash_separated_symbol():
    assert base_asset_of("HYPE/USDC") == "HYPE"


def test_parses_base_from_perp_symbol_with_settle_suffix():
    assert base_asset_of("BTC/USDC:USDC") == "BTC"


def test_returns_none_for_unparsable_raw_index_symbol():
    assert base_asset_of("@705") is None
