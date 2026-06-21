"""Symbol conversion helpers for XCoin spot markets."""


def xcoin_symbol_to_ccxt(symbol: str) -> str:
    """Convert XCoin symbol format to Freqtrade/ccxt spot symbol format."""
    if not symbol:
        return symbol
    if "/" in symbol:
        return symbol
    base, quote, *_ = symbol.split("-")
    return f"{base}/{quote}"


def ccxt_symbol_to_xcoin(symbol: str) -> str:
    """Convert Freqtrade/ccxt spot symbol format to XCoin symbol format."""
    if "-" in symbol and "/" not in symbol:
        return symbol
    return symbol.replace("/", "-")
