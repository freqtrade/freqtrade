from typing import Tuple


def get_sides(is_short: bool) -> Tuple[str, str]:
    return ("sell", "buy") if is_short else ("buy", "sell")
