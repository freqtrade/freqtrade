import re
from typing import List


def expand_pairlist(wildcardpl: List[str], available_pairs: List[str]) -> List[str]:
    """
    Expand pairlist potentially containing wildcards based on available markets.
    This will implicitly filter all pairs in the wildcard-list which are not in available_pairs.
    :param wildcardpl: List of Pairlists, which may contain regex
    :param available_pairs: List of all available pairs (`exchange.get_markets().keys()`)
    :return expanded pairlist, with Regexes from wildcardpl applied to match all available pairs.
    :raises: ValueError if a wildcard is invalid (like '*/BTC' - which should be `.*/BTC`)
    """
    result = []
    for pair_wc in wildcardpl:
        try:
            comp = re.compile(pair_wc)
            result += [
                pair for pair in available_pairs if re.match(comp, pair)
            ]
        except re.error as err:
            raise ValueError(f"Wildcard error in {pair_wc}, {err}")
    return result
