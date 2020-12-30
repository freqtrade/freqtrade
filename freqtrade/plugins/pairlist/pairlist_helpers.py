import re
from typing import List


def expand_pairlist(wildcardpl: List[str], available_pairs: List[str]) -> List[str]:
    """
    TODO: Add docstring here
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
