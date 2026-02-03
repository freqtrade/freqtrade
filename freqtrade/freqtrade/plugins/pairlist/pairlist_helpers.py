import re

from freqtrade.constants import Config


def expand_pairlist(
    wildcardpl: list[str], available_pairs: list[str], keep_invalid: bool = False
) -> list[str]:
    """
    Expand pairlist potentially containing wildcards based on available markets.
    This will implicitly filter all pairs in the wildcard-list which are not in available_pairs.
    :param wildcardpl: List of Pairlists, which may contain regex
    :param available_pairs: List of all available pairs (`exchange.get_markets().keys()`)
    :param keep_invalid: If sets to True, drops invalid pairs silently while expanding regexes
    :return: expanded pairlist, with Regexes from wildcardpl applied to match all available pairs.
    :raises: ValueError if a wildcard is invalid (like '*/BTC' - which should be `.*/BTC`)
    """
    result = []
    if keep_invalid:
        for pair_wc in wildcardpl:
            try:
                comp = re.compile(pair_wc, re.IGNORECASE)
                result_partial = [pair for pair in available_pairs if re.fullmatch(comp, pair)]
                # Add all matching pairs.
                # If there are no matching pairs (Pair not on exchange) keep it.
                result += result_partial or [pair_wc]
            except re.error as err:
                raise ValueError(f"Wildcard error in {pair_wc}, {err}")

        # Remove wildcard pairs that didn't have a match.
        result = [element for element in result if re.fullmatch(r"^[A-Za-z0-9:/-]+$", element)]

    else:
        for pair_wc in wildcardpl:
            try:
                comp = re.compile(pair_wc, re.IGNORECASE)
                result += [pair for pair in available_pairs if re.fullmatch(comp, pair)]
            except re.error as err:
                raise ValueError(f"Wildcard error in {pair_wc}, {err}")
    return result


def dynamic_expand_pairlist(config: Config, markets: list[str]) -> list[str]:
    expanded_pairs = expand_pairlist(config["pairs"], markets)
    if config.get("freqai", {}).get("enabled", False):
        corr_pairlist = config["freqai"]["feature_parameters"]["include_corr_pairlist"]
        expanded_pairs += [pair for pair in corr_pairlist if pair not in config["pairs"]]

    return expanded_pairs
