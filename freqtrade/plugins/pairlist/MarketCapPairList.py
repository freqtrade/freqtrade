"""
MarketCap PairList provider

Provides dynamic pair list based on market cap
"""
import logging
from typing import Any, Dict, List

from cachetools.ttl import TTLCache
from pandas.core.common import flatten
from pycoingecko import CoinGeckoAPI

from freqtrade.exceptions import OperationalException
from freqtrade.misc import chunks
from freqtrade.plugins.pairlist.IPairList import IPairList


logger = logging.getLogger(__name__)


class MarketCapPairList(IPairList):
    def __init__(self, exchange, pairlistmanager,
                 config: Dict[str, Any], pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

        if 'number_assets' not in self._pairlistconfig:
            raise OperationalException(
                '`number_assets` not specified. Please check your configuration '
                'for "pairlist.config.number_assets"')
        if (not type(self._pairlistconfig['number_assets']) is int) or (
                int(self._pairlistconfig['number_assets']) <= 0):
            raise OperationalException(
                '"number_assets" should be a positive integer. '
                'Please edit your config and restart the bot.'
            )
        self._number_pairs = int(self._pairlistconfig['number_assets'])

        self._stake_currency = config['stake_currency']
        self._refresh_period = self._pairlistconfig.get('refresh_period', 6*60*60)
        self._marketcap_cache: TTLCache = TTLCache(maxsize=1, ttl=self._refresh_period)
        self._cg = CoinGeckoAPI()
        self._marketcap_ranks: Dict[str, Any] = {}
        self._coins_list: Dict[str, str] = {}

        if not self._exchange.exchange_has('fetchTickers'):
            raise OperationalException(
                'Exchange does not support dynamic whitelist. '
                'Please edit your config and restart the bot.'
            )

    @property
    def needstickers(self) -> bool:
        """
        Boolean property defining if tickers are necessary.
        If no Pairlist requires tickers, an empty Dict is passed
        as tickers argument to filter_pairlist
        """
        return True

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        """
        return f"{self.name} - top {self._pairlistconfig['number_assets']} marketcap pairs."

    def _request_coins_list(self):
        """
        Update the symbol -> coin id mapping. Coin ids are needed to request market data
        for single coins.
        """
        coins_list = self._cg.get_coins_list()
        for x in coins_list:
            # yes, lower() is needed as there are a few uppercase symbols
            symbol = x['symbol'].lower()
            cid = x['id']

            # ignore binance-peg-
            if 'binance-peg-' in cid:
                continue
            # ignore -bsc
            if cid.endswith('-bsc'):
                continue
            # yes, not all symbols are unique.
            if symbol in self._coins_list:
                # in a few cases the id matches the symbol
                # we have that version already, keep it.
                if self._coins_list[symbol] == symbol:
                    continue
                # the new one matches, use it.
                elif cid == symbol:
                    self._coins_list[symbol]
                else:
                    self._coins_list[symbol] = list(flatten([self._coins_list[symbol], cid]))
            else:
                self._coins_list[symbol] = cid

    def _request_marketcap_ranks(self):
        """
        To keep the request count low, download the most common coins.
        Then update all coins that where not in the top 250 ranks.
        """
        ids = []

        # there might be some stuff to add here..
        # probably also needs to be configurable and/or have
        # exchange specific settings
        symbol_map = {
                'acm': 'ac-milan-fan-token',
                'ant': 'aragon',
                'atm': 'atletico-madrid',
                'bat': 'basic-attention-token',
                'bts': 'bitshares',
                'comp': 'compound-governance-token',
                'dego': 'dego-finance',
                'eps': 'ellipsis',
                'grt': 'the-graph',
                'hnt': 'helium',
                'hot': 'holotoken',
                'iota': 'miota',
                'lit': 'litentry',
                'luna': 'terra-luna',
                'mask': 'mask-network',
                'mdx': 'mdex',
                'mir': 'mirror-protocol',
                'og': 'og-fan-token',
                'one': 'harmony',
                'pax': 'paxos-standard',
                'pnt': 'pnetwork',
                'rune': 'thorchain',
                'sand': 'the-sandbox',
                'stx': 'blockstack',
                'super': 'superfarm',
                'tct': 'tokenclub',
                'trb': 'tellor',
                'tru': 'truefi',
                'trx': 'tron',
                'uni': 'uniswap',
                'wing': 'wing-finance',
                }

        for symbol in self._marketcap_ranks.keys():
            # symbol needs mapping
            if symbol in symbol_map:
                ids.append(symbol_map[symbol])
            elif symbol in self._coins_list:
                _id = self._coins_list[symbol]
                # symbol is not unique (sigh!)
                if type(_id) == list:
                    pair = symbol.upper() + f'/{self._stake_currency}'
                    _id_text = ', '.join(_id)
                    self.log_once(f'Symbol for {pair} is not unique on coingecko '
                                  f'({_id_text}), dropping.', logger.warning)
                    continue
                ids.append(_id)
            elif symbol in self._coins_list.values():
                # try to add some symbols automatically
                symbol_map[symbol] = [k for k, v in self._coins_list.items() if v == symbol][0]
                ids.append(symbol)

        # reverse map
        rev_symbol_map = {v: k for k, v in symbol_map.items()}

        _marketcap_ranks = {}

        # seems coingecko limits the number of ids to 52, using 50.
        for _ids in chunks(ids, 50):
            base_marketkaps = self._cg.get_coins_markets(
                    vs_currency='usd',
                    ids=','.join(_ids),
                    order='market_cap_desc',
                    per_page=len(ids),
                    sparkline=False,
                    page=1)
            for x in base_marketkaps:
                # same here, keep symbol lowercase.
                _symbol = x['symbol'].lower()
                if _symbol in rev_symbol_map:
                    _symbol = rev_symbol_map[_symbol]
                _marketcap_ranks[_symbol] = x['market_cap_rank']

        for x in self._marketcap_ranks.keys():
            if x not in _marketcap_ranks:
                _marketcap_ranks[x] = None
        self._marketcap_ranks = _marketcap_ranks

    def update_marketcap_ranks(self, symbols: List[str]):
        """
        Updates the dict containing the marketcap ranks of the requested
        list of symbols, if needed.
        """
        marketcaps_uptodate = self._marketcap_cache.get('marketcaps_uptodate')
        startup = not (self._marketcap_ranks and self._coins_list)

        for symbol in symbols:
            if symbol.lower() not in self._marketcap_ranks:
                marketcaps_uptodate = False
                self._marketcap_ranks[symbol.lower()] = None
        if not marketcaps_uptodate:
            try:
                self._request_coins_list()
                self._request_marketcap_ranks()
            except Exception as e:
                if startup:
                    raise OperationalException(
                        f'Failed to download marketcap data from coingecko: {e}'
                    )
                else:
                    self.log_once(
                            'Failed to update marketcap data from coingecko: .'
                            f'{e}. Using old data.',
                            logger.warning)
            else:
                self._marketcap_cache['marketcaps_uptodate'] = True

    def gen_pairlist(self, tickers: Dict) -> List[str]:
        """
        Generate the pairlist
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: List of pairs
        """

        # using the pairlist from the exchange works, but if the list is long, it will create a very
        # long startup time due to coingecko rate limits. Might be worth to add an config option
        # with a big warning.
        pairlist = self._whitelist_for_active_markets(
               self.verify_whitelist(self._config['exchange']['pair_whitelist'], logger.info))
        return self.filter_pairlist(pairlist, tickers)

    def filter_pairlist(self, pairlist: List[str], tickers: Dict) -> List[str]:
        """
        Filters and sorts pairlist and returns the whitelist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: new whitelist
        """
        symbols = [p.replace(f'/{self._stake_currency}', '').lower() for p in pairlist]
        self.update_marketcap_ranks(symbols)

        marketcap_ranks = {}
        for symbol in symbols:
            rank = self._marketcap_ranks[symbol]
            if not rank:
                pair = symbol.upper() + f'/{self._stake_currency}'
                self.log_once(f'No known marketcap for {pair}, dropping.', logger.warning)
                continue
            marketcap_ranks[symbol] = rank

        # sort marketcaps
        pairlist = [
                k.upper() + f'/{self._stake_currency}'
                for k, v in sorted(marketcap_ranks.items(), key=lambda x: x[1])
                ]

        # Validate whitelist to only have active market pairs
        pairlist = self._whitelist_for_active_markets(pairlist)
        pairlist = self.verify_blacklist(pairlist, logger.info)
        # Limit pairlist to the requested number of pairs
        pairlist = pairlist[:self._number_pairs]
        self.log_once(f"Searching {self._number_pairs} pairs: {pairlist}", logger.info)

        return pairlist
