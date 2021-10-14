"""
Announcements PairList provider

Provides dynamic pair list based on exchanges announcements.

Supported exchanges:
- Binance

"""
import logging
import re
from abc import abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytz
from bs4 import BeautifulSoup
from requests import get

import pandas as pd
from cachetools.ttl import TTLCache

from freqtrade.exceptions import OperationalException
from freqtrade.plugins.pairlist.IPairList import IPairList


logger = logging.getLogger(__name__)


# for futures updates
SORT_VALUES = ['exchange']


class BinanceAnnouncementMixin:
    BINANCE_CATALOG_ID = 48
    BINANCE_BASE_URL = "https://www.binance.com/"
    BINANCE_ANNOUNCEMENT_URL = BINANCE_BASE_URL + 'en/support/announcement/'
    BINANCE_ANNOUNCEMENTS_URL = BINANCE_ANNOUNCEMENT_URL + "c-{}?navId={}#/{}"
    BINANCE_API_QUERY = "query?catalogId={}&pageNo={}&pageSize={}"
    BINANCE_API_URL = BINANCE_BASE_URL + "bapi/composite/v1/public/cms/article/catalog/list/" + BINANCE_API_QUERY

    # css classes
    BINANCE_DATETIME_CSS_CLASS = 'css-17s7mnd'

    # token info
    BINANCE_TOKEN_REGEX = re.compile(r'\((\w+)\)')
    BINANCE_KEY_WORDS = ['list', ]  # 'token sale', 'perpetual', 'open trading',

    # 'opens trading',  'defi', 'uniswap', 'airdrop'
    BINANCE_KEY_WORDS_BLACKLIST = ['listing postponed', 'futures', 'leveraged']

    REFRESH_PERIOD = 3

    # storage
    COLS = ['Token', 'Text', 'Link', 'Datetime discover', 'Datetime announcement']
    DB = "BinanceAnnouncements_announcements.csv"

    _last_update: Optional[datetime] = None
    _df: Optional[pd.DataFrame] = None

    def update_binance_announcements(self, page_number=1, page_size=10, history=False):
        response = None
        url = self.get_api_url(page_number, page_size)

        if history:
            # recursive updating
            return [self.update_binance_announcements(
                page, page_size, history=False
            ) for page in reversed(range(2, 56))][-1]

        try:
            now = datetime.now(tz=pytz.utc)
            df = self._get_df()

            response = get(url)

            if response.status_code != 200:
                raise OperationalException(f"Invalid response from url: {url}.\n"
                                           f"Status code: {response.status_code}\n"
                                           f"Content: {response.content.decode()}")

            self.last_update = now
            updated_list = []

            for article in response.json()['data']['articles']:
                article_link = self.get_announcement_url(article['code'])
                article_text = article['title']

                tokens = self._get_tokens(article_text)

                if not tokens:
                    token = self.get_token_by_article(article_link, raise_exceptions=False)
                    if token:
                        tokens = [token]

                for token in tokens:
                    if token:
                        updated_list.extend(
                            self._get_new_data(
                                now=now,
                                token=token,
                                key_words=self.BINANCE_KEY_WORDS,
                                article_text_lower=article_text.lower(),
                                article_link=article_link,
                                article_text=article_text,
                                df=df
                            )
                        )

            if df is not None:
                df = df.append(pd.DataFrame(updated_list, columns=self.COLS), ignore_index=True)
            else:
                df = pd.DataFrame(updated_list, columns=self.COLS)

            if updated_list:
                msg = f"Adding tokens to database: {[upd[0] for upd in updated_list]}"
                logger.info(msg)
                self._save_df(df)
                self.notify_user(msg)
            return df

        except Exception as e:
            logger.error(e)
            raise OperationalException(f"Some errors occurred processing Binance data. "
                                       f"Url: {url}.\n"
                                       f"Status code: {response.status_code if response else None}\n"
                                       f"Content: {response.content.decode() if response else None}\n"
                                       f"Exception: {e}")

    def _get_new_data(self, now, token, key_words, article_text_lower, article_link, article_text, df=None):
        have_df = df is not None
        updated_list = []
        for item in key_words:
            conditions_buy = (
                (item in article_text_lower)  # key matched
                and (
                    not have_df  # is first time data
                    or not (token is None or token in df['Token'].values)  # not an existing or null token
                )
            )
            if conditions_buy:
                if any(i in article_text_lower for i in self.BINANCE_KEY_WORDS_BLACKLIST):
                    logger.debug(f'BLACKLISTED: "{article_text}", skip.')
                    continue

                if token:
                    logger.info(f'Found new announcement: "{article_text}". Token: {token}.')
                    updated_list.append(
                        [token, article_text, article_link, now, self.get_datetime_announcement(article_link)]
                    )
        return updated_list

    def _get_tokens(self, text: str):
        return self.BINANCE_TOKEN_REGEX.findall(text)

    def _get_df(self):
        if self._df is None:
            try:
                self._df = pd.read_csv(self.db_path, parse_dates=['Datetime announcement', 'Datetime discover'])
            except FileNotFoundError:
                pass
        return self._df

    def _save_df(self, df: pd.DataFrame):
        self._df = df.sort_values(by='Datetime announcement')
        self._df.to_csv(self.db_path, index=False)

    @property
    def db_path(self) -> str:
        return "".join(["user_data/data/", self.DB])

    def get_datetime_announcement(self, announcement_url: str):
        response = get(announcement_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        for el in soup.find_all(class_=self.BINANCE_DATETIME_CSS_CLASS):
            try:
                return datetime.strptime(el.text, '%Y-%m-%d %H:%M').replace(tzinfo=pytz.utc)
            except Exception as e:
                logger.error(e)
                continue

        msg = f"Cannot find datetime_announcement in announcement_url: {announcement_url}. " \
              f"Probably a CSS class change."

        self.notify_user(msg)
        exc = OperationalException(msg)
        logger.error(exc)

    def get_token_by_article(self, article_link, raise_exceptions: bool = True):
        # TODO
        if raise_exceptions:
            raise ValueError("Token not found")

    @abstractmethod
    def notify_user(self, message: str):
        # # notify user using WARNING message
        # if self._freqtrade_bot is not None:
        #     # monkey patching id DONE
        #     try:
        #         self._freqtrade_bot.rpc.send_msg({
        #             'type': RPCMessageType.WARNING,
        #             'status': message
        #         })
        #         return
        #     except Exception as e:
        #         logger.error(e)
        logger.error(f"Message {message} cannot be sent to user.")

    @property
    def last_update(self) -> datetime:
        if not self._last_update:
            try:
                with open('user_data/data/._last_update', 'r') as f:
                    self._last_update = datetime.fromtimestamp(float(f.read()), tz=pytz.utc)
            except FileNotFoundError:
                self.last_update = datetime.now(tz=pytz.utc) - timedelta(seconds=self.REFRESH_PERIOD * 2)
            return self._last_update

    @last_update.setter
    def last_update(self, value: datetime) -> None:
        with open('user_data/data/._last_update', 'w') as f:
            f.write(str(value.timestamp()))
            self._last_update = value

    def get_announcement_url(self, code: str) -> str:
        return "".join([self.BINANCE_ANNOUNCEMENT_URL, code])

    @property
    def announcements_url(self) -> str:
        return self.BINANCE_ANNOUNCEMENTS_URL.format(*[self.BINANCE_CATALOG_ID for _ in range(3)])

    @staticmethod
    def get_token_from_pair(pair: str, index: int = 0) -> str:
        return pair.split('/')[index]

    def get_api_url(self, page_number: int = 1, page_size: int = 10) -> str:
        return self.BINANCE_API_URL.format(self.BINANCE_CATALOG_ID, page_number, page_size)


class AnnouncementsPairList(IPairList, BinanceAnnouncementMixin):

    # sleep at least 3 seconds every request
    REFRESH_PERIOD = 3

    def __init__(self, exchange, pairlistmanager,
                 config: Dict[str, Any], pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:

        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

        self._stake_currency = config['stake_currency']
        self._hours = self._pairlistconfig.get('hours', 24)
        self._refresh_period = self._pairlistconfig.get('refresh_period', self.REFRESH_PERIOD)
        self._pair_cache: TTLCache = TTLCache(maxsize=1, ttl=self._refresh_period)

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
        return f"{self.name} - Binance exchange announced pairs."

    def gen_pairlist(self, tickers: Dict) -> List[str]:
        """
        Generate the pairlist
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: List of pairs
        """
        pairlist = self._pair_cache.get('pairlist')
        if pairlist:
            # Item found - no refresh necessary
            return pairlist.copy()
        else:
            filtered_tickers = [
                v for k, v in tickers.items()
                if (self._exchange.get_pair_quote_currency(k) == self._stake_currency)
             ]

            pairlist = [s['symbol'] for s in filtered_tickers]
            pairlist = self.filter_pairlist(pairlist, tickers)
            self._pair_cache['pairlist'] = pairlist.copy()

        return pairlist

    def filter_pairlist(self, pairlist: List[str], tickers: Dict) -> List[str]:
        """
        Filters and sorts pairlist and returns the whitelist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: new whitelist
        """
        logger.info("Updating from Binance ...")
        df = self.update_binance_announcements()
        # TODO migliorare l'efficienza del calcolo
        pairlist = [
            v for v in pairlist if not df[
                (df['Token'] == self.get_token_from_pair(v)) &
                (df['Datetime announcement'] > datetime.now().replace(tzinfo=pytz.utc) - timedelta(hours=self._hours))
            ].empty
        ]
        return pairlist

    def notify_user(self, msg):
        # TODO
        pass
