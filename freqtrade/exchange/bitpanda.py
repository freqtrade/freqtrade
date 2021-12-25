""" Bitpanda exchange subclass """
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from freqtrade.exchange import Exchange


logger = logging.getLogger(__name__)


class Bitpanda(Exchange):
    """
    Bitpanda exchange class. Contains adjustments needed for Freqtrade to work
    with this exchange.
    """

    def get_trades_for_order(self, order_id: str, pair: str, since: datetime,
                             params: Optional[Dict] = None) -> List:
        """
        Fetch Orders using the "fetch_my_trades" endpoint and filter them by order-id.
        The "since" argument passed in is coming from the database and is in UTC,
        as timezone-native datetime object.
        From the python documentation:
            > Naive datetime instances are assumed to represent local time
        Therefore, calling "since.timestamp()" will get the UTC timestamp, after applying the
        transformation from local timezone to UTC.
        This works for timezones UTC+ since then the result will contain trades from a few hours
        instead of from the last 5 seconds, however fails for UTC- timezones,
        since we're then asking for trades with a "since" argument in the future.

        :param order_id order_id: Order-id as given when creating the order
        :param pair: Pair the order is for
        :param since: datetime object of the order creation time. Assumes object is in UTC.
        """
        params = {'to': int(datetime.now(timezone.utc).timestamp() * 1000)}
        return super().get_trades_for_order(order_id, pair, since, params)
