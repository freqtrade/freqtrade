from freqtrade.data.dataprovider import DataProvider
from freqtrade.plugins.pairlist.pairlist_helpers import dynamic_expand_pairlist
from freqtrade.exchange.exchange import market_is_active
from freqtrade.exchange import timeframe_to_seconds
from freqtrade.data.history.history_utils import refresh_backtest_ohlcv_data
from datetime import datetime, timezone
from freqtrade.exceptions import OperationalException
from freqtrade.configuration import TimeRange


def download_all_data_for_training(dp: DataProvider, config: dict) -> None:
    """
    Called only once upon start of bot to download the necessary data for
    populating indicators and training a FreqAI model.
    :param timerange: TimeRange = The full data timerange for populating the indicators
                                    and training the model.
    :param dp: DataProvider instance attached to the strategy
    """

    if dp._exchange is not None:
        markets = [p for p, m in dp._exchange.markets.items() if market_is_active(m)
                   or config.get('include_inactive')]
    else:
        # This should not occur:
        raise OperationalException('No exchange object found.')

    all_pairs = dynamic_expand_pairlist(config, markets)

    if not dp._exchange:
        # Not realistic - this is only called in live mode.
        raise OperationalException("Dataprovider did not have an exchange attached.")

    time = datetime.now(tz=timezone.utc).timestamp()

    for tf in config["freqai"]["feature_parameters"].get("include_timeframes"):
        timerange = TimeRange()
        timerange.startts = int(time)
        timerange.stopts = int(time)
        startup_candles = dp.get_required_startup(str(tf))
        tf_seconds = timeframe_to_seconds(str(tf))
        timerange.subtract_start(tf_seconds * startup_candles)
        new_pairs_days = int((timerange.stopts - timerange.startts) / 86400)
        # FIXME: now that we are looping on `refresh_backtest_ohlcv_data`, the function
        # redownloads the funding rate for each pair.
        refresh_backtest_ohlcv_data(
            dp._exchange,
            pairs=all_pairs,
            timeframes=[tf],
            datadir=config["datadir"],
            timerange=timerange,
            new_pairs_days=new_pairs_days,
            erase=False,
            data_format=config.get("dataformat_ohlcv", "json"),
            trading_mode=config.get("trading_mode", "spot"),
            prepend=config.get("prepend_data", False),
        )
