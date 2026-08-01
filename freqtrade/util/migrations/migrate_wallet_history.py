import logging

import numpy as np
import pandas as pd

from freqtrade.constants import Config
from freqtrade.data.btanalysis.bt_fileutils import trade_list_to_dataframe
from freqtrade.data.btanalysis.trade_parallelism import balance_distribution_over_time
from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_prev_date
from freqtrade.persistence import KeyValueStore, Trade, WalletHistory
from freqtrade.util import dt_now, dt_ts


logger = logging.getLogger(__name__)


def migrate_wallet_history(config: Config, exchange: Exchange, starting_balance: float):
    if config.get("skip_wallet_history_migration") or not exchange.get_option(
        "ohlcv_has_history", True
    ):
        # we can't fill up wallet history without ohlcv history
        return
    if KeyValueStore.get_int_value("wallet_history_migration"):
        logger.debug("Wallet history migration already completed.")
        return
    logger.info("Starting wallet history migration...")
    _migrate_wallet_history(config, exchange, starting_balance)
    logger.info("Wallet history migration completed.")
    KeyValueStore.store_value("wallet_history_migration", 1)
    KeyValueStore.store_value("wallet_history_migration_date", dt_now())


def _migrate_wallet_history(config: Config, exchange: Exchange, starting_balance: float):
    # Prepare balance distribution data with OHLCV rates
    balance_dist, pairlist_valid = _prepare_balance_distribution(config, exchange, starting_balance)
    if not balance_dist.empty and pairlist_valid:
        _create_wallet_history_entries(
            config, exchange, balance_dist, pairlist_valid, config["stake_currency"]
        )


def _prepare_balance_distribution(
    config: Config, exchange: Exchange, starting_balance: float
) -> tuple[pd.DataFrame, list[str]]:
    trade_df = trade_list_to_dataframe(Trade.get_trades_proxy(), minified=False)
    if trade_df.empty:
        # no trades, nothing to do
        return pd.DataFrame(), []
    pairlist = list(trade_df["pair"].unique())
    timeframe = "1d"
    stake_currency = config["stake_currency"]
    min_date = timeframe_to_prev_date(timeframe, KeyValueStore.get_datetime_value("bot_start_time"))
    balance_dist = balance_distribution_over_time(
        trade_df,
        min_date=min_date,
        max_date=dt_now(),
        start_balance=starting_balance,
        stake_currency=stake_currency,
        timeframe=timeframe,
        pairlist=pairlist,
    )
    pairlist_valid = [p for p in pairlist if p in exchange.markets]
    pairlist_invalid = set(pairlist) - set(pairlist_valid)
    if pairlist_invalid:
        logger.warning(
            f"The following trading pairs from the trade history are not available on the exchange "
            f"and will be skipped during wallet history migration: {', '.join(pairlist_invalid)}"
        )

    logger.info("Wallet History migration: Fetching OHLCV data ...")
    data = exchange.refresh_latest_ohlcv(
        [(p, timeframe, config["candle_type_def"]) for p in pairlist_valid],
        since_ms=dt_ts(min_date),
        cache=False,
        drop_incomplete=False,
    )
    logger.info(
        "Wallet History migration: Done fetching OHLCV data for wallet history migration..."
    )

    dfs = []
    # Combine all dataframes into one using the open rate
    for p, x in data.items():
        x = x.set_index("date", drop=True)
        col = f"{p[0]}_open"
        x[col] = x["open"]
        dfs.append(x[[col]])

    if not dfs:
        logger.warning(
            "No OHLCV data available for the trading pairs; skipping wallet history migration."
        )
        return pd.DataFrame(), []
    merged = pd.concat(dfs, axis=1)

    balance_dist = balance_dist.join(merged, how="left")
    df_value = pd.DataFrame(
        index=balance_dist.index, columns=[f"{p}_value" for p in pairlist_valid], dtype=float
    )
    for p in pairlist_valid:
        # df_value[f"{p}_value"] = balance_dist[f"{p}_open"] * balance_dist[p]
        # Identical calculation to rpc and wallets.py
        df_value[f"{p}_value"] = np.where(
            balance_dist[f"{p}_is_short"] == 0,
            (balance_dist[f"{p}_open"] * balance_dist[p])
            - balance_dist[f"{p}_collateral"] * (balance_dist[f"{p}_leverage"] - 1),
            (
                balance_dist[f"{p}_collateral"] * (1 + balance_dist[f"{p}_leverage"])
                - balance_dist[f"{p}_open"] * balance_dist[p]
            ),
        )
    balance_dist = pd.concat([balance_dist, df_value], axis=1)

    # Aggregate total value at each point in time
    balance_dist["total_value"] = balance_dist[
        [f"{p}_value" for p in pairlist_valid] + [stake_currency]
    ].sum(axis=1)

    return balance_dist, pairlist_valid


def _create_wallet_history_entries(
    config: Config,
    exchange: Exchange,
    balance_dist: pd.DataFrame,
    pairlist_valid: list[str],
    stake_currency: str,
):
    is_futures = config["trading_mode"] == "futures"
    # Precompute column indices for faster tuple-based iteration
    # Assume the first column is the index (date)
    stake_idx = balance_dist.columns.get_loc(stake_currency)
    pair_balance_idx = {pair: balance_dist.columns.get_loc(pair) + 1 for pair in pairlist_valid}
    pair_leverage_idx = {
        pair: balance_dist.columns.get_loc(f"{pair}_leverage") + 1 for pair in pairlist_valid
    }
    pair_collateral_idx = {
        pair: balance_dist.columns.get_loc(f"{pair}_collateral") + 1 for pair in pairlist_valid
    }
    pair_is_short_idx = {
        pair: balance_dist.columns.get_loc(f"{pair}_is_short") + 1 for pair in pairlist_valid
    }
    pair_rate_idx = {
        pair: balance_dist.columns.get_loc(f"{pair}_open") + 1 for pair in pairlist_valid
    }
    # Convert balance_dist to WalletHistory entries
    wallet_entries = []
    for row in balance_dist.itertuples(index=True, name=None):
        date = row[0]

        # Add stake currency entry
        stake_balance = row[stake_idx + 1]
        if not pd.isna(stake_balance):
            wallet_entries.append(
                WalletHistory(
                    timestamp=date,
                    currency=stake_currency,
                    rate=1.0,  # Stake currency price is always 1.0
                    balance=stake_balance,
                    total_quote=stake_balance,
                    quote_currency=stake_currency,
                    leverage=1.0,
                    bot_managed=True,
                )
            )

        # Add entries for each trading pair
        for pair in pairlist_valid:
            base_currency = exchange.get_pair_base_currency(pair)
            balance = row[pair_balance_idx[pair]]
            leverage = row[pair_leverage_idx[pair]]
            # Only add entry if balance is not empty/NaN
            if not pd.isna(balance) and balance > 0:
                rate_value = row[pair_rate_idx[pair]]
                rate = rate_value if not pd.isna(rate_value) else None

                total_quote = balance * rate if rate else None
                collateral: float | None = None
                if is_futures:
                    collateral = row[pair_collateral_idx[pair]]
                    is_short = row[pair_is_short_idx[pair]]
                    if collateral is not None and not pd.isna(collateral) and rate is not None:
                        # Same formula than in rpc's _rpc_balance
                        total_quote = (
                            (rate * balance - collateral * (leverage - 1))
                            if is_short == 0
                            else (collateral * (1 + leverage) - rate * balance)
                        )
                wallet_entries.append(
                    WalletHistory(
                        timestamp=date,
                        currency=base_currency,
                        quote_currency=stake_currency,
                        rate=rate,
                        balance=balance,
                        total_quote=total_quote,
                        leverage=leverage if not pd.isna(leverage) else 1.0,
                        bot_managed=True,
                        total_position_value=balance * rate if is_futures and rate else None,
                        # collateral=collateral,
                    )
                )

    # Save entries to database
    if wallet_entries:
        try:
            # Use bulk_save_objects for better performance
            WalletHistory.session.bulk_save_objects(wallet_entries)
            WalletHistory.session.commit()
            logger.info(f"Successfully created {len(wallet_entries)} wallet balance records")
        except Exception as e:
            WalletHistory.session.rollback()
            logger.error(f"Error saving wallet balance records: {e}")
