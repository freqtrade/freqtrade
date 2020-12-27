from datetime import date, datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from freqtrade.constants import DATETIME_PRINT_FORMAT


class Ping(BaseModel):
    status: str


class AccessToken(BaseModel):
    access_token: str


class AccessAndRefreshToken(AccessToken):
    refresh_token: str


class Version(BaseModel):
    version: str


class StatusMsg(BaseModel):
    status: str


class ResultMsg(BaseModel):
    result: str


class Balance(BaseModel):
    currency: str
    free: float
    balance: float
    used: float
    est_stake: float
    stake: str


class Balances(BaseModel):
    currencies: List[Balance]
    total: float
    symbol: str
    value: float
    stake: str
    note: str


class Count(BaseModel):
    current: int
    max: int
    total_stake: float


class PerformanceEntry(BaseModel):
    pair: str
    profit: float
    count: int


class Profit(BaseModel):
    profit_closed_coin: float
    profit_closed_percent: float
    profit_closed_percent_mean: float
    profit_closed_ratio_mean: float
    profit_closed_percent_sum: float
    profit_closed_ratio_sum: float
    profit_closed_fiat: float
    profit_all_coin: float
    profit_all_percent: float
    profit_all_percent_mean: float
    profit_all_ratio_mean: float
    profit_all_percent_sum: float
    profit_all_ratio_sum: float
    profit_all_fiat: float
    trade_count: int
    closed_trade_count: int
    first_trade_date: str
    first_trade_timestamp: int
    latest_trade_date: str
    latest_trade_timestamp: int
    avg_duration: str
    best_pair: str
    best_rate: float
    winning_trades: int
    losing_trades: int


class SellReason(BaseModel):
    wins: int
    losses: int
    draws: int


class Stats(BaseModel):
    sell_reasons: Dict[str, SellReason]
    durations: Dict[str, Union[str, float]]


class DailyRecord(BaseModel):
    date: date
    abs_profit: float
    fiat_value: float
    trade_count: int


class Daily(BaseModel):
    data: List[DailyRecord]
    fiat_display_currency: str
    stake_currency: str


class LockModel(BaseModel):
    active: bool
    lock_end_time: str
    lock_end_timestamp: int
    lock_time: str
    lock_timestamp: int
    pair: str
    reason: str


class Locks(BaseModel):
    lock_count: int
    locks: List[LockModel]


class Logs(BaseModel):
    log_count: int
    logs: List[List]


class ForceBuyPayload(BaseModel):
    pair: str
    price: Optional[float]


class ForceSellPayload(BaseModel):
    tradeid: str


class BlacklistPayload(BaseModel):
    blacklist: List[str]


class BlacklistResponse(BaseModel):
    blacklist: List[str]
    blacklist_expanded: List[str]
    errors: Dict
    length: int
    method: List[str]


class WhitelistResponse(BaseModel):
    whitelist: List[str]
    length: int
    method: List[str]


class DeleteTrade(BaseModel):
    cancel_order_count: int
    result: str
    result_msg: str
    trade_id: int


class PlotConfig(BaseModel):
    main_plot: Optional[Dict[str, Any]]
    subplots: Optional[Dict[str, Any]]


class StrategyListResponse(BaseModel):
    strategies: List[str]


class StrategyResponse(BaseModel):
    strategy: str
    code: str


class AvailablePairs(BaseModel):
    length: int
    pairs: List[str]
    pair_interval: List[List[str]]


class PairHistory(BaseModel):
    strategy: str
    pair: str
    timeframe: str
    timeframe_ms: int
    columns: List[str]
    data: List[Any]
    length: int
    buy_signals: int
    sell_signals: int
    last_analyzed: datetime
    last_analyzed_ts: int
    data_start_ts: int
    data_start: str
    data_stop: str
    data_stop_ts: int

    class Config:
        json_encoders = {
            datetime: lambda v: v.strftime(DATETIME_PRINT_FORMAT),
        }
