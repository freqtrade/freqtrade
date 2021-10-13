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
    starting_capital: float
    starting_capital_ratio: float
    starting_capital_pct: float
    starting_capital_fiat: float
    starting_capital_fiat_ratio: float
    starting_capital_fiat_pct: float


class Count(BaseModel):
    current: int
    max: int
    total_stake: float


class PerformanceEntry(BaseModel):
    pair: str
    profit: float
    profit_abs: float
    count: int


class Profit(BaseModel):
    profit_closed_coin: float
    profit_closed_percent_mean: float
    profit_closed_ratio_mean: float
    profit_closed_percent_sum: float
    profit_closed_ratio_sum: float
    profit_closed_percent: float
    profit_closed_ratio: float
    profit_closed_fiat: float
    profit_all_coin: float
    profit_all_percent_mean: float
    profit_all_ratio_mean: float
    profit_all_percent_sum: float
    profit_all_ratio_sum: float
    profit_all_percent: float
    profit_all_ratio: float
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


class ShowConfig(BaseModel):
    dry_run: bool
    stake_currency: str
    stake_amount: Union[float, str]
    available_capital: Optional[float]
    stake_currency_decimals: int
    max_open_trades: int
    minimal_roi: Dict[str, Any]
    stoploss: Optional[float]
    trailing_stop: Optional[bool]
    trailing_stop_positive: Optional[float]
    trailing_stop_positive_offset: Optional[float]
    trailing_only_offset_is_reached: Optional[bool]
    use_custom_stoploss: Optional[bool]
    timeframe: Optional[str]
    timeframe_ms: int
    timeframe_min: int
    exchange: str
    strategy: Optional[str]
    forcebuy_enabled: bool
    ask_strategy: Dict[str, Any]
    bid_strategy: Dict[str, Any]
    bot_name: str
    state: str
    runmode: str


class TradeSchema(BaseModel):
    trade_id: int
    pair: str
    is_open: bool
    exchange: str
    amount: float
    amount_requested: float
    stake_amount: float
    strategy: str
    buy_tag: Optional[str]
    timeframe: int
    fee_open: Optional[float]
    fee_open_cost: Optional[float]
    fee_open_currency: Optional[str]
    fee_close: Optional[float]
    fee_close_cost: Optional[float]
    fee_close_currency: Optional[str]
    open_date: str
    open_timestamp: int
    open_rate: float
    open_rate_requested: Optional[float]
    open_trade_value: float
    close_date: Optional[str]
    close_timestamp: Optional[int]
    close_rate: Optional[float]
    close_rate_requested: Optional[float]
    close_profit: Optional[float]
    close_profit_pct: Optional[float]
    close_profit_abs: Optional[float]
    profit_ratio: Optional[float]
    profit_pct: Optional[float]
    profit_abs: Optional[float]
    profit_fiat: Optional[float]
    sell_reason: Optional[str]
    sell_order_status: Optional[str]
    stop_loss_abs: Optional[float]
    stop_loss_ratio: Optional[float]
    stop_loss_pct: Optional[float]
    stoploss_order_id: Optional[str]
    stoploss_last_update: Optional[str]
    stoploss_last_update_timestamp: Optional[int]
    initial_stop_loss_abs: Optional[float]
    initial_stop_loss_ratio: Optional[float]
    initial_stop_loss_pct: Optional[float]
    min_rate: Optional[float]
    max_rate: Optional[float]
    open_order_id: Optional[str]


class OpenTradeSchema(TradeSchema):
    stoploss_current_dist: Optional[float]
    stoploss_current_dist_pct: Optional[float]
    stoploss_current_dist_ratio: Optional[float]
    stoploss_entry_dist: Optional[float]
    stoploss_entry_dist_ratio: Optional[float]
    current_profit: float
    current_profit_abs: float
    current_profit_pct: float
    current_rate: float
    open_order: Optional[str]


class TradeResponse(BaseModel):
    trades: List[TradeSchema]
    trades_count: int
    total_trades: int


class ForceBuyResponse(BaseModel):
    __root__: Union[TradeSchema, StatusMsg]


class LockModel(BaseModel):
    id: int
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


class DeleteLockRequest(BaseModel):
    pair: Optional[str]
    lockid: Optional[int]


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


class PlotConfig_(BaseModel):
    main_plot: Dict[str, Any]
    subplots: Dict[str, Any]


class PlotConfig(BaseModel):
    __root__: Union[PlotConfig_, Dict]


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


class BacktestRequest(BaseModel):
    strategy: str
    timeframe: Optional[str]
    timeframe_detail: Optional[str]
    timerange: Optional[str]
    max_open_trades: Optional[int]
    stake_amount: Optional[Union[float, str]]
    enable_protections: bool
    dry_run_wallet: Optional[float]


class BacktestResponse(BaseModel):
    status: str
    running: bool
    status_msg: str
    step: str
    progress: float
    trade_count: Optional[float]
    # TODO: Properly type backtestresult...
    backtest_result: Optional[Dict[str, Any]]


class SysInfo(BaseModel):
    cpu_pct: List[float]
    ram_pct: float
