from datetime import date, datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from freqtrade.constants import DATETIME_PRINT_FORMAT, IntOrInf
from freqtrade.enums import MarginMode, OrderTypeValues, SignalDirection, TradingMode
from freqtrade.types import ValidExchangesType


class ExchangeModePayloadMixin(BaseModel):
    trading_mode: Optional[TradingMode]
    margin_mode: Optional[MarginMode]
    exchange: Optional[str]


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


class BgJobStarted(StatusMsg):
    job_id: str


class BackgroundTaskStatus(BaseModel):
    job_id: str
    job_category: str
    status: str
    running: bool
    progress: Optional[float]


class BackgroundTaskResult(BaseModel):
    error: Optional[str]
    status: str


class ResultMsg(BaseModel):
    result: str


class Balance(BaseModel):
    currency: str
    free: float
    balance: float
    used: float
    bot_owned: Optional[float]
    est_stake: float
    est_stake_bot: Optional[float]
    stake: str
    # Starting with 2.x
    side: str
    leverage: float
    is_position: bool
    position: float
    is_bot_managed: bool


class Balances(BaseModel):
    currencies: List[Balance]
    total: float
    total_bot: float
    symbol: str
    value: float
    value_bot: float
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
    profit_ratio: float
    profit_pct: float
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
    first_trade_humanized: str
    first_trade_timestamp: int
    latest_trade_date: str
    latest_trade_humanized: str
    latest_trade_timestamp: int
    avg_duration: str
    best_pair: str
    best_rate: float
    best_pair_profit_ratio: float
    winning_trades: int
    losing_trades: int
    profit_factor: float
    max_drawdown: float
    max_drawdown_abs: float
    trading_volume: Optional[float]
    bot_start_timestamp: int
    bot_start_date: str


class SellReason(BaseModel):
    wins: int
    losses: int
    draws: int


class Stats(BaseModel):
    exit_reasons: Dict[str, SellReason]
    durations: Dict[str, Optional[float]]


class DailyRecord(BaseModel):
    date: date
    abs_profit: float
    rel_profit: float
    starting_balance: float
    fiat_value: float
    trade_count: int


class Daily(BaseModel):
    data: List[DailyRecord]
    fiat_display_currency: str
    stake_currency: str


class UnfilledTimeout(BaseModel):
    entry: Optional[int]
    exit: Optional[int]
    unit: Optional[str]
    exit_timeout_count: Optional[int]


class OrderTypes(BaseModel):
    entry: OrderTypeValues
    exit: OrderTypeValues
    emergency_exit: Optional[OrderTypeValues]
    force_exit: Optional[OrderTypeValues]
    force_entry: Optional[OrderTypeValues]
    stoploss: OrderTypeValues
    stoploss_on_exchange: bool
    stoploss_on_exchange_interval: Optional[int]


class ShowConfig(BaseModel):
    version: str
    strategy_version: Optional[str]
    api_version: float
    dry_run: bool
    trading_mode: str
    short_allowed: bool
    stake_currency: str
    stake_amount: str
    available_capital: Optional[float]
    stake_currency_decimals: int
    max_open_trades: IntOrInf
    minimal_roi: Dict[str, Any]
    stoploss: Optional[float]
    stoploss_on_exchange: bool
    trailing_stop: Optional[bool]
    trailing_stop_positive: Optional[float]
    trailing_stop_positive_offset: Optional[float]
    trailing_only_offset_is_reached: Optional[bool]
    unfilledtimeout: Optional[UnfilledTimeout]  # Empty in webserver mode
    order_types: Optional[OrderTypes]
    use_custom_stoploss: Optional[bool]
    timeframe: Optional[str]
    timeframe_ms: int
    timeframe_min: int
    exchange: str
    strategy: Optional[str]
    force_entry_enable: bool
    exit_pricing: Dict[str, Any]
    entry_pricing: Dict[str, Any]
    bot_name: str
    state: str
    runmode: str
    position_adjustment_enable: bool
    max_entry_position_adjustment: int


class OrderSchema(BaseModel):
    pair: str
    order_id: str
    status: str
    remaining: Optional[float]
    amount: float
    safe_price: float
    cost: float
    filled: Optional[float]
    ft_order_side: str
    order_type: str
    is_open: bool
    order_timestamp: Optional[int]
    order_filled_timestamp: Optional[int]


class TradeSchema(BaseModel):
    trade_id: int
    pair: str
    base_currency: str
    quote_currency: str
    is_open: bool
    is_short: bool
    exchange: str
    amount: float
    amount_requested: float
    stake_amount: float
    max_stake_amount: Optional[float]
    strategy: str
    enter_tag: Optional[str]
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

    realized_profit: float
    realized_profit_ratio: Optional[float]

    exit_reason: Optional[str]
    exit_order_status: Optional[str]

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
    orders: List[OrderSchema]

    leverage: Optional[float]
    interest_rate: Optional[float]
    liquidation_price: Optional[float]
    funding_fees: Optional[float]
    trading_mode: Optional[TradingMode]

    amount_precision: Optional[float]
    price_precision: Optional[float]
    precision_mode: Optional[int]


class OpenTradeSchema(TradeSchema):
    stoploss_current_dist: Optional[float]
    stoploss_current_dist_pct: Optional[float]
    stoploss_current_dist_ratio: Optional[float]
    stoploss_entry_dist: Optional[float]
    stoploss_entry_dist_ratio: Optional[float]
    current_rate: float
    total_profit_abs: float
    total_profit_fiat: Optional[float]
    total_profit_ratio: Optional[float]

    open_order: Optional[str]


class TradeResponse(BaseModel):
    trades: List[TradeSchema]
    trades_count: int
    offset: int
    total_trades: int


class ForceEnterResponse(BaseModel):
    __root__: Union[TradeSchema, StatusMsg]


class LockModel(BaseModel):
    id: int
    active: bool
    lock_end_time: str
    lock_end_timestamp: int
    lock_time: str
    lock_timestamp: int
    pair: str
    side: str
    reason: Optional[str]


class Locks(BaseModel):
    lock_count: int
    locks: List[LockModel]


class DeleteLockRequest(BaseModel):
    pair: Optional[str]
    lockid: Optional[int]


class Logs(BaseModel):
    log_count: int
    logs: List[List]


class ForceEnterPayload(BaseModel):
    pair: str
    side: SignalDirection = SignalDirection.LONG
    price: Optional[float]
    ordertype: Optional[OrderTypeValues]
    stakeamount: Optional[float]
    entry_tag: Optional[str]
    leverage: Optional[float]


class ForceExitPayload(BaseModel):
    tradeid: str
    ordertype: Optional[OrderTypeValues]
    amount: Optional[float]


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


class WhitelistEvaluateResponse(BackgroundTaskResult):
    result: Optional[WhitelistResponse]


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


class ExchangeListResponse(BaseModel):
    exchanges: List[ValidExchangesType]


class PairListResponse(BaseModel):
    name: str
    description: str
    is_pairlist_generator: bool
    params: Dict[str, Any]


class PairListsResponse(BaseModel):
    pairlists: List[PairListResponse]


class PairListsPayload(ExchangeModePayloadMixin, BaseModel):
    pairlists: List[Dict[str, Any]]
    blacklist: List[str]
    stake_currency: str


class FreqAIModelListResponse(BaseModel):
    freqaimodels: List[str]


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
    enter_long_signals: int
    exit_long_signals: int
    enter_short_signals: int
    exit_short_signals: int
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


class BacktestFreqAIInputs(BaseModel):
    identifier: str


class BacktestRequest(BaseModel):
    strategy: str
    timeframe: Optional[str]
    timeframe_detail: Optional[str]
    timerange: Optional[str]
    max_open_trades: Optional[IntOrInf]
    stake_amount: Optional[str]
    enable_protections: bool
    dry_run_wallet: Optional[float]
    backtest_cache: Optional[str]
    freqaimodel: Optional[str]
    freqai: Optional[BacktestFreqAIInputs]


class BacktestResponse(BaseModel):
    status: str
    running: bool
    status_msg: str
    step: str
    progress: float
    trade_count: Optional[float]
    # TODO: Properly type backtestresult...
    backtest_result: Optional[Dict[str, Any]]


class BacktestHistoryEntry(BaseModel):
    filename: str
    strategy: str
    run_id: str
    backtest_start_time: int


class SysInfo(BaseModel):
    cpu_pct: List[float]
    ram_pct: float


class Health(BaseModel):
    last_process: Optional[datetime]
    last_process_ts: Optional[int]
