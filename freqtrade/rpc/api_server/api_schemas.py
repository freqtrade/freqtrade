from datetime import date, datetime
from typing import Any, Optional, Union

from pydantic import AwareDatetime, BaseModel, RootModel, SerializeAsAny

from freqtrade.constants import IntOrInf
from freqtrade.enums import MarginMode, OrderTypeValues, SignalDirection, TradingMode
from freqtrade.ft_types import ValidExchangesType


class ExchangeModePayloadMixin(BaseModel):
    trading_mode: Optional[TradingMode] = None
    margin_mode: Optional[MarginMode] = None
    exchange: Optional[str] = None


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
    progress: Optional[float] = None
    error: Optional[str] = None


class BackgroundTaskResult(BaseModel):
    error: Optional[str] = None
    status: str


class ResultMsg(BaseModel):
    result: str


class Balance(BaseModel):
    currency: str
    free: float
    balance: float
    used: float
    bot_owned: Optional[float] = None
    est_stake: float
    est_stake_bot: Optional[float] = None
    stake: str
    # Starting with 2.x
    side: str
    is_position: bool
    position: float
    is_bot_managed: bool


class Balances(BaseModel):
    currencies: list[Balance]
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


class __BaseStatsModel(BaseModel):
    profit_ratio: float
    profit_pct: float
    profit_abs: float
    count: int


class Entry(__BaseStatsModel):
    enter_tag: str


class Exit(__BaseStatsModel):
    exit_reason: str


class MixTag(__BaseStatsModel):
    mix_tag: str


class PerformanceEntry(__BaseStatsModel):
    pair: str
    profit: float


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
    winrate: float
    expectancy: float
    expectancy_ratio: float
    max_drawdown: float
    max_drawdown_abs: float
    max_drawdown_start: str
    max_drawdown_start_timestamp: int
    max_drawdown_end: str
    max_drawdown_end_timestamp: int
    trading_volume: Optional[float] = None
    bot_start_timestamp: int
    bot_start_date: str


class SellReason(BaseModel):
    wins: int
    losses: int
    draws: int


class Stats(BaseModel):
    exit_reasons: dict[str, SellReason]
    durations: dict[str, Optional[float]]


class DailyWeeklyMonthlyRecord(BaseModel):
    date: date
    abs_profit: float
    rel_profit: float
    starting_balance: float
    fiat_value: float
    trade_count: int


class DailyWeeklyMonthly(BaseModel):
    data: list[DailyWeeklyMonthlyRecord]
    fiat_display_currency: str
    stake_currency: str


class UnfilledTimeout(BaseModel):
    entry: Optional[int] = None
    exit: Optional[int] = None
    unit: Optional[str] = None
    exit_timeout_count: Optional[int] = None


class OrderTypes(BaseModel):
    entry: OrderTypeValues
    exit: OrderTypeValues
    emergency_exit: Optional[OrderTypeValues] = None
    force_exit: Optional[OrderTypeValues] = None
    force_entry: Optional[OrderTypeValues] = None
    stoploss: OrderTypeValues
    stoploss_on_exchange: bool
    stoploss_on_exchange_interval: Optional[int] = None


class ShowConfig(BaseModel):
    version: str
    strategy_version: Optional[str] = None
    api_version: float
    dry_run: bool
    trading_mode: str
    short_allowed: bool
    stake_currency: str
    stake_amount: str
    available_capital: Optional[float] = None
    stake_currency_decimals: int
    max_open_trades: IntOrInf
    minimal_roi: dict[str, Any]
    stoploss: Optional[float] = None
    stoploss_on_exchange: bool
    trailing_stop: Optional[bool] = None
    trailing_stop_positive: Optional[float] = None
    trailing_stop_positive_offset: Optional[float] = None
    trailing_only_offset_is_reached: Optional[bool] = None
    unfilledtimeout: Optional[UnfilledTimeout] = None  # Empty in webserver mode
    order_types: Optional[OrderTypes] = None
    use_custom_stoploss: Optional[bool] = None
    timeframe: Optional[str] = None
    timeframe_ms: int
    timeframe_min: int
    exchange: str
    strategy: Optional[str] = None
    force_entry_enable: bool
    exit_pricing: dict[str, Any]
    entry_pricing: dict[str, Any]
    bot_name: str
    state: str
    runmode: str
    position_adjustment_enable: bool
    max_entry_position_adjustment: int


class OrderSchema(BaseModel):
    pair: str
    order_id: str
    status: str
    remaining: Optional[float] = None
    amount: float
    safe_price: float
    cost: float
    filled: Optional[float] = None
    ft_order_side: str
    order_type: str
    is_open: bool
    order_timestamp: Optional[int] = None
    order_filled_timestamp: Optional[int] = None
    ft_fee_base: Optional[float] = None
    ft_order_tag: Optional[str] = None


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
    max_stake_amount: Optional[float] = None
    strategy: str
    enter_tag: Optional[str] = None
    timeframe: int
    fee_open: Optional[float] = None
    fee_open_cost: Optional[float] = None
    fee_open_currency: Optional[str] = None
    fee_close: Optional[float] = None
    fee_close_cost: Optional[float] = None
    fee_close_currency: Optional[str] = None

    open_date: str
    open_timestamp: int
    open_fill_date: Optional[str]
    open_fill_timestamp: Optional[int]
    open_rate: float
    open_rate_requested: Optional[float] = None
    open_trade_value: float

    close_date: Optional[str] = None
    close_timestamp: Optional[int] = None
    close_rate: Optional[float] = None
    close_rate_requested: Optional[float] = None

    close_profit: Optional[float] = None
    close_profit_pct: Optional[float] = None
    close_profit_abs: Optional[float] = None

    profit_ratio: Optional[float] = None
    profit_pct: Optional[float] = None
    profit_abs: Optional[float] = None
    profit_fiat: Optional[float] = None

    realized_profit: float
    realized_profit_ratio: Optional[float] = None

    exit_reason: Optional[str] = None
    exit_order_status: Optional[str] = None

    stop_loss_abs: Optional[float] = None
    stop_loss_ratio: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    stoploss_last_update: Optional[str] = None
    stoploss_last_update_timestamp: Optional[int] = None
    initial_stop_loss_abs: Optional[float] = None
    initial_stop_loss_ratio: Optional[float] = None
    initial_stop_loss_pct: Optional[float] = None

    min_rate: Optional[float] = None
    max_rate: Optional[float] = None
    has_open_orders: bool
    orders: list[OrderSchema]

    leverage: Optional[float] = None
    interest_rate: Optional[float] = None
    liquidation_price: Optional[float] = None
    funding_fees: Optional[float] = None
    trading_mode: Optional[TradingMode] = None

    amount_precision: Optional[float] = None
    price_precision: Optional[float] = None
    precision_mode: Optional[int] = None


class OpenTradeSchema(TradeSchema):
    stoploss_current_dist: Optional[float] = None
    stoploss_current_dist_pct: Optional[float] = None
    stoploss_current_dist_ratio: Optional[float] = None
    stoploss_entry_dist: Optional[float] = None
    stoploss_entry_dist_ratio: Optional[float] = None
    current_rate: float
    total_profit_abs: float
    total_profit_fiat: Optional[float] = None
    total_profit_ratio: Optional[float] = None


class TradeResponse(BaseModel):
    trades: list[TradeSchema]
    trades_count: int
    offset: int
    total_trades: int


ForceEnterResponse = RootModel[Union[TradeSchema, StatusMsg]]


class LockModel(BaseModel):
    id: int
    active: bool
    lock_end_time: str
    lock_end_timestamp: int
    lock_time: str
    lock_timestamp: int
    pair: str
    side: str
    reason: Optional[str] = None


class Locks(BaseModel):
    lock_count: int
    locks: list[LockModel]


class LocksPayload(BaseModel):
    pair: str
    side: str = "*"  # Default to both sides
    until: AwareDatetime
    reason: Optional[str] = None


class DeleteLockRequest(BaseModel):
    pair: Optional[str] = None
    lockid: Optional[int] = None


class Logs(BaseModel):
    log_count: int
    logs: list[list]


class ForceEnterPayload(BaseModel):
    pair: str
    side: SignalDirection = SignalDirection.LONG
    price: Optional[float] = None
    ordertype: Optional[OrderTypeValues] = None
    stakeamount: Optional[float] = None
    entry_tag: Optional[str] = None
    leverage: Optional[float] = None


class ForceExitPayload(BaseModel):
    tradeid: Union[str, int]
    ordertype: Optional[OrderTypeValues] = None
    amount: Optional[float] = None


class BlacklistPayload(BaseModel):
    blacklist: list[str]


class BlacklistResponse(BaseModel):
    blacklist: list[str]
    blacklist_expanded: list[str]
    errors: dict
    length: int
    method: list[str]


class WhitelistResponse(BaseModel):
    whitelist: list[str]
    length: int
    method: list[str]


class WhitelistEvaluateResponse(BackgroundTaskResult):
    result: Optional[WhitelistResponse] = None


class DeleteTrade(BaseModel):
    cancel_order_count: int
    result: str
    result_msg: str
    trade_id: int


class PlotConfig_(BaseModel):
    main_plot: dict[str, Any]
    subplots: dict[str, Any]


PlotConfig = RootModel[Union[PlotConfig_, dict]]


class StrategyListResponse(BaseModel):
    strategies: list[str]


class ExchangeListResponse(BaseModel):
    exchanges: list[ValidExchangesType]


class HyperoptLoss(BaseModel):
    name: str
    description: str


class HyperoptLossListResponse(BaseModel):
    loss_functions: list[HyperoptLoss]


class PairListResponse(BaseModel):
    name: str
    description: str
    is_pairlist_generator: bool
    params: dict[str, Any]


class PairListsResponse(BaseModel):
    pairlists: list[PairListResponse]


class PairListsPayload(ExchangeModePayloadMixin, BaseModel):
    pairlists: list[dict[str, Any]]
    blacklist: list[str]
    stake_currency: str


class FreqAIModelListResponse(BaseModel):
    freqaimodels: list[str]


class StrategyResponse(BaseModel):
    strategy: str
    code: str
    timeframe: Optional[str]


class AvailablePairs(BaseModel):
    length: int
    pairs: list[str]
    pair_interval: list[list[str]]


class PairCandlesRequest(BaseModel):
    pair: str
    timeframe: str
    limit: Optional[int] = None
    columns: Optional[list[str]] = None


class PairHistoryRequest(PairCandlesRequest):
    timerange: str
    strategy: str
    freqaimodel: Optional[str] = None


class PairHistory(BaseModel):
    strategy: str
    pair: str
    timeframe: str
    timeframe_ms: int
    columns: list[str]
    all_columns: list[str] = []
    data: SerializeAsAny[list[Any]]
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


class BacktestFreqAIInputs(BaseModel):
    identifier: str


class BacktestRequest(BaseModel):
    strategy: str
    timeframe: Optional[str] = None
    timeframe_detail: Optional[str] = None
    timerange: Optional[str] = None
    max_open_trades: Optional[IntOrInf] = None
    stake_amount: Optional[Union[str, float]] = None
    enable_protections: bool
    dry_run_wallet: Optional[float] = None
    backtest_cache: Optional[str] = None
    freqaimodel: Optional[str] = None
    freqai: Optional[BacktestFreqAIInputs] = None


class BacktestResponse(BaseModel):
    status: str
    running: bool
    status_msg: str
    step: str
    progress: float
    trade_count: Optional[float] = None
    # TODO: Properly type backtestresult...
    backtest_result: Optional[dict[str, Any]] = None


# TODO: This is a copy of BacktestHistoryEntryType
class BacktestHistoryEntry(BaseModel):
    filename: str
    strategy: str
    run_id: str
    backtest_start_time: int
    notes: Optional[str] = ""
    backtest_start_ts: Optional[int] = None
    backtest_end_ts: Optional[int] = None
    timeframe: Optional[str] = None
    timeframe_detail: Optional[str] = None


class BacktestMetadataUpdate(BaseModel):
    strategy: str
    notes: str = ""


class BacktestMarketChange(BaseModel):
    columns: list[str]
    length: int
    data: list[list[Any]]


class SysInfo(BaseModel):
    cpu_pct: list[float]
    ram_pct: float


class Health(BaseModel):
    last_process: Optional[datetime] = None
    last_process_ts: Optional[int] = None
    bot_start: Optional[datetime] = None
    bot_start_ts: Optional[int] = None
    bot_startup: Optional[datetime] = None
    bot_startup_ts: Optional[int] = None
