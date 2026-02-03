from datetime import date, datetime
from typing import Any

from pydantic import AwareDatetime, BaseModel, RootModel, SerializeAsAny, model_validator

from freqtrade.constants import DL_DATA_TIMEFRAMES, IntOrInf
from freqtrade.enums import MarginMode, OrderTypeValues, SignalDirection, TradingMode
from freqtrade.ft_types import AnnotationType, ValidExchangesType
from freqtrade.rpc.api_server.webserver_bgwork import ProgressTask


class ExchangeModePayloadMixin(BaseModel):
    trading_mode: TradingMode | None = None
    margin_mode: MarginMode | None = None
    exchange: str | None = None


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
    progress: float | None = None
    progress_tasks: dict[str, ProgressTask] | None = None
    error: str | None = None


class BackgroundTaskResult(BaseModel):
    error: str | None = None
    status: str


class ResultMsg(BaseModel):
    result: str


class Balance(BaseModel):
    currency: str
    free: float
    balance: float
    used: float
    bot_owned: float | None = None
    est_stake: float
    est_stake_bot: float | None = None
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
    best_pair_profit_abs: float
    winning_trades: int
    losing_trades: int
    profit_factor: float
    winrate: float
    expectancy: float
    expectancy_ratio: float
    sharpe: float
    sortino: float
    sqn: float
    calmar: float
    cagr: float
    max_drawdown: float
    max_drawdown_abs: float
    max_drawdown_start: str
    max_drawdown_start_timestamp: int
    max_drawdown_end: str
    max_drawdown_end_timestamp: int
    current_drawdown: float
    current_drawdown_abs: float
    current_drawdown_high: float
    current_drawdown_start: str
    current_drawdown_start_timestamp: int
    trading_volume: float | None = None
    bot_start_timestamp: int
    bot_start_date: str


class ProfitAll(BaseModel):
    all: Profit
    long: Profit | None = None
    short: Profit | None = None


class SellReason(BaseModel):
    wins: int
    losses: int
    draws: int


class Stats(BaseModel):
    exit_reasons: dict[str, SellReason]
    durations: dict[str, float | None]


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
    entry: int | None = None
    exit: int | None = None
    unit: str | None = None
    exit_timeout_count: int | None = None


class OrderTypes(BaseModel):
    entry: OrderTypeValues
    exit: OrderTypeValues
    emergency_exit: OrderTypeValues | None = None
    force_exit: OrderTypeValues | None = None
    force_entry: OrderTypeValues | None = None
    stoploss: OrderTypeValues
    stoploss_on_exchange: bool
    stoploss_on_exchange_interval: int | None = None


class ShowConfig(BaseModel):
    version: str
    strategy_version: str | None = None
    api_version: float
    dry_run: bool
    trading_mode: str
    margin_mode: str
    short_allowed: bool
    stake_currency: str
    stake_amount: str
    available_capital: float | None = None
    stake_currency_decimals: int
    max_open_trades: IntOrInf
    minimal_roi: dict[str, Any]
    stoploss: float | None = None
    stoploss_on_exchange: bool
    trailing_stop: bool | None = None
    trailing_stop_positive: float | None = None
    trailing_stop_positive_offset: float | None = None
    trailing_only_offset_is_reached: bool | None = None
    unfilledtimeout: UnfilledTimeout | None = None  # Empty in webserver mode
    order_types: OrderTypes | None = None
    use_custom_stoploss: bool | None = None
    timeframe: str | None = None
    timeframe_ms: int
    timeframe_min: int
    exchange: str
    strategy: str | None = None
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
    remaining: float | None = None
    amount: float
    safe_price: float
    cost: float
    filled: float | None = None
    ft_order_side: str
    order_type: str
    is_open: bool
    order_timestamp: int | None = None
    order_filled_timestamp: int | None = None
    ft_fee_base: float | None = None
    ft_order_tag: str | None = None


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
    max_stake_amount: float | None = None
    strategy: str
    enter_tag: str | None = None
    timeframe: int
    fee_open: float | None = None
    fee_open_cost: float | None = None
    fee_open_currency: str | None = None
    fee_close: float | None = None
    fee_close_cost: float | None = None
    fee_close_currency: str | None = None

    open_date: str
    open_timestamp: int
    open_fill_date: str | None
    open_fill_timestamp: int | None
    open_rate: float
    open_rate_requested: float | None = None
    open_trade_value: float

    close_date: str | None = None
    close_timestamp: int | None = None
    close_rate: float | None = None
    close_rate_requested: float | None = None

    close_profit: float | None = None
    close_profit_pct: float | None = None
    close_profit_abs: float | None = None

    profit_ratio: float | None = None
    profit_pct: float | None = None
    profit_abs: float | None = None
    profit_fiat: float | None = None

    realized_profit: float
    realized_profit_ratio: float | None = None

    exit_reason: str | None = None
    exit_order_status: str | None = None

    stop_loss_abs: float | None = None
    stop_loss_ratio: float | None = None
    stop_loss_pct: float | None = None
    stoploss_last_update: str | None = None
    stoploss_last_update_timestamp: int | None = None
    initial_stop_loss_abs: float | None = None
    initial_stop_loss_ratio: float | None = None
    initial_stop_loss_pct: float | None = None

    min_rate: float | None = None
    max_rate: float | None = None
    nr_of_successful_entries: int
    nr_of_successful_exits: int
    has_open_orders: bool
    orders: list[OrderSchema]

    leverage: float | None = None
    interest_rate: float | None = None
    liquidation_price: float | None = None
    funding_fees: float | None = None
    trading_mode: TradingMode | None = None

    amount_precision: float | None = None
    price_precision: float | None = None
    precision_mode: int | None = None


class OpenTradeSchema(TradeSchema):
    stoploss_current_dist: float | None = None
    stoploss_current_dist_pct: float | None = None
    stoploss_current_dist_ratio: float | None = None
    stoploss_entry_dist: float | None = None
    stoploss_entry_dist_ratio: float | None = None
    current_rate: float
    total_profit_abs: float
    total_profit_fiat: float | None = None
    total_profit_ratio: float | None = None


class TradeResponse(BaseModel):
    trades: list[TradeSchema]
    trades_count: int
    offset: int
    total_trades: int


ForceEnterResponse = RootModel[TradeSchema | StatusMsg]


class LockModel(BaseModel):
    id: int
    active: bool
    lock_end_time: str
    lock_end_timestamp: int
    lock_time: str
    lock_timestamp: int
    pair: str
    side: str
    reason: str | None = None


class Locks(BaseModel):
    lock_count: int
    locks: list[LockModel]


class LocksPayload(BaseModel):
    pair: str
    side: str = "*"  # Default to both sides
    until: AwareDatetime
    reason: str | None = None


class DeleteLockRequest(BaseModel):
    pair: str | None = None
    lockid: int | None = None


class Logs(BaseModel):
    log_count: int
    logs: list[list]


class ForceEnterPayload(BaseModel):
    pair: str
    side: SignalDirection = SignalDirection.LONG
    price: float | None = None
    ordertype: OrderTypeValues | None = None
    stakeamount: float | None = None
    entry_tag: str | None = None
    leverage: float | None = None


class ForceExitPayload(BaseModel):
    tradeid: str | int
    ordertype: OrderTypeValues | None = None
    amount: float | None = None
    price: float | None = None


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
    result: WhitelistResponse | None = None


class DeleteTrade(BaseModel):
    cancel_order_count: int
    result: str
    result_msg: str
    trade_id: int


class PlotConfig_(BaseModel):
    main_plot: dict[str, Any]
    subplots: dict[str, Any]


PlotConfig = RootModel[PlotConfig_ | dict]


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


class DownloadDataPayload(ExchangeModePayloadMixin, BaseModel):
    pairs: list[str]
    timeframes: list[str] | None = DL_DATA_TIMEFRAMES
    days: int | None = None
    timerange: str | None = None
    erase: bool = False
    download_trades: bool = False
    candle_types: list[str] | None = None

    @model_validator(mode="before")
    def check_mutually_exclusive(cls, values):
        timeframes, days = values.get("timerange"), values.get("days")
        if timeframes and days:
            raise ValueError("Only one of timeframes or days can be provided, not both.")
        return values


class FreqAIModelListResponse(BaseModel):
    freqaimodels: list[str]


class StrategyResponse(BaseModel):
    strategy: str
    code: str
    timeframe: str | None


class AvailablePairs(BaseModel):
    length: int
    pairs: list[str]
    pair_interval: list[list[str]]


class PairCandlesRequest(BaseModel):
    pair: str
    timeframe: str
    limit: int | None = None
    columns: list[str] | None = None


class PairHistoryRequest(PairCandlesRequest, ExchangeModePayloadMixin):
    timerange: str
    strategy: str | None = None
    freqaimodel: str | None = None
    live_mode: bool = False


class PairHistory(BaseModel):
    strategy: str
    pair: str
    timeframe: str
    timeframe_ms: int
    columns: list[str]
    all_columns: list[str] = []
    data: SerializeAsAny[list[Any]]
    annotations: list[AnnotationType] | None = None
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
    timeframe: str | None = None
    timeframe_detail: str | None = None
    timerange: str | None = None
    max_open_trades: IntOrInf | None = None
    stake_amount: str | float | None = None
    enable_protections: bool
    dry_run_wallet: float | None = None
    backtest_cache: str | None = None
    freqaimodel: str | None = None
    freqai: BacktestFreqAIInputs | None = None


class BacktestResponse(BaseModel):
    status: str
    running: bool
    status_msg: str
    step: str
    progress: float
    trade_count: float | None = None
    # TODO: Properly type backtestresult...
    backtest_result: dict[str, Any] | None = None


# TODO: This is a copy of BacktestHistoryEntryType
class BacktestHistoryEntry(BaseModel):
    filename: str
    strategy: str
    run_id: str
    backtest_start_time: int
    notes: str | None = ""
    backtest_start_ts: int | None = None
    backtest_end_ts: int | None = None
    timeframe: str | None = None
    timeframe_detail: str | None = None


class BacktestMetadataUpdate(BaseModel):
    strategy: str
    notes: str = ""


class BacktestMarketChange(BaseModel):
    columns: list[str]
    length: int
    data: list[list[Any]]


class MarketRequest(ExchangeModePayloadMixin, BaseModel):
    base: str | None = None
    quote: str | None = None


class MarketModel(BaseModel):
    symbol: str
    base: str
    quote: str
    spot: bool
    swap: bool


class MarketResponse(BaseModel):
    markets: dict[str, MarketModel]
    exchange_id: str


class SysInfo(BaseModel):
    cpu_pct: list[float]
    ram_pct: float


class Health(BaseModel):
    last_process: datetime | None = None
    last_process_ts: int | None = None
    bot_start: datetime | None = None
    bot_start_ts: int | None = None
    bot_startup: datetime | None = None
    bot_startup_ts: int | None = None


class CustomDataEntry(BaseModel):
    key: str
    type: str
    value: Any
    created_at: datetime
    updated_at: datetime | None = None


class ListCustomData(BaseModel):
    trade_id: int
    custom_data: list[CustomDataEntry]
