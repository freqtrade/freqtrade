from typing import Dict, List, Union

from pydantic import BaseModel


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
