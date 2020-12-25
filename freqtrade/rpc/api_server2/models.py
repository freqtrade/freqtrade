from typing import List
from pydantic import BaseModel


class Ping(BaseModel):
    status: str


class Version(BaseModel):
    version: str


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
