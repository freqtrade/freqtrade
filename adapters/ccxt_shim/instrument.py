from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re


class InstrumentType(str, Enum):
    CASH = "cash"
    FUT = "future"
    OPT = "option"


@dataclass(frozen=True)
class InstrumentSpec:
    """
    Canonical specification for a tradeable instrument pair.

    Attributes:
        type: InstrumentType indicating cash, futures, or options.
        underlying: Underlying symbol (upper-case).
        quote: Quote currency (default INR).
        expiry_yyyymmdd: Expiry in YYYYMMDD (required for FUT/OPT).
        strike: Strike price for options.
        right: Option right, CE or PE.
    """

    type: InstrumentType
    underlying: str
    quote: str = "INR"
    expiry_yyyymmdd: str | None = None
    strike: float | None = None
    right: str | None = None

    def validate(self) -> None:
        if not self.underlying:
            raise ValueError("Underlying is required.")
        if self.quote != "INR":
            raise ValueError("Quote must be INR for canonical pairs.")
        if self.type == InstrumentType.CASH:
            if any([self.expiry_yyyymmdd, self.strike, self.right]):
                raise ValueError("Cash instruments must not include expiry, strike, or right.")
            return
        if not self.expiry_yyyymmdd or not re.fullmatch(r"\d{8}", self.expiry_yyyymmdd):
            raise ValueError("Expiry must be an 8-digit YYYYMMDD value for FUT/OPT.")
        if self.type == InstrumentType.FUT:
            if any([self.strike, self.right]):
                raise ValueError("Futures must not include strike or right.")
            return
        if self.type == InstrumentType.OPT:
            if self.strike is None:
                raise ValueError("Options require a strike price.")
            if self.right not in {"CE", "PE"}:
                raise ValueError("Options require right to be CE or PE.")
            return
        raise ValueError(f"Unsupported instrument type: {self.type}")


def _format_strike(strike: float) -> str:
    if float(strike).is_integer():
        return str(int(strike))
    return f"{strike:g}"


def parse_pair(pair: str) -> InstrumentSpec:
    """
    Parse a canonical pair string into an InstrumentSpec.

    Accepts only canonical forms:
      - Cash: UNDERLYING/INR
      - Futures: UNDERLYING-YYYYMMDD-FUT/INR
      - Options: UNDERLYING-YYYYMMDD-STRIKE-CE/INR
    """

    cash_match = re.fullmatch(r"(?P<underlying>[A-Z0-9]+)\/INR", pair, re.IGNORECASE)
    if cash_match:
        spec = InstrumentSpec(
            type=InstrumentType.CASH,
            underlying=cash_match.group("underlying").upper(),
        )
        spec.validate()
        return spec

    fut_match = re.fullmatch(
        r"(?P<underlying>[A-Z0-9]+)-(?P<expiry>\d{8})-FUT/INR",
        pair,
        re.IGNORECASE,
    )
    if fut_match:
        spec = InstrumentSpec(
            type=InstrumentType.FUT,
            underlying=fut_match.group("underlying").upper(),
            expiry_yyyymmdd=fut_match.group("expiry"),
        )
        spec.validate()
        return spec

    opt_match = re.fullmatch(
        r"(?P<underlying>[A-Z0-9]+)-(?P<expiry>\d{8})-(?P<strike>\d+(?:\.\d+)?)-(?P<right>CE|PE)/INR",
        pair,
        re.IGNORECASE,
    )
    if opt_match:
        spec = InstrumentSpec(
            type=InstrumentType.OPT,
            underlying=opt_match.group("underlying").upper(),
            expiry_yyyymmdd=opt_match.group("expiry"),
            strike=float(opt_match.group("strike")),
            right=opt_match.group("right").upper(),
        )
        spec.validate()
        return spec

    raise ValueError(f"Pair does not match canonical schema: {pair}")


def format_pair(spec: InstrumentSpec) -> str:
    """
    Format an InstrumentSpec into its canonical string.
    """

    spec.validate()
    underlying = spec.underlying.upper()
    if spec.type == InstrumentType.CASH:
        return f"{underlying}/INR"
    if spec.type == InstrumentType.FUT:
        return f"{underlying}-{spec.expiry_yyyymmdd}-FUT/INR"
    if spec.type == InstrumentType.OPT:
        strike = _format_strike(float(spec.strike))
        return f"{underlying}-{spec.expiry_yyyymmdd}-{strike}-{spec.right}/INR"
    raise ValueError(f"Unsupported instrument type: {spec.type}")
