import logging
from dataclasses import dataclass
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from pandas import DataFrame

from freqtrade.configuration.timerange import TimeRange
from freqtrade.data.converter import trim_dataframe
from freqtrade.data.history import load_data
from freqtrade.enums import CandleType
from freqtrade.playback_controller import PlaybackController
from freqtrade.rpc.api_server.deps import get_config, get_exchange, get_rpc_optional
from freqtrade.rpc.rpc import RPC, RPCException
from freqtrade.util import dt_now
from freqtrade.exceptions import OperationalException, StrategyError
from freqtrade.persistence.usedb_context import FtNoDBContext

logger = logging.getLogger(__name__)

# Private API, protected by authentication and webserver_mode dependency
router = APIRouter()


@dataclass
class LocalPlaybackSession:
    pair: str
    timeframe: str
    candle_type: CandleType
    raw: "DataFrame"
    start_pos: int
    strategy: object | None
    strategy_name: str
    controller: PlaybackController
    startup_candles: int
    last_data: dict | None = None
    last_limit: int | None = None
    last_selected_cols: list[str] | None = None

    def get_slice(self, limit: int | None = None):
        idx = self.controller.get_state()["current_step"]
        end_pos = min(self.start_pos + idx, len(self.raw) - 1)
        if limit:
            window = limit + self.startup_candles
            slice_start = max(0, end_pos - window + 1)
        else:
            slice_start = 0

        raw_slice = self.raw.iloc[slice_start : end_pos + 1].copy().reset_index(drop=True)
        if self.strategy:
            analyzed = self.strategy.analyze_ticker(raw_slice, {"pair": self.pair})
            if self.start_pos > slice_start:
                analyzed = analyzed.iloc[self.start_pos - slice_start :]
        else:
            if self.start_pos > slice_start:
                analyzed = raw_slice.iloc[self.start_pos - slice_start :]
            else:
                analyzed = raw_slice
        data = analyzed
        if limit:
            data = data.iloc[-limit:]
        return data

    def get_annotations(self, dataframe: "DataFrame") -> list:
        if not self.strategy:
            return []
        try:
            # Ensure we don't require a DB session in webserver-only playback
            with FtNoDBContext(self.timeframe):
                return self.strategy.ft_plot_annotations(pair=self.pair, dataframe=dataframe)
        except StrategyError as err:
            logger.warning("Playback annotations failed: %s", err)
            return []
        except Exception:
            logger.exception("Playback annotations failed.")
            return []


_local_playback_session: LocalPlaybackSession | None = None


class PlaybackInitRequest(BaseModel):
    pair: str
    timeframe: str
    timerange: str | None = None
    limit: int | None = 500
    candle_type: CandleType | None = None
    selected_cols: list[str] | None = None
    strategy: str | None = None


class PlaybackControlRequest(BaseModel):
    action: Literal[
        "play",
        "pause",
        "next",
        "previous",
        "reverse",
        "reset",
        "speed",
        "set_step",
    ]
    speed: float | None = None
    step: int | None = None
    limit: int | None = None
    selected_cols: list[str] | None = None


@router.post("/playback/init", tags=["Playback"])
def playback_init(
    req: PlaybackInitRequest,
    config=Depends(get_config),
    exchange=Depends(get_exchange),
    rpc: RPC | None = Depends(get_rpc_optional),
):
    try:
        if rpc:
            return rpc._rpc_playback_init(
                pair=req.pair,
                timeframe=req.timeframe,
                timerange=req.timerange,
                limit=req.limit,
                candle_type=req.candle_type,
                selected_cols=req.selected_cols,
            )

        global _local_playback_session
        timerange_parsed = TimeRange.parse_timerange(req.timerange or config.get("timerange"))

        from freqtrade.persistence.usedb_context import FtNoDBContext
        from freqtrade.resolvers.strategy_resolver import StrategyResolver
        from freqtrade.data.dataprovider import DataProvider

        with FtNoDBContext():
            config_override = dict(config)
            strategy_name = req.strategy or config.get("strategy")
            if not strategy_name:
                raise RPCException(
                    "No strategy set. Provide 'strategy' in the request body or set it in config.json."
                )
            config_override["strategy"] = strategy_name
            strategy = StrategyResolver.load_strategy(config_override)
            strategy.dp = DataProvider(config, exchange=exchange, pairlists=None)
            strategy.ft_bot_start()
            strategy_name = strategy.get_strategy_name()
            startup_candles = strategy.startup_candle_count

            candle_type = req.candle_type or config.get("candle_type_def", CandleType.SPOT)
            try:
                _data = load_data(
                    datadir=config["datadir"],
                    pairs=[req.pair],
                    timeframe=req.timeframe,
                    timerange=timerange_parsed,
                    data_format=config["dataformat_ohlcv"],
                    candle_type=candle_type,
                    startup_candles=startup_candles,
                    fail_without_data=True,
                )
            except OperationalException:
                raise RPCException(
                    "No data found. Verify pair/timeframe/candletype and datadir. "
                    "Use /api/v1/available_pairs to list available data."
                )
            if req.pair not in _data:
                raise RPCException(
                    f"No data for {req.pair}, {req.timeframe} in {timerange_parsed.timerange_str} found."
                )

            data = _data[req.pair]
            trimmed = trim_dataframe(data, timerange_parsed, startup_candles=startup_candles)
            if len(trimmed) > 0:
                start_date = trimmed["date"].iloc[0]
                start_idx = data.index[data["date"] == start_date]
                start_pos = int(start_idx[0]) if len(start_idx) > 0 else max(0, len(data) - len(trimmed))
            else:
                start_pos = 0

            controller = PlaybackController(total_steps=max(1, len(trimmed)), speed=1.0)
            controller.set_step(0)

            _local_playback_session = LocalPlaybackSession(
                pair=req.pair,
                timeframe=req.timeframe,
                candle_type=candle_type,
                raw=data,
                start_pos=start_pos,
                strategy=strategy,
                strategy_name=strategy_name,
                controller=controller,
                startup_candles=startup_candles,
            )

            return _local_playback_state(limit=req.limit, selected_cols=req.selected_cols)
    except RPCException as e:
        raise HTTPException(status_code=400, detail=e.message)


@router.post("/playback/control", tags=["Playback"])
def playback_control(
    req: PlaybackControlRequest,
    config=Depends(get_config),
    rpc: RPC | None = Depends(get_rpc_optional),
):
    try:
        if rpc:
            return rpc._rpc_playback_control(
                action=req.action,
                speed=req.speed,
                step=req.step,
                limit=req.limit,
                selected_cols=req.selected_cols,
            )

        return _local_playback_control(
            action=req.action,
            speed=req.speed,
            step=req.step,
            limit=req.limit,
            selected_cols=req.selected_cols,
        )
    except RPCException as e:
        raise HTTPException(status_code=400, detail=e.message)


@router.get("/playback/state", tags=["Playback"])
def playback_state(
    limit: int | None = 500,
    rpc: RPC | None = Depends(get_rpc_optional),
):
    try:
        if rpc:
            return rpc._rpc_playback_state(limit=limit)
        return _local_playback_state(limit=limit)
    except RPCException as e:
        raise HTTPException(status_code=400, detail=e.message)


def _local_playback_state(
    limit: int | None = None,
    selected_cols: list[str] | None = None,
):
    if not _local_playback_session:
        raise RPCException("Playback not initialized. Use /playback/init first.")

    from freqtrade.rpc.rpc import RPC as RPCHelper

    data_slice = _local_playback_session.get_slice(limit)
    annotations = _local_playback_session.get_annotations(data_slice)
    data = RPCHelper._convert_dataframe_to_dict(
        _local_playback_session.strategy_name,
        _local_playback_session.pair,
        _local_playback_session.timeframe,
        data_slice,
        dt_now(),
        selected_cols,
        annotations,
    )
    _local_playback_session.last_data = data
    _local_playback_session.last_limit = limit
    _local_playback_session.last_selected_cols = selected_cols
    return {
        "state": _local_playback_session.controller.get_state(),
        "data": data,
    }


def _local_playback_control(
    action: str,
    speed: float | None = None,
    step: int | None = None,
    limit: int | None = None,
    selected_cols: list[str] | None = None,
):
    if not _local_playback_session:
        raise RPCException("Playback not initialized. Use /playback/init first.")

    controller = _local_playback_session.controller
    if action == "play":
        controller.play()
    elif action == "pause":
        controller.pause()
    elif action in ("next", "step"):
        controller.next()
    elif action in ("previous", "reverse"):
        controller.previous()
    elif action == "reset":
        controller.reset()
    elif action == "speed":
        if speed is None:
            raise RPCException("Speed value is required for 'speed' action.")
        controller.set_speed(speed)
    elif action == "set_step":
        if step is None:
            raise RPCException("Step value is required for 'set_step' action.")
        controller.set_step(step)
    else:
        raise RPCException("Invalid playback action.")

    if (
        action == "speed"
        and _local_playback_session.last_data is not None
        and _local_playback_session.last_limit == limit
        and _local_playback_session.last_selected_cols == selected_cols
    ):
        return {
            "state": controller.get_state(),
            "data": _local_playback_session.last_data,
        }

    return _local_playback_state(limit=limit, selected_cols=selected_cols)
