"""
Backtest API — on-demand backtesting of individual strategies.

Runs backtests asynchronously and streams progress via WebSocket.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Dict

from fastapi import APIRouter, HTTPException, Request

from genetic_algorithm.web.models.strategy import BacktestRequest, BacktestResultModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/backtest", tags=["backtest"])

# In-memory store for backtest results (simple for now)
_backtests: Dict[str, BacktestResultModel] = {}
_backtest_lock = threading.Lock()
_active_count = 0


def _get_max_concurrent(request: Request) -> int:
    """Read max concurrent backtests from WebConfig, fallback to 2."""
    try:
        return request.app.state.web_config.max_concurrent_backtests
    except Exception:
        return 2


@router.post("", response_model=BacktestResultModel)
async def start_backtest(body: BacktestRequest, request: Request):
    """Start an on-demand backtest for a strategy."""
    global _active_count
    max_concurrent = _get_max_concurrent(request)

    with _backtest_lock:
        if _active_count >= max_concurrent:
            raise HTTPException(429, "Too many concurrent backtests")
        _active_count += 1

    backtest_id = f"bt_{uuid.uuid4().hex[:8]}"
    result = BacktestResultModel(backtest_id=backtest_id, status="running")

    with _backtest_lock:
        _backtests[backtest_id] = result

    # Run backtest in background thread
    thread = threading.Thread(
        target=_run_backtest,
        args=(backtest_id, body),
        daemon=True,
        name=f"backtest-{backtest_id}",
    )
    thread.start()

    return result


@router.get("/{backtest_id}", response_model=BacktestResultModel)
async def get_backtest(backtest_id: str):
    """Get backtest status / result (summary — trades truncated to first 500)."""
    with _backtest_lock:
        result = _backtests.get(backtest_id)
    if not result:
        raise HTTPException(404, f"Backtest {backtest_id} not found")
    return result


@router.get("/{backtest_id}/trades")
async def get_backtest_trades(
    backtest_id: str,
    offset: int = 0,
    limit: int = 100,
    pair: str | None = None,
):
    """
    Return trade-level detail for a completed backtest.

    Supports pagination (offset/limit) and optional pair filter.
    """
    with _backtest_lock:
        result = _backtests.get(backtest_id)
    if not result:
        raise HTTPException(404, f"Backtest {backtest_id} not found")
    if result.status != "completed" or not result.result:
        raise HTTPException(400, f"Backtest {backtest_id} is not completed yet (status: {result.status})")

    trades = result.result.get("trades", [])

    # Optional pair filter
    if pair:
        trades = [t for t in trades if t.get("pair") == pair]

    total = len(trades)
    page = trades[offset : offset + limit]

    return {
        "backtest_id": backtest_id,
        "total": total,
        "offset": offset,
        "limit": limit,
        "trades": page,
    }


def _run_backtest(backtest_id: str, body: BacktestRequest) -> None:
    """Execute a backtest in a background thread."""
    global _active_count
    try:
        from genetic_algorithm.core.strategy_gene import StrategyGene
        from genetic_algorithm.strategies.generator import StrategyGenerator
        from genetic_algorithm.evaluation.direct_backtester import DirectBacktester

        import yaml
        from pathlib import Path

        # Load default config for the backtester
        config_path = Path("genetic_algorithm/config/ga_config.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Override backtest params from request
        config["backtesting"]["timerange"] = body.timerange
        if body.pairs:
            config["backtesting"]["pairs"] = body.pairs
        config["backtesting"]["stake_amount"] = body.stake_amount
        config["backtesting"]["exchange"] = body.exchange

        # Build strategy
        gene = StrategyGene.from_dict(body.strategy_gene)
        generator = StrategyGenerator(config)
        code = generator.generate_strategy_code(gene)
        strategy_name = f"BacktestOnDemand_{backtest_id}"
        
        # Rename the class in generated code to match strategy_name
        # The generator creates a class like GAStrategy_Gen3_Ind5,
        # but we need it to match strategy_name for FreqTrade's result lookup
        import re
        class_pattern = r'class\s+(\w+)\s*\('
        match = re.search(class_pattern, code)
        if match:
            original_class = match.group(1)
            code = re.sub(
                rf'\b{original_class}\b',
                strategy_name,
                code
            )

        # Run backtest (with trade collection for visualization)
        backtester = DirectBacktester(config)
        bt_result = backtester.backtest_strategy_with_trades(code, strategy_name)

        # Build result
        with _backtest_lock:
            r = _backtests[backtest_id]
            if bt_result.success:
                r.status = "completed"
                r.progress = 1.0
                r.result = {
                    "total_profit": bt_result.total_profit,
                    "profit_percent": bt_result.profit_percent,
                    "total_trades": bt_result.total_trades,
                    "wins": bt_result.wins,
                    "losses": bt_result.losses,
                    "win_rate": bt_result.win_rate,
                    "max_drawdown": bt_result.max_drawdown,
                    "sharpe_ratio": bt_result.sharpe_ratio,
                    "sortino_ratio": bt_result.sortino_ratio,
                    "profit_factor": bt_result.profit_factor,
                    "avg_profit": bt_result.avg_profit,
                    "avg_duration": str(bt_result.avg_duration),
                    "timeframe": body.timeframe or gene.timeframe,
                    "exchange": config.get("backtesting", {}).get("exchange", "binance"),
                    "error_message": bt_result.error_message,
                    "trades": [
                        {
                            "pair": t.get("pair", ""),
                            "open_date": str(t.get("open_date", "")),
                            "close_date": str(t.get("close_date", "")),
                            "profit_ratio": t.get("profit_ratio", 0),
                            "profit_abs": t.get("profit_abs", 0),
                            "trade_duration": t.get("trade_duration", 0),
                            "is_short": t.get("is_short", False),
                        }
                        for t in (bt_result.trades or [])[:500]
                    ],
                }
            else:
                r.status = "failed"
                r.error = bt_result.error_message

    except Exception as e:
        logger.exception("Backtest %s failed", backtest_id)
        with _backtest_lock:
            r = _backtests.get(backtest_id)
            if r:
                r.status = "failed"
                r.error = str(e)
    finally:
        with _backtest_lock:
            _active_count = max(0, _active_count - 1)
