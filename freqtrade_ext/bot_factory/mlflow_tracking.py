from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from freqtrade_ext.bot_factory.backtest_results import BacktestMetrics


def log_backtest_to_mlflow(
    metrics: BacktestMetrics,
    run_dir: Path,
    *,
    tracking_uri: str | None = None,
    experiment_name: str = "bot_factory_backtests",
) -> dict[str, Any]:
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{metrics.strategy_name}_{run_dir.name}") as active_run:
        mlflow.log_param("strategy_name", metrics.strategy_name)
        mlflow.log_param("run_id", run_dir.name)
        if metrics.backtest_start:
            mlflow.log_param("backtest_start", metrics.backtest_start)
        if metrics.backtest_end:
            mlflow.log_param("backtest_end", metrics.backtest_end)

        for key, value in asdict(metrics).items():
            if isinstance(value, bool):
                continue
            if isinstance(value, int | float):
                mlflow.log_metric(key, float(value))

        for artifact_name in (
            "metrics.json",
            "report.md",
            "trades.csv",
            "static_check.json",
            "result.json",
        ):
            artifact_path = run_dir / artifact_name
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))

        run = active_run.info
        return {
            "run_id": run.run_id,
            "experiment_id": run.experiment_id,
            "tracking_uri": mlflow.get_tracking_uri(),
            "artifact_uri": run.artifact_uri,
        }
