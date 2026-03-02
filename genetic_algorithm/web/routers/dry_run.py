"""
Dry-Run API — launch paper-trading sessions from evolved strategies.

Generates strategy code, writes it to user_data/strategies/, and spawns
a freqtrade trade subprocess in dry-run mode.  Log output is captured
in a deque and served via the status endpoint.
"""

from __future__ import annotations

import collections
import logging
import os
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, HTTPException

from genetic_algorithm.web.models.strategy import DryRunRequest, DryRunStatus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/dry-run", tags=["dry-run"])

# In-memory registry of dry-run sessions
_sessions: Dict[str, "_DryRunSession"] = {}
_lock = threading.Lock()

STRATEGY_DIR = Path("user_data/strategies")
LOG_LINES = 200


class _DryRunSession:
    """Manages a single freqtrade subprocess."""

    __slots__ = (
        "dry_run_id",
        "strategy_name",
        "proc",
        "started_at",
        "log_buf",
        "error",
        "_reader_thread",
    )

    def __init__(
        self,
        dry_run_id: str,
        strategy_name: str,
        proc: subprocess.Popen,
    ):
        self.dry_run_id = dry_run_id
        self.strategy_name = strategy_name
        self.proc = proc
        self.started_at = time.time()
        self.log_buf: collections.deque = collections.deque(maxlen=LOG_LINES)
        self.error: str | None = None

        # Background thread to stream stdout/stderr into log_buf
        self._reader_thread = threading.Thread(
            target=self._read_output, daemon=True, name=f"dryrun-log-{dry_run_id}"
        )
        self._reader_thread.start()

    # ── helpers ────────────────────────────────────────────────────

    def _read_output(self) -> None:
        """Read lines from the subprocess stdout+stderr (merged)."""
        try:
            assert self.proc.stdout is not None
            for raw_line in self.proc.stdout:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
                self.log_buf.append(line)
        except Exception as exc:
            self.log_buf.append(f"[log reader error] {exc}")

    @property
    def status_str(self) -> str:
        rc = self.proc.poll()
        if rc is None:
            return "running"
        if rc == 0:
            return "stopped"
        if self.error:
            return "failed"
        return "stopped"

    def to_status(self) -> DryRunStatus:
        return DryRunStatus(
            dry_run_id=self.dry_run_id,
            status=self.status_str,
            strategy_name=self.strategy_name,
            pid=self.proc.pid,
            started_at=self.started_at,
            error=self.error,
            log_tail=list(self.log_buf),
        )

    def stop(self) -> None:
        """Send SIGTERM, wait briefly, then SIGKILL if still alive."""
        if self.proc.poll() is not None:
            return
        try:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        except ProcessLookupError:
            pass  # already dead


# ── Endpoints ──────────────────────────────────────────────────────


@router.post("", response_model=DryRunStatus)
async def start_dry_run(body: DryRunRequest):
    """Launch a new dry-run trading session."""
    try:
        from genetic_algorithm.core.strategy_gene import StrategyGene
        from genetic_algorithm.strategies.generator import StrategyGenerator
        import yaml

        # Load default config for generator
        config_path = Path("genetic_algorithm/config/ga_config.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        gene = StrategyGene.from_dict(body.strategy_gene)
        generator = StrategyGenerator(config)
        code = generator.generate_strategy_code(gene)

        dry_run_id = f"dr_{uuid.uuid4().hex[:8]}"
        strategy_name = f"DryRun_{dry_run_id}"

        # Write strategy file
        STRATEGY_DIR.mkdir(parents=True, exist_ok=True)
        strat_file = STRATEGY_DIR / f"{strategy_name}.py"
        with open(strat_file, "w") as f:
            f.write(code)

        # Build freqtrade command
        pairs_csv = " ".join(body.pairs) if body.pairs else ""
        cmd: List[str] = [
            "freqtrade",
            "trade",
            "--strategy",
            strategy_name,
            "--strategy-path",
            str(STRATEGY_DIR),
            "--config",
            "user_data/config.json",
            "--dry-run",
            "--timeframe",
            body.timeframe,
        ]
        if body.pairs:
            for p in body.pairs:
                cmd.extend(["--pairs", p])

        logger.info("Launching dry-run %s: %s", dry_run_id, " ".join(cmd))

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        session = _DryRunSession(dry_run_id, strategy_name, proc)

        with _lock:
            _sessions[dry_run_id] = session

        return session.to_status()

    except Exception as e:
        logger.exception("Failed to start dry-run")
        raise HTTPException(500, f"Failed to start dry-run: {e}")


@router.get("", response_model=List[DryRunStatus])
async def list_dry_runs():
    """List all dry-run sessions (active and completed)."""
    with _lock:
        return [s.to_status() for s in _sessions.values()]


@router.get("/{dry_run_id}", response_model=DryRunStatus)
async def get_dry_run(dry_run_id: str):
    """Get status and recent log output for a dry-run session."""
    with _lock:
        session = _sessions.get(dry_run_id)
    if not session:
        raise HTTPException(404, f"Dry-run {dry_run_id} not found")
    return session.to_status()


@router.post("/{dry_run_id}/stop", response_model=DryRunStatus)
async def stop_dry_run(dry_run_id: str):
    """Stop a running dry-run session."""
    with _lock:
        session = _sessions.get(dry_run_id)
    if not session:
        raise HTTPException(404, f"Dry-run {dry_run_id} not found")
    session.stop()
    return session.to_status()
