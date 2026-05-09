from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


BYBIT_OPEN_INTEREST_DOC_URL = "https://bybit-exchange.github.io/docs/v5/market/open-interest"
BYBIT_LONG_SHORT_RATIO_DOC_URL = "https://bybit-exchange.github.io/docs/v5/market/long-short-ratio"
BYBIT_ALL_LIQUIDATION_WS_DOC_URL = (
    "https://bybit-exchange.github.io/docs/v5/websocket/public/all-liquidation"
)
BYBIT_ORDERBOOK_REST_DOC_URL = "https://bybit-exchange.github.io/docs/v5/market/orderbook"
BYBIT_ORDERBOOK_WS_DOC_URL = "https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook"


@dataclass(frozen=True)
class StructuralDataCapabilityInputs:
    root_dir: Path
    open_interest_path: Path | None = None
    open_interest_quality_report_paths: Sequence[Path] = ()
    long_short_ratio_path: Path | None = None
    long_short_ratio_quality_report_paths: Sequence[Path] = ()
    liquidation_paths: Sequence[Path] = ()
    liquidation_quality_report_paths: Sequence[Path] = ()
    order_book_paths: Sequence[Path] = ()
    order_book_quality_report_paths: Sequence[Path] = ()
    output_path: Path | None = None
    created_at: str | None = None
    command: Sequence[str] = ()


def build_structural_data_capability_report(
    inputs: StructuralDataCapabilityInputs,
) -> dict[str, Any]:
    root = inputs.root_dir.resolve()
    generated_at = inputs.created_at or datetime.now(UTC).isoformat()
    open_interest_path = _resolve_optional(inputs.open_interest_path, root)
    long_short_ratio_path = _resolve_optional(inputs.long_short_ratio_path, root)
    open_interest_quality_paths = [
        path
        for path in (
            _resolve_optional(candidate, root)
            for candidate in inputs.open_interest_quality_report_paths
        )
        if path is not None
    ]
    long_short_ratio_quality_paths = [
        path
        for path in (
            _resolve_optional(candidate, root)
            for candidate in inputs.long_short_ratio_quality_report_paths
        )
        if path is not None
    ]
    order_book_quality_paths = [
        path
        for path in (
            _resolve_optional(candidate, root)
            for candidate in inputs.order_book_quality_report_paths
        )
        if path is not None
    ]
    liquidation_quality_paths = [
        path
        for path in (
            _resolve_optional(candidate, root)
            for candidate in inputs.liquidation_quality_report_paths
        )
        if path is not None
    ]
    liquidation_paths = [
        path for path in (_resolve_optional(candidate, root) for candidate in inputs.liquidation_paths) if path is not None
    ]
    order_book_paths = [
        path for path in (_resolve_optional(candidate, root) for candidate in inputs.order_book_paths) if path is not None
    ]

    open_interest_quality = _quality_report_summary(open_interest_quality_paths, root)
    long_short_ratio_quality = _quality_report_summary(long_short_ratio_quality_paths, root)
    order_book_quality = _quality_report_summary(order_book_quality_paths, root)
    liquidation_quality = _quality_report_summary(liquidation_quality_paths, root)
    open_interest_present = bool(open_interest_path and open_interest_path.is_file())
    long_short_ratio_present = bool(long_short_ratio_path and long_short_ratio_path.is_file())
    open_interest_codegen_supported = open_interest_present and open_interest_quality["ok"]
    long_short_ratio_codegen_supported = (
        long_short_ratio_present and long_short_ratio_quality["ok"]
    )
    liquidation_present = any(path.is_file() for path in liquidation_paths)
    liquidation_local_quality_ok = liquidation_present and liquidation_quality["ok"]
    order_book_present = any(path.is_file() for path in order_book_paths)
    order_book_local_quality_ok = order_book_present and order_book_quality["ok"]

    capabilities = {
        "open_interest": {
            "local_data_present": open_interest_present,
            "local_paths": [_rel(open_interest_path, root)] if open_interest_path else [],
            "quality_reports": open_interest_quality,
            "local_quality_ok": open_interest_quality["ok"],
            "historical_download_supported": True,
            "local_event_supported": True,
            "research_selection_quality_gate_supported": True,
            "strategy_codegen_supported": open_interest_codegen_supported,
            "collection_mode": "historical_rest_market_data",
            "official_reference_urls": [BYBIT_OPEN_INTEREST_DOC_URL],
            "notes": [
                "Local open-interest research screens are supported when a passing quality report exists.",
                "Generated strategy code may use the local parquet only when the data file is present and its quality report passes.",
            ],
        },
        "long_short_ratio": {
            "local_data_present": long_short_ratio_present,
            "local_paths": [_rel(long_short_ratio_path, root)] if long_short_ratio_path else [],
            "quality_reports": long_short_ratio_quality,
            "local_quality_ok": long_short_ratio_quality["ok"],
            "historical_download_supported": True,
            "local_event_supported": True,
            "research_selection_quality_gate_supported": True,
            "strategy_codegen_supported": long_short_ratio_codegen_supported,
            "collection_mode": "historical_rest_market_data",
            "official_reference_urls": [BYBIT_LONG_SHORT_RATIO_DOC_URL],
            "notes": [
                "Local long/short account-ratio screens are supported when a passing quality report exists.",
                "Generated strategy code may use the local parquet only when the data file is present and its quality report passes.",
            ],
        },
        "liquidation": {
            "local_data_present": liquidation_present,
            "local_paths": [_rel(path, root) for path in liquidation_paths],
            "quality_reports": liquidation_quality,
            "local_quality_ok": liquidation_local_quality_ok,
            "historical_download_supported": False,
            "local_event_supported": True,
            "research_selection_quality_gate_supported": True,
            "strategy_codegen_supported": False,
            "collection_mode": "public_websocket_realtime_only",
            "official_reference_urls": [BYBIT_ALL_LIQUIDATION_WS_DOC_URL],
            "notes": [
                "Bybit documents all-liquidation as a public websocket stream, not a historical REST download.",
                "A local historical liquidation parquet may be quality-checked and used for local event or edge discovery features, but strategy codegen is not implemented for liquidation data yet.",
            ],
        },
        "order_book": {
            "local_data_present": order_book_present,
            "local_paths": [_rel(path, root) for path in order_book_paths],
            "quality_reports": order_book_quality,
            "local_quality_ok": order_book_local_quality_ok,
            "historical_download_supported": False,
            "local_event_supported": True,
            "research_selection_quality_gate_supported": True,
            "strategy_codegen_supported": False,
            "collection_mode": "current_snapshot_or_user_supplied_historical_snapshots",
            "official_reference_urls": [BYBIT_ORDERBOOK_REST_DOC_URL, BYBIT_ORDERBOOK_WS_DOC_URL],
            "notes": [
                "Bybit REST orderbook is a current snapshot and websocket orderbook is a live stream.",
                "A local timestamped snapshot parquet may be quality-checked and used for local event or edge discovery features, but strategy codegen is not implemented for order-book data yet.",
            ],
        },
    }
    local_research_usable = [
        name
        for name, capability in capabilities.items()
        if capability["local_data_present"]
        and capability["local_quality_ok"]
        and capability["local_event_supported"]
    ]
    blocked_without_new_data = [
        name
        for name, capability in capabilities.items()
        if not capability["local_data_present"] or not capability["local_quality_ok"]
    ]
    must_not_codegen = [
        name for name, capability in capabilities.items() if not capability["strategy_codegen_supported"]
    ]
    checks = [
        _check(
            "open_interest_local_quality_usable",
            "open_interest" in local_research_usable,
            {
                "local_data_present": open_interest_present,
                "quality_report_ok": open_interest_quality["ok"],
            },
        ),
        _check(
            "long_short_ratio_local_quality_usable",
            "long_short_ratio" in local_research_usable,
            {
                "local_data_present": long_short_ratio_present,
                "quality_report_ok": long_short_ratio_quality["ok"],
            },
        ),
        _check(
            "liquidation_historical_local_data_quality_usable",
            liquidation_local_quality_ok,
            {
                "historical_download_supported": False,
                "local_data_present": liquidation_present,
                "quality_report_ok": liquidation_quality["ok"],
                "collection_mode": "public_websocket_realtime_only",
            },
        ),
        _check(
            "order_book_historical_local_data_present",
            order_book_local_quality_ok,
            {
                "historical_download_supported": False,
                "local_data_present": order_book_present,
                "quality_report_ok": order_book_quality["ok"],
                "collection_mode": "current_snapshot_or_user_supplied_historical_snapshots",
            },
        ),
        _check(
            "structural_strategy_codegen_supported",
            bool(open_interest_codegen_supported or long_short_ratio_codegen_supported),
            {
                "supported": [
                    name
                    for name in ("open_interest", "long_short_ratio")
                    if capabilities[name]["strategy_codegen_supported"]
                ],
                "unsupported": must_not_codegen,
            },
        ),
    ]
    return {
        "factory": "structural_data_capability_report",
        "generated_at": generated_at,
        "command": list(inputs.command),
        "capabilities": capabilities,
        "proposal_guidance": {
            "local_research_usable": local_research_usable,
            "blocked_without_new_data": blocked_without_new_data,
            "must_not_codegen": must_not_codegen,
            "next_data_needed": [
                "historical_liquidation_data_with_quality_check"
                if not liquidation_local_quality_ok
                else "liquidation_strategy_codegen_variant_before_promotion",
                "order_book_local_event_features_before_research_selection"
                if not order_book_local_quality_ok
                else "order_book_strategy_codegen_variant_before_promotion",
            ],
            "do_not_continue_oi_threshold_retunes": True,
        },
        "checks": checks,
        "blockers": [check for check in checks if check["status"] == "blocked"],
        "safety_scope": {
            "api_key_required": False,
            "order_endpoint_used": False,
            "trade_or_paper_process_started": False,
            "historical_or_local_market_data_only": True,
        },
    }


def write_structural_data_capability_report(artifact: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def default_structural_data_capability_output_path() -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("registry") / "strategies" / "checks" / f"{ts}_structural_data_capabilities.json"


def _quality_report_summary(paths: Sequence[Path], root: Path) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    missing_reports: list[str] = []
    ok = bool(paths)
    for path in paths:
        rel_path = _rel(path, root)
        if not path.is_file():
            ok = False
            missing_reports.append(rel_path)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            ok = False
            reports.append({"path": rel_path, "ok": False, "error": str(exc)})
            continue
        report_ok = bool(payload.get("ok"))
        ok = ok and report_ok
        reports.append(
            {
                "path": rel_path,
                "ok": report_ok,
                "report_count": len(payload.get("reports") or []),
            }
        )
    return {"ok": ok, "reports": reports, "missing_reports": missing_reports}


def _resolve_optional(path: Path | None, root: Path) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def _rel(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def _check(name: str, passed: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if passed else "blocked",
        "severity": "blocker" if not passed else "info",
        "details": details,
    }
