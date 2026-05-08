from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


@dataclass(frozen=True)
class FreqAIPredictionDiagnosticsInputs:
    root_dir: Path
    generated_metadata_path: Path
    predictions_dir: Path
    output_root: Path = Path("registry/strategies/diagnostics")
    diagnostics_id: str | None = None
    signal_diagnostics_path: Path | None = None
    freqai_metadata_path: Path | None = None
    training_manifest_path: Path | None = None
    reviewer_notes: Sequence[str] = field(default_factory=list)


def diagnose_freqai_predictions(inputs: FreqAIPredictionDiagnosticsInputs) -> dict[str, Any]:
    root = inputs.root_dir.resolve()
    metadata_path = _resolve_inside(inputs.generated_metadata_path, root)
    predictions_dir = _resolve_inside(inputs.predictions_dir, root)
    signal_diagnostics = (
        _load_json(_resolve_inside(inputs.signal_diagnostics_path, root))
        if inputs.signal_diagnostics_path
        else {}
    )
    freqai_metadata = (
        _load_json(_resolve_inside(inputs.freqai_metadata_path, root))
        if inputs.freqai_metadata_path
        else {}
    )
    training_manifest = (
        _load_json(_resolve_inside(inputs.training_manifest_path, root))
        if inputs.training_manifest_path
        else {}
    )
    metadata = _load_json(metadata_path)
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    diagnostics_id = inputs.diagnostics_id or _diagnostics_id(generated_at)
    strategy_name = str(metadata.get("strategy_name") or metadata.get("strategy_class_name") or "unknown")
    candidate_id = str(metadata.get("candidate_id") or "unknown_candidate")
    target_definition = str(metadata.get("target_definition") or "future_return")
    expected_target_column = f"&-{target_definition}"
    expected_freqai_identifier = _string_or_none(metadata.get("freqai_identifier"))
    observed_freqai_identifiers = _observed_freqai_identifiers(
        freqai_metadata,
        training_manifest,
    )
    freqai_identifier_match = _freqai_identifier_match(
        expected_freqai_identifier,
        observed_freqai_identifiers,
    )
    prediction_threshold = _number(metadata.get("prediction_threshold")) or 0.0
    prediction_files = _prediction_files(predictions_dir)
    dataframe = _load_prediction_frames(prediction_files)
    model_label_columns = _model_label_columns(predictions_dir.parent)
    target_columns = _target_columns(dataframe.columns if dataframe is not None else [])
    expected_present = dataframe is not None and expected_target_column in dataframe.columns
    do_predict_summary = _do_predict_summary(dataframe)
    target_summary = (
        _prediction_column_summary(
            dataframe,
            expected_target_column,
            prediction_threshold,
            do_predict_column_present=do_predict_summary["present"],
        )
        if expected_present and dataframe is not None
        else None
    )
    alternate_target_summaries = {
        column: _prediction_column_summary(
            dataframe,
            column,
            prediction_threshold,
            do_predict_column_present=do_predict_summary["present"],
        )
        for column in target_columns
        if dataframe is not None and column != expected_target_column
    }
    diagnosis_codes = _diagnosis_codes(
        prediction_file_count=len(prediction_files),
        expected_present=expected_present,
        target_columns=target_columns,
        expected_target_column=expected_target_column,
        model_label_columns=model_label_columns,
        target_summary=target_summary,
        alternate_target_summaries=alternate_target_summaries,
        do_predict_summary=do_predict_summary,
        freqai_identifier_match=freqai_identifier_match,
    )
    row_count = int(len(dataframe)) if dataframe is not None else 0
    return {
        "generated_at": generated_at,
        "factory": "freqai_prediction_diagnostics",
        "diagnostics_id": diagnostics_id,
        "status": "completed" if prediction_files else "blocked",
        "strategy_name": strategy_name,
        "candidate_id": candidate_id,
        "generated_metadata_path": _rel(metadata_path, root),
        "predictions_dir": _rel(predictions_dir, root),
        "freqai_metadata_path": _rel(_resolve_inside(inputs.freqai_metadata_path, root), root)
        if inputs.freqai_metadata_path
        else None,
        "training_manifest_path": _rel(_resolve_inside(inputs.training_manifest_path, root), root)
        if inputs.training_manifest_path
        else None,
        "signal_diagnostics_path": _rel(_resolve_inside(inputs.signal_diagnostics_path, root), root)
        if inputs.signal_diagnostics_path
        else None,
        "generator_mode": metadata.get("generator_mode"),
        "target_definition": target_definition,
        "expected_target_column": expected_target_column,
        "expected_freqai_identifier": expected_freqai_identifier,
        "observed_freqai_identifiers": observed_freqai_identifiers,
        "freqai_identifier_match": freqai_identifier_match,
        "prediction_threshold": prediction_threshold,
        "prediction_file_count": len(prediction_files),
        "prediction_files": [_rel(path, root) for path in prediction_files],
        "row_count": row_count,
        "date_range": _date_range(dataframe),
        "target_columns": target_columns,
        "model_label_columns": model_label_columns,
        "expected_target_column_present": expected_present,
        "target_summary": target_summary,
        "alternate_target_summaries": alternate_target_summaries,
        "do_predict_summary": do_predict_summary,
        "signal_diagnostics_summary": _signal_diagnostics_summary(signal_diagnostics),
        "freqai_metadata_summary": _freqai_metadata_summary(freqai_metadata),
        "training_manifest_summary": _training_manifest_summary(training_manifest),
        "diagnosis_codes": diagnosis_codes,
        "recommended_actions": _recommended_actions(
            expected_present=expected_present,
            target_columns=target_columns,
            expected_target_column=expected_target_column,
            model_label_columns=model_label_columns,
            freqai_identifier_match=freqai_identifier_match,
        ),
        "reviewer_notes": list(inputs.reviewer_notes),
        "safety_scope": {
            "historical_only": True,
            "backtest_started": False,
            "training_started": False,
            "paper_trading_started": False,
            "dry_run_trading_started": False,
            "live_trading": False,
            "exchange_order_placement": False,
            "process_control": False,
            "prediction_artifacts_read_only": True,
            "local_artifacts_source_of_truth": True,
        },
    }


def write_freqai_prediction_diagnostics_artifacts(
    diagnostics: dict[str, Any], *, root_dir: Path, output_root: Path
) -> tuple[Path, Path]:
    root = root_dir.resolve()
    strategy = _safe_path_component(str(diagnostics.get("strategy_name") or "unknown_strategy"))
    candidate = _safe_path_component(str(diagnostics.get("candidate_id") or "unknown_candidate"))
    diagnostics_id = _safe_path_component(str(diagnostics.get("diagnostics_id") or "diagnostics"))
    out_dir = _resolve_inside(output_root, root) / strategy / candidate / diagnostics_id
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "freqai_prediction_diagnostics.json"
    report_path = out_dir / "freqai_prediction_diagnostics_report.md"
    json_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(_render_report(diagnostics), encoding="utf-8")
    return json_path, report_path


def _prediction_files(predictions_dir: Path) -> list[Path]:
    if not predictions_dir.is_dir():
        return []
    return sorted(
        [
            path
            for path in predictions_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".feather", ".csv"}
        ]
    )


def _load_prediction_frames(paths: Sequence[Path]) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if path.suffix.lower() == ".feather":
            frame = pd.read_feather(path)
        else:
            frame = pd.read_csv(path)
        frame = frame.copy()
        frame["prediction_source_file"] = path.name
        frames.append(frame)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"], utc=True, errors="coerce")
        combined = combined.sort_values("date").reset_index(drop=True)
    return combined


def _model_label_columns(models_dir: Path) -> list[str]:
    labels: set[str] = set()
    if not models_dir.is_dir():
        return []
    for path in sorted(models_dir.glob("sub-train-*/*_metadata.json")):
        try:
            payload = _load_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        for label in payload.get("label_list") or []:
            if isinstance(label, str):
                labels.add(label)
        for key in (payload.get("labels_mean") or {}):
            labels.add(str(key))
    return sorted(labels)


def _target_columns(columns: Sequence[Any]) -> list[str]:
    result: list[str] = []
    for column in columns:
        name = str(column)
        if not name.startswith("&-"):
            continue
        if name.endswith("_mean") or name.endswith("_std"):
            continue
        result.append(name)
    return sorted(dict.fromkeys(result))


def _prediction_column_summary(
    dataframe: pd.DataFrame,
    column: str,
    threshold: float,
    *,
    do_predict_column_present: bool,
) -> dict[str, Any]:
    series = pd.to_numeric(dataframe[column], errors="coerce")
    non_null = series.dropna()
    above = series > threshold
    summary: dict[str, Any] = {
        "column": column,
        "count": int(non_null.count()),
        "missing_count": int(series.isna().sum()),
        "min": _rounded(non_null.min()) if not non_null.empty else None,
        "max": _rounded(non_null.max()) if not non_null.empty else None,
        "mean": _rounded(non_null.mean()) if not non_null.empty else None,
        "median": _rounded(non_null.median()) if not non_null.empty else None,
        "above_threshold_count": int(above.sum()),
        "above_threshold_ratio": _ratio(int(above.sum()), len(series)),
    }
    if do_predict_column_present:
        do_predict = pd.to_numeric(dataframe["do_predict"], errors="coerce").fillna(0) > 0
        summary["do_predict_and_above_threshold_count"] = int((do_predict & above).sum())
        summary["do_predict_and_above_threshold_ratio"] = _ratio(
            int((do_predict & above).sum()), int(do_predict.sum())
        )
    return summary


def _do_predict_summary(dataframe: pd.DataFrame | None) -> dict[str, Any]:
    if dataframe is None or "do_predict" not in dataframe.columns:
        return {"present": False}
    series = pd.to_numeric(dataframe["do_predict"], errors="coerce").fillna(0)
    positive = int((series > 0).sum())
    return {
        "present": True,
        "positive_count": positive,
        "non_positive_count": int(len(series) - positive),
        "positive_ratio": _ratio(positive, len(series)),
    }


def _diagnosis_codes(
    *,
    prediction_file_count: int,
    expected_present: bool,
    target_columns: Sequence[str],
    expected_target_column: str,
    model_label_columns: Sequence[str],
    target_summary: dict[str, Any] | None,
    alternate_target_summaries: dict[str, dict[str, Any]],
    do_predict_summary: dict[str, Any],
    freqai_identifier_match: bool | None,
) -> list[str]:
    codes: list[str] = []
    if prediction_file_count:
        codes.append("PREDICTION_FILES_PRESENT")
    else:
        codes.append("NO_PREDICTION_FILES")
    if expected_present:
        codes.append("EXPECTED_TARGET_PREDICTION_PRESENT")
    else:
        codes.append("EXPECTED_TARGET_PREDICTION_MISSING")
    if target_columns and not expected_present:
        codes.append("ALTERNATE_TARGET_PREDICTIONS_PRESENT")
        codes.append("PREDICTION_TARGET_MISMATCH")
    if model_label_columns and expected_target_column not in model_label_columns:
        codes.append("MODEL_LABEL_MISMATCH")
    if target_summary and int(target_summary.get("above_threshold_count") or 0) > 0:
        codes.append("EXPECTED_TARGET_ABOVE_THRESHOLD")
    if any(
        int(summary.get("above_threshold_count") or 0) > 0
        for summary in alternate_target_summaries.values()
    ):
        codes.append("ALTERNATE_TARGET_ABOVE_THRESHOLD")
    if do_predict_summary.get("present"):
        if do_predict_summary.get("positive_ratio") == 1.0:
            codes.append("DO_PREDICT_ALL_VALID")
        else:
            codes.append("DO_PREDICT_PARTIAL")
    if freqai_identifier_match is True:
        codes.append("FREQAI_IDENTIFIER_MATCH")
    elif freqai_identifier_match is False:
        codes.append("FREQAI_IDENTIFIER_MISMATCH")
    return list(dict.fromkeys(codes))


def _recommended_actions(
    *,
    expected_present: bool,
    target_columns: Sequence[str],
    expected_target_column: str,
    model_label_columns: Sequence[str],
    freqai_identifier_match: bool | None,
) -> list[str]:
    actions: list[str] = [
        "Do not judge or tune the ML threshold until prediction target consistency is verified.",
        "Keep local prediction diagnostics as evidence; do not start paper, dry-run, or live trading.",
    ]
    if not expected_present:
        actions.append(
            f"Regenerate or rerun FreqAI with predictions for {expected_target_column} before "
            "treating the ML gate as a strategy failure."
        )
    if target_columns and not expected_present:
        actions.append(
            "Use a candidate-specific FreqAI identifier or an explicit non-destructive cache policy "
            "so stale prediction targets cannot be reused across generated candidates."
        )
    if model_label_columns and expected_target_column not in model_label_columns:
        actions.append(
            "Align generated target_definition, set_freqai_targets label names, and FreqAI model "
            "label_list before the next hybrid candidate evaluation."
        )
    if freqai_identifier_match is False:
        actions.append(
            "Use the generated candidate-specific FreqAI identifier for historical, walk-forward, "
            "and training wrappers before reading prediction artifacts."
        )
    return actions


def _observed_freqai_identifiers(
    freqai_metadata: dict[str, Any],
    training_manifest: dict[str, Any],
) -> list[str]:
    identifiers: list[str] = []
    for payload in (freqai_metadata, training_manifest):
        value = _string_or_none(payload.get("freqai_identifier"))
        if value:
            identifiers.append(value)
    return list(dict.fromkeys(identifiers))


def _freqai_identifier_match(
    expected: str | None,
    observed: Sequence[str],
) -> bool | None:
    if not expected or not observed:
        return None
    return expected in set(observed)


def _signal_diagnostics_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {"available": False}
    return {
        "available": True,
        "diagnostics_id": payload.get("diagnostics_id"),
        "entry_count": payload.get("entry_count"),
        "diagnosis_codes": list(payload.get("diagnosis_codes") or []),
        "first_zero_component": payload.get("first_zero_component"),
    }


def _freqai_metadata_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {"available": False}
    return {
        "available": True,
        "status": payload.get("status"),
        "freqai_identifier": payload.get("freqai_identifier"),
        "freqaimodel": payload.get("freqaimodel"),
        "timerange": payload.get("timerange"),
    }


def _training_manifest_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {"available": False}
    return {
        "available": True,
        "status": payload.get("status"),
        "recommendation": payload.get("recommendation"),
        "summary": payload.get("summary"),
    }


def _date_range(dataframe: pd.DataFrame | None) -> dict[str, str | None]:
    if dataframe is None or "date" not in dataframe.columns or dataframe["date"].dropna().empty:
        return {"start": None, "end": None}
    return {
        "start": dataframe["date"].min().isoformat(),
        "end": dataframe["date"].max().isoformat(),
    }


def _render_report(diagnostics: dict[str, Any]) -> str:
    target_summary = diagnostics.get("target_summary") or {}
    lines = [
        "# FreqAI Prediction Diagnostics",
        "",
        f"- diagnostics_id: {diagnostics.get('diagnostics_id')}",
        f"- strategy: {diagnostics.get('strategy_name')}",
        f"- candidate_id: {diagnostics.get('candidate_id')}",
        f"- status: {diagnostics.get('status')}",
        f"- expected_target_column: {diagnostics.get('expected_target_column')}",
        f"- expected_target_column_present: {diagnostics.get('expected_target_column_present')}",
        f"- expected_freqai_identifier: {diagnostics.get('expected_freqai_identifier') or 'n/a'}",
        f"- observed_freqai_identifiers: {', '.join(diagnostics.get('observed_freqai_identifiers') or []) or 'n/a'}",
        f"- prediction_file_count: {diagnostics.get('prediction_file_count')}",
        f"- row_count: {diagnostics.get('row_count')}",
        f"- diagnosis_codes: {', '.join(diagnostics.get('diagnosis_codes') or [])}",
        "",
        "## Expected Target",
        "",
    ]
    if target_summary:
        lines.append(
            "- "
            f"above_threshold={target_summary.get('above_threshold_count')} "
            f"mean={target_summary.get('mean')} max={target_summary.get('max')}"
        )
    else:
        lines.append("- Expected target prediction column was not present.")
    lines.extend(["", "## Alternate Targets", ""])
    alternate = diagnostics.get("alternate_target_summaries") or {}
    for column, summary in alternate.items():
        lines.append(
            "- "
            f"{column}: rows={summary.get('count')}, "
            f"above_threshold={summary.get('above_threshold_count')}, "
            f"mean={summary.get('mean')}, max={summary.get('max')}"
        )
    if not alternate:
        lines.append("- None.")
    lines.extend(["", "## Recommended Actions", ""])
    lines.extend([f"- {item}" for item in diagnostics.get("recommended_actions", [])])
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- Diagnostic only; no backtest, training, paper, dry-run, live, exchange order, or process-control command is started.",
            "",
        ]
    )
    return "\n".join(lines)


def _ratio(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total


def _rounded(value: Any) -> float | None:
    try:
        return round(float(value), 12)
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _diagnostics_id(generated_at: str) -> str:
    parsed = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    return parsed.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _resolve_inside(path: Path | None, root: Path) -> Path:
    if path is None:
        raise ValueError("Path is required.")
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path must resolve inside the workspace: {path}") from exc
    return resolved


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _safe_path_component(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return cleaned.strip("._") or "unknown"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return payload
