from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class ResearchSelectionTemplateInputs:
    root_dir: Path
    causal_failure_map_path: Path
    output_root: Path = Path("registry/strategies/research_decisions")
    template_id: str | None = None
    reviewer_notes: Sequence[str] = field(default_factory=list)
    created_by_agent: str = "codex"
    created_at: str | None = None
    command: Sequence[str] = field(default_factory=list)


def build_research_selection_template(
    inputs: ResearchSelectionTemplateInputs,
) -> dict[str, Any]:
    root = inputs.root_dir.resolve()
    generated_at = inputs.created_at or datetime.now(UTC).replace(microsecond=0).isoformat()
    template_id = inputs.template_id or _template_id(generated_at)
    map_path = _resolve_inside(inputs.causal_failure_map_path, root)
    causal_map, map_error = _load_json(map_path)
    guidance = (
        causal_map.get("research_selection_guidance", {})
        if isinstance(causal_map, dict)
        else {}
    )
    checks = [
        _check("causal_failure_map_file_present", map_path.is_file(), {"path": _rel(map_path, root)}),
        _check(
            "causal_failure_map_parseable",
            isinstance(causal_map, dict) and map_error is None,
            {"error": map_error},
        ),
        _check(
            "causal_failure_map_factory_valid",
            isinstance(causal_map, dict)
            and causal_map.get("factory") in {"candidate_failure_map", "causal_failure_map"},
            {
                "factory": causal_map.get("factory") if isinstance(causal_map, dict) else None,
                "valid_factories": ["candidate_failure_map", "causal_failure_map"],
            },
        ),
        _check(
            "research_decision_required",
            bool(guidance.get("requires_research_decision_before_proposal")),
            {
                "requires_research_decision_before_proposal": guidance.get(
                    "requires_research_decision_before_proposal"
                )
            },
        ),
    ]
    status = "completed" if all(check["status"] == "pass" for check in checks) else "blocked"
    categories = _required_categories(guidance)
    questions = _required_questions(guidance)
    local_rejections = _validated_local_rejections(guidance)
    return {
        "generated_at": generated_at,
        "factory": "research_selection_response_template",
        "template_id": template_id,
        "status": status,
        "causal_failure_map_path": _rel(map_path, root),
        "map_id": causal_map.get("map_id") if isinstance(causal_map, dict) else None,
        "source_synthesis_id": (
            causal_map.get("source_synthesis_id") if isinstance(causal_map, dict) else None
        ),
        "source_synthesis_path": (
            causal_map.get("source_synthesis_path") if isinstance(causal_map, dict) else None
        ),
        "minimum_research_selection_score": guidance.get("minimum_research_selection_score"),
        "requires_research_question_responses": bool(
            guidance.get("requires_research_question_responses")
        ),
        "required_causal_failure_response_count": len(categories),
        "required_causal_failure_responses": categories,
        "required_research_question_response_count": len(questions),
        "required_research_question_responses": questions,
        "validated_local_falsification_rejection_count": len(local_rejections),
        "validated_local_falsification_rejections": local_rejections,
        "blocked_next_actions": [str(item) for item in guidance.get("blocked_next_actions", [])],
        "cli_argument_templates": _cli_argument_templates(categories, questions),
        "research_selection_input_template": _research_selection_input_template(
            map_path=_rel(map_path, root),
            source_synthesis_path=(
                causal_map.get("source_synthesis_path")
                if isinstance(causal_map, dict)
                else None
            ),
            categories=categories,
            questions=questions,
        ),
        "select_research_thesis_input_json_command_template": (
            ".\\.venv\\Scripts\\python.exe "
            "scripts\\bot_factory_select_research_thesis.py "
            "--research-selection-input-json "
            "<filled-research-selection-input.json>"
        ),
        "select_research_thesis_command_template": (
            _select_research_thesis_command_template(
                map_path=_rel(map_path, root),
                source_synthesis_path=(
                    causal_map.get("source_synthesis_path")
                    if isinstance(causal_map, dict)
                    else None
                ),
                categories=categories,
                questions=questions,
            )
        ),
        "checks": checks,
        "blockers": [check for check in checks if check["status"] != "pass"],
        "reviewer_notes": [str(note) for note in inputs.reviewer_notes],
        "created_by_agent": str(inputs.created_by_agent),
        "command": list(inputs.command),
        "safety_scope": {
            "historical_only": True,
            "local_artifacts_source_of_truth": True,
            "strategy_code_generated": False,
            "backtest_started": False,
            "paper_trading_started": False,
            "dry_run_trading_started": False,
            "live_trading": False,
            "exchange_order_placement": False,
            "shorting": False,
            "leverage": 1.0,
            "process_control": False,
            "promotion_authorized_by_this_command": False,
        },
    }


def write_research_selection_template_artifacts(
    artifact: dict[str, Any], *, root_dir: Path, output_root: Path
) -> tuple[Path, Path]:
    root = root_dir.resolve()
    template_id = _safe_path_component(str(artifact.get("template_id") or "research_selection_template"))
    out_dir = _resolve_inside(output_root, root) / template_id
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "research_selection_response_template.json"
    report_path = out_dir / "research_selection_response_template.md"
    json_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(_render_report(artifact), encoding="utf-8")
    return json_path, report_path


def _required_categories(guidance: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in guidance.get("dominant_failure_categories", []) or []:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category") or "").strip()
        if not category:
            continue
        rows.append(
            {
                "category": category,
                "candidate_count": _int_or_none(item.get("candidate_count")),
                "response_key": category,
                "cli_argument": f"--causal-failure-response \"{category}=<substantive response>\"",
                "response_template": "",
            }
        )
    return rows


def _required_questions(guidance: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, question in enumerate(
        guidance.get("required_research_questions", []) or [],
        start=1,
    ):
        text = str(question or "").strip()
        if not text:
            continue
        rows.append(
            {
                "index": index,
                "question": text,
                "response_key": str(index),
                "cli_argument": f"--research-question-response \"{index}=<substantive response>\"",
                "response_template": "",
            }
        )
    return rows


def _validated_local_rejections(guidance: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in guidance.get("validated_local_falsification_rejections", []) or []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "thesis_id": item.get("thesis_id"),
                "mechanism_class": item.get("mechanism_class"),
                "net_edge_bps": item.get("net_edge_bps"),
                "profitable_windows_ratio": item.get("profitable_windows_ratio"),
                "profitable_calendar_windows_ratio": item.get(
                    "profitable_calendar_windows_ratio"
                ),
                "path": item.get("path"),
            }
        )
    return rows


def _cli_argument_templates(
    categories: Sequence[dict[str, Any]], questions: Sequence[dict[str, Any]]
) -> list[str]:
    return [
        *[str(item["cli_argument"]) for item in categories],
        *[str(item["cli_argument"]) for item in questions],
    ]


def _research_selection_input_template(
    *,
    map_path: str,
    source_synthesis_path: str | None,
    categories: Sequence[dict[str, Any]],
    questions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "failure_synthesis_json": source_synthesis_path or "",
        "causal_failure_map_json": map_path,
        "thesis_id": "",
        "thesis_family": "",
        "mechanism_class": "",
        "thesis_statement": "",
        "mechanism_summary": "",
        "novelty_rationale": "",
        "required_data": [],
        "local_data_paths": [],
        "local_data_quality_report_jsons": [],
        "structural_data_capability_report_jsons": [],
        "local_falsification_jsons": [],
        "prior_local_falsification_jsons": [],
        "edge_rationale": "",
        "transaction_cost_exposure": "",
        "falsification_plan": "",
        "stop_conditions": [],
        "research_references": [],
        "causal_failure_responses": {
            str(item["response_key"]): "" for item in categories
        },
        "research_question_responses": {
            str(item["response_key"]): "" for item in questions
        },
        "decision_id": "",
        "created_at": "",
        "reviewer_notes": [],
    }


def _select_research_thesis_command_template(
    *,
    map_path: str,
    source_synthesis_path: str | None,
    categories: Sequence[dict[str, Any]],
    questions: Sequence[dict[str, Any]],
) -> str:
    synthesis_path = source_synthesis_path or "<candidate_failure_synthesis.json>"
    lines = [
        ".\\.venv\\Scripts\\python.exe scripts\\bot_factory_select_research_thesis.py `",
        f"  --failure-synthesis-json {synthesis_path} `",
        f"  --causal-failure-map-json {map_path} `",
        "  --thesis-id <NEW_THESIS_ID> `",
        "  --thesis-family <NEW_THESIS_FAMILY> `",
        "  --mechanism-class <NEW_MECHANISM_CLASS> `",
        '  --thesis-statement "<theory-backed falsifiable statement>" `',
        '  --mechanism-summary "<mechanism summary>" `',
        (
            '  --novelty-rationale "<why this is outside failed thesis IDs, '
            'families, and local rejections>" `'
        ),
        '  --required-data "<local closed-candle data required>" `',
        "  --local-data-path <path-to-local-data> `",
        "  --local-data-quality-report-json <path-to-local-data-quality-report.json> `",
        (
            "  --structural-data-capability-report-json "
            "<path-to-structural-data-capability-report.json> `"
        ),
        "  --local-falsification-json <path-to-passing-local-falsification.json> `",
        "  --prior-local-falsification-json <path-to-prior-local-falsification.json> `",
        '  --edge-rationale "<why post-cost edge should exist>" `',
        '  --transaction-cost-exposure "<fee, slippage, and turnover exposure>" `',
        '  --falsification-plan "<local event/falsification plan>" `',
        '  --stop-condition "<hard stop condition>" `',
        (
            '  --research-reference \'{"reference_id":"<id>","title":"<title>",'
            '"source":"<source>","published_at":"<date-or-null>",'
            '"relevance":"<relevance>","motivated_thesis_ids":["<NEW_THESIS_ID>"]}\' `'
        ),
        "  --decision-id <NEW_DECISION_ID> `",
        "  --created-at <ISO8601_UTC_TIMESTAMP> `",
    ]
    lines.extend(f"  {item['cli_argument']} `" for item in categories)
    lines.extend(f"  {item['cli_argument']} `" for item in questions)
    if lines:
        lines[-1] = lines[-1].removesuffix(" `")
    return "\n".join(lines)


def _render_report(artifact: dict[str, Any]) -> str:
    lines = [
        "# Research Selection Response Template",
        "",
        f"- template_id: {artifact.get('template_id')}",
        f"- status: {artifact.get('status')}",
        f"- causal_failure_map_path: {artifact.get('causal_failure_map_path')}",
        f"- map_id: {artifact.get('map_id')}",
        f"- source_synthesis_id: {artifact.get('source_synthesis_id')}",
        f"- minimum_research_selection_score: {artifact.get('minimum_research_selection_score')}",
        "",
        "## Required Causal Failure Responses",
        "",
    ]
    for item in artifact.get("required_causal_failure_responses", []) or []:
        lines.append(
            f"- {item.get('category')} "
            f"(candidate_count={item.get('candidate_count')}): "
            f"{item.get('cli_argument')}"
        )
    if not artifact.get("required_causal_failure_responses"):
        lines.append("- None.")
    lines.extend(["", "## Required Research Question Responses", ""])
    for item in artifact.get("required_research_question_responses", []) or []:
        lines.append(f"- {item.get('index')}. {item.get('question')}")
        lines.append(f"  - {item.get('cli_argument')}")
    if not artifact.get("required_research_question_responses"):
        lines.append("- None.")
    lines.extend(["", "## Validated Local Rejections", ""])
    for item in artifact.get("validated_local_falsification_rejections", []) or []:
        lines.append(
            f"- {item.get('thesis_id')} / {item.get('mechanism_class')}: "
            f"net_edge_bps={item.get('net_edge_bps')}, "
            f"profitable_calendar_windows_ratio={item.get('profitable_calendar_windows_ratio')}"
        )
    if not artifact.get("validated_local_falsification_rejections"):
        lines.append("- None.")
    lines.extend(["", "## Blocked Next Actions", ""])
    lines.extend([f"- {item}" for item in artifact.get("blocked_next_actions", [])] or ["- None."])
    command_template = artifact.get("select_research_thesis_command_template")
    lines.extend(
        [
            "",
            "## Select Research Thesis Input JSON Command Template",
            "",
            "```powershell",
            str(artifact.get("select_research_thesis_input_json_command_template") or ""),
            "```",
            "",
            "## Select Research Thesis Command Template",
            "",
            "```powershell",
            str(command_template or ""),
            "```",
        ]
    )
    input_template = artifact.get("research_selection_input_template")
    lines.extend(
        [
            "",
            "## Research Selection Input JSON Template",
            "",
            "```json",
            json.dumps(input_template or {}, indent=2, ensure_ascii=False),
            "```",
        ]
    )
    lines.extend(["", "## Checks", ""])
    lines.extend(
        [f"- {item.get('name')}: {item.get('status')}" for item in artifact.get("checks", [])]
        or ["- None."]
    )
    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "file_not_found"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive parse detail
        return None, str(exc)
    if not isinstance(payload, dict):
        return None, "json_root_not_object"
    return payload, None


def _check(name: str, passed: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "details": details,
    }


def _template_id(generated_at: str) -> str:
    stamp = generated_at.replace("+00:00", "Z").replace(":", "").replace("-", "")
    return f"{stamp}_research_selection_response_template"


def _resolve_inside(path: Path, root: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path must resolve inside workspace: {path}") from exc
    return resolved


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def _safe_path_component(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
