#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from freqtrade_ext.bot_factory.research_selection import (
    ResearchSelectionInputs,
    select_research_thesis,
    write_research_selection_artifacts,
)
from freqtrade_ext.bot_factory.strategy_proposals import (
    StrategyProposalResearchReference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select whether a new Bot Factory research thesis is worth moving "
            "toward proposal generation. This command writes local decision "
            "artifacts only; it does not generate code, run backtests, start "
            "paper/live trading, or manage any bot process."
        )
    )
    parser.add_argument(
        "--research-selection-input-json",
        default=None,
        help=(
            "Filled JSON input produced from "
            "research_selection_response_template.json. CLI arguments override "
            "or extend matching JSON fields."
        ),
    )
    parser.add_argument("--failure-synthesis-json", default=None)
    parser.add_argument("--thesis-id", default=None)
    parser.add_argument("--thesis-family", default=None)
    parser.add_argument("--mechanism-class", default=None)
    parser.add_argument("--thesis-statement", default=None)
    parser.add_argument("--mechanism-summary", default=None)
    parser.add_argument("--novelty-rationale", default=None)
    parser.add_argument("--required-data", action="append", default=[])
    parser.add_argument("--local-data-path", action="append", default=[])
    parser.add_argument(
        "--local-data-quality-report-json",
        action="append",
        default=[],
        help=(
            "Local JSON quality report for structural market data such as "
            "open-interest, liquidation, or order-book artifacts."
        ),
    )
    parser.add_argument(
        "--structural-data-capability-report-json",
        action="append",
        default=[],
        help=(
            "Local structural_data_capability_report JSON. Required for "
            "structural market-data theses so unavailable liquidation/order-book "
            "history cannot be bypassed with a generic quality report."
        ),
    )
    parser.add_argument(
        "--local-falsification-json",
        action="append",
        default=[],
        help=(
            "Local JSON evidence for pre-proposal falsification, such as "
            "cost/edge bps evidence for high-risk cost-sensitive maps."
        ),
    )
    parser.add_argument(
        "--prior-local-falsification-json",
        action="append",
        default=[],
        help=(
            "Previously generated local_falsification.json evidence. Failed or "
            "rejected artifacts that match the current thesis ID or mechanism "
            "class block repeated research selection."
        ),
    )
    parser.add_argument("--causal-failure-map-json", default=None)
    parser.add_argument(
        "--causal-failure-response",
        action="append",
        default=[],
        help=(
            "Response to a dominant causal failure category from the map, formatted "
            "as CATEGORY=RATIONALE or CATEGORY: RATIONALE. Repeat for each required "
            "category."
        ),
    )
    parser.add_argument(
        "--research-question-response",
        action="append",
        default=[],
        help=(
            "Response to a required research question from the causal failure "
            "map, formatted as 1=RATIONALE or QUESTION=RATIONALE. Repeat for "
            "each required question."
        ),
    )
    parser.add_argument("--edge-rationale", default=None)
    parser.add_argument("--transaction-cost-exposure", default=None)
    parser.add_argument("--falsification-plan", default=None)
    parser.add_argument("--stop-condition", action="append", default=[])
    parser.add_argument(
        "--research-reference",
        action="append",
        default=[],
        help=(
            "Structured reference JSON with reference_id, title, source, "
            "published_at, relevance, and motivated_thesis_ids. Prefix with @ "
            "to read a local JSON file."
        ),
    )
    parser.add_argument("--decision-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--reviewer-note", action="append", default=[])
    parser.add_argument("--created-at", default=None)
    return parser.parse_args()


def build_inputs_from_args(
    args: argparse.Namespace, *, root_dir: Path = ROOT_DIR
) -> ResearchSelectionInputs:
    payload = _research_selection_input_payload(
        getattr(args, "research_selection_input_json", None),
        root_dir=root_dir,
    )
    research_references = _research_reference_inputs(
        list(getattr(args, "research_reference", []) or []),
        root_dir=root_dir,
    )
    research_references.extend(
        _research_reference_payloads(
            _payload_list(payload, "research_references", "research_reference"),
            root_dir=root_dir,
        )
    )
    return ResearchSelectionInputs(
        root_dir=root_dir,
        failure_synthesis_path=Path(
            _required_scalar(
                args.failure_synthesis_json,
                payload,
                "failure_synthesis_json",
                "failure_synthesis_path",
                cli_name="--failure-synthesis-json",
            )
        ),
        thesis_id=_required_scalar(
            args.thesis_id, payload, "thesis_id", cli_name="--thesis-id"
        ),
        thesis_family=_required_scalar(
            args.thesis_family,
            payload,
            "thesis_family",
            cli_name="--thesis-family",
        ),
        mechanism_class=_required_scalar(
            args.mechanism_class,
            payload,
            "mechanism_class",
            cli_name="--mechanism-class",
        ),
        thesis_statement=_required_scalar(
            args.thesis_statement,
            payload,
            "thesis_statement",
            cli_name="--thesis-statement",
        ),
        mechanism_summary=_required_scalar(
            args.mechanism_summary,
            payload,
            "mechanism_summary",
            cli_name="--mechanism-summary",
        ),
        novelty_rationale=_required_scalar(
            args.novelty_rationale,
            payload,
            "novelty_rationale",
            cli_name="--novelty-rationale",
        ),
        required_data=_required_list(
            list(args.required_data or []),
            payload,
            "required_data",
            cli_name="--required-data",
        ),
        local_data_paths=[
            Path(path)
            for path in _merged_list(
                list(args.local_data_path or []),
                payload,
                "local_data_paths",
                "local_data_path",
            )
        ],
        local_data_quality_report_paths=[
            Path(path)
            for path in _merged_list(
                list(args.local_data_quality_report_json or []),
                payload,
                "local_data_quality_report_jsons",
                "local_data_quality_report_json",
                "local_data_quality_report_paths",
            )
        ],
        structural_data_capability_report_paths=[
            Path(path)
            for path in _merged_list(
                list(args.structural_data_capability_report_json or []),
                payload,
                "structural_data_capability_report_jsons",
                "structural_data_capability_report_json",
                "structural_data_capability_report_paths",
            )
        ],
        local_falsification_paths=[
            Path(path)
            for path in _merged_list(
                list(args.local_falsification_json or []),
                payload,
                "local_falsification_jsons",
                "local_falsification_json",
                "local_falsification_paths",
            )
        ],
        prior_local_falsification_paths=[
            Path(path)
            for path in _merged_list(
                list(args.prior_local_falsification_json or []),
                payload,
                "prior_local_falsification_jsons",
                "prior_local_falsification_json",
                "prior_local_falsification_paths",
            )
        ],
        causal_failure_map_path=Path(
            _first_scalar(
                args.causal_failure_map_json,
                payload,
                "causal_failure_map_json",
                "causal_failure_map_path",
            )
        )
        if _first_scalar(
            args.causal_failure_map_json,
            payload,
            "causal_failure_map_json",
            "causal_failure_map_path",
        )
        else None,
        causal_failure_responses=_response_values(
            list(args.causal_failure_response or []),
            payload,
            "causal_failure_responses",
            template_key="required_causal_failure_responses",
        ),
        research_question_responses=_response_values(
            list(args.research_question_response or []),
            payload,
            "research_question_responses",
            template_key="required_research_question_responses",
        ),
        edge_rationale=_required_scalar(
            args.edge_rationale,
            payload,
            "edge_rationale",
            cli_name="--edge-rationale",
        ),
        transaction_cost_exposure=_required_scalar(
            args.transaction_cost_exposure,
            payload,
            "transaction_cost_exposure",
            cli_name="--transaction-cost-exposure",
        ),
        falsification_plan=_required_scalar(
            args.falsification_plan,
            payload,
            "falsification_plan",
            cli_name="--falsification-plan",
        ),
        stop_conditions=_required_list(
            list(args.stop_condition or []),
            payload,
            "stop_conditions",
            "stop_condition",
            cli_name="--stop-condition",
        ),
        research_references=research_references,
        output_root=Path(
            _first_scalar(args.output_root, payload, "output_root")
            or "registry/strategies/research_decisions"
        ),
        decision_id=_first_scalar(args.decision_id, payload, "decision_id"),
        reviewer_notes=_merged_list(
            list(args.reviewer_note or []),
            payload,
            "reviewer_notes",
            "reviewer_note",
        ),
        created_at=_first_scalar(args.created_at, payload, "created_at"),
        command=sys.argv,
    )


def main() -> int:
    inputs = build_inputs_from_args(parse_args())
    decision = select_research_thesis(inputs)
    decision_path, report_path = write_research_selection_artifacts(
        decision,
        root_dir=ROOT_DIR,
        output_root=inputs.output_root,
    )
    print(
        json.dumps(
            {
                "research_decision_path": str(decision_path),
                "research_decision_report_path": str(report_path),
                "status": decision["status"],
                "proposal_generation_allowed": decision["proposal_generation_allowed"],
                "code_generation_allowed": decision["code_generation_allowed"],
                "research_selection_score": decision.get(
                    "research_selection_score", {}
                ).get("score"),
                "minimum_research_selection_score": decision.get(
                    "research_selection_score", {}
                ).get("minimum_score_required"),
                "blocker_count": len(decision["blockers"]),
                "deferral_count": len(decision["deferrals"]),
            },
            indent=2,
        )
    )
    return 0 if decision.get("status") == "approved_for_proposal_generation" else 1


def _research_reference_inputs(
    values: list[str] | None,
    *,
    root_dir: Path = ROOT_DIR,
) -> list[StrategyProposalResearchReference]:
    references: list[StrategyProposalResearchReference] = []
    for raw in values or []:
        text = _research_reference_text(raw, root_dir=root_dir)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--research-reference must be JSON or @path JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise SystemExit("--research-reference must decode to a JSON object")
        motivated = payload.get("motivated_thesis_ids", [])
        if isinstance(motivated, str):
            motivated = [motivated]
        references.append(
            StrategyProposalResearchReference(
                reference_id=str(payload.get("reference_id", "")),
                title=str(payload.get("title", "")),
                source=str(payload.get("source", "")),
                published_at=payload.get("published_at"),
                relevance=str(payload.get("relevance", "")),
                motivated_thesis_ids=[str(item) for item in motivated],
            )
        )
    return references


def _research_reference_text(raw: str, *, root_dir: Path = ROOT_DIR) -> str:
    if not raw.startswith("@"):
        return raw
    root = root_dir.resolve()
    raw_path = Path(raw[1:]).expanduser()
    path = raw_path if raw_path.is_absolute() else root / raw_path
    try:
        resolved = path.resolve()
        resolved.relative_to(root)
    except ValueError as exc:
        raise SystemExit(
            f"--research-reference file must be inside the workspace: {raw_path}"
        ) from exc
    try:
        return resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"--research-reference file is not readable: {resolved}: {exc}") from exc


def _research_selection_input_payload(
    raw_path: str | None,
    *,
    root_dir: Path = ROOT_DIR,
) -> dict[str, object]:
    if not _filled(raw_path):
        return {}
    root = root_dir.resolve()
    path = Path(str(raw_path)).expanduser()
    resolved = path if path.is_absolute() else root / path
    try:
        resolved = resolved.resolve()
        resolved.relative_to(root)
    except ValueError as exc:
        raise SystemExit(
            f"--research-selection-input-json must be inside the workspace: {raw_path}"
        ) from exc
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"--research-selection-input-json is not readable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("--research-selection-input-json must decode to a JSON object")
    nested = payload.get("research_selection_input_template")
    if isinstance(nested, dict):
        return nested
    return payload


def _required_scalar(
    cli_value: object,
    payload: dict[str, object],
    *keys: str,
    cli_name: str,
) -> str:
    value = _first_scalar(cli_value, payload, *keys)
    if not _filled(value):
        joined = " or ".join(keys)
        raise SystemExit(f"{cli_name} is required, or provide {joined} in input JSON")
    return str(value)


def _first_scalar(
    cli_value: object,
    payload: dict[str, object],
    *keys: str,
) -> str | None:
    if _filled(cli_value):
        return str(cli_value)
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (list, dict)):
            continue
        if _filled(value):
            return str(value)
    return None


def _required_list(
    cli_values: list[object],
    payload: dict[str, object],
    *keys: str,
    cli_name: str,
) -> list[str]:
    values = _merged_list(cli_values, payload, *keys)
    if not values:
        joined = " or ".join(keys)
        raise SystemExit(f"{cli_name} is required, or provide {joined} in input JSON")
    return values


def _merged_list(
    cli_values: list[object],
    payload: dict[str, object],
    *keys: str,
) -> list[str]:
    values: list[str] = []
    for key in keys:
        values.extend(_payload_list(payload, key))
    values.extend(str(value) for value in cli_values if _filled(value))
    return values


def _payload_list(payload: dict[str, object], *keys: str) -> list[object]:
    values: list[object] = []
    for key in keys:
        raw = payload.get(key)
        if raw is None:
            continue
        if isinstance(raw, list):
            values.extend(item for item in raw if _filled(item) or isinstance(item, dict))
        elif _filled(raw) or isinstance(raw, dict):
            values.append(raw)
    return values


def _response_values(
    cli_values: list[str],
    payload: dict[str, object],
    key: str,
    *,
    template_key: str,
) -> list[str]:
    values: list[str] = []
    values.extend(_payload_response_values(payload.get(key)))
    values.extend(_payload_response_values(payload.get(template_key)))
    values.extend(str(value) for value in cli_values if _filled(value))
    return values


def _payload_response_values(raw: object) -> list[str]:
    if isinstance(raw, dict):
        rows = []
        for key, value in raw.items():
            if not _filled(value):
                continue
            text = str(value).strip()
            rows.append(text if _has_response_separator(text) else f"{key}={text}")
        return rows
    if isinstance(raw, list):
        rows: list[str] = []
        for item in raw:
            if isinstance(item, str) and _filled(item):
                rows.append(item.strip())
                continue
            if not isinstance(item, dict):
                continue
            key = (
                item.get("response_key")
                or item.get("category")
                or item.get("index")
                or item.get("question")
            )
            response = (
                item.get("response")
                or item.get("response_template")
                or item.get("answer")
                or item.get("rationale")
            )
            if _filled(key) and _filled(response):
                text = str(response).strip()
                rows.append(text if _has_response_separator(text) else f"{key}={text}")
        return rows
    if _filled(raw):
        return [str(raw).strip()]
    return []


def _has_response_separator(text: str) -> bool:
    return "=" in text or ":" in text


def _research_reference_payloads(
    values: list[object],
    *,
    root_dir: Path = ROOT_DIR,
) -> list[StrategyProposalResearchReference]:
    references: list[StrategyProposalResearchReference] = []
    for item in values:
        if isinstance(item, str):
            references.extend(_research_reference_inputs([item], root_dir=root_dir))
            continue
        if not isinstance(item, dict):
            continue
        if not any(_filled(item.get(key)) for key in ("reference_id", "title", "source")):
            continue
        motivated = item.get("motivated_thesis_ids", [])
        if isinstance(motivated, str):
            motivated = [motivated]
        references.append(
            StrategyProposalResearchReference(
                reference_id=str(item.get("reference_id", "")),
                title=str(item.get("title", "")),
                source=str(item.get("source", "")),
                published_at=item.get("published_at"),
                relevance=str(item.get("relevance", "")),
                motivated_thesis_ids=[str(value) for value in motivated if _filled(value)],
            )
        )
    return references


def _filled(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    return not (text.startswith("<") and text.endswith(">"))


if __name__ == "__main__":
    raise SystemExit(main())
