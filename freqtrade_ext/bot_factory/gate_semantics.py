from __future__ import annotations

from typing import Any


GATE_GLOSSARY: dict[str, dict[str, str]] = {
    "initial_backtest_gate.pass": {
        "permits": "candidate may proceed to historical walk-forward review",
        "does_not_permit": "paper trading, dry-run trading, live trading, or exchange order placement",
        "next_required_gate": "walk_forward_gate.pass",
    },
    "eligible_for_walk_forward_review": {
        "permits": "candidate may be evaluated across predefined historical windows",
        "does_not_permit": "paper trading, dry-run trading, live trading, or exchange order placement",
        "next_required_gate": "walk_forward_gate.pass",
    },
    "REGIME_SCOPED_SELECTOR_ELIGIBLE": {
        "permits": "local selector simulation may consider this candidate only inside eligible regimes",
        "does_not_permit": "paper trading, dry-run trading, live trading, or process control",
        "next_required_gate": "paper_readiness.pass",
    },
    "GLOBAL_SELECTOR_ELIGIBLE": {
        "permits": "local selector simulation may consider this candidate across declared regimes",
        "does_not_permit": "paper trading, dry-run trading, live trading, or process control",
        "next_required_gate": "paper_readiness.pass",
    },
    "paper_readiness.pass": {
        "permits": "a human may separately request a later no-startup paper plan",
        "does_not_permit": "starting freqtrade trade, dry-run, paper, live, or canary processes",
        "next_required_gate": "explicit human startup request plus Phase 3 startup preflight",
    },
}


def gate_glossary() -> dict[str, dict[str, str]]:
    return {key: dict(value) for key, value in GATE_GLOSSARY.items()}


def gate_semantics(name: str) -> dict[str, str]:
    return dict(
        GATE_GLOSSARY.get(
            name,
            {
                "permits": "local artifact review only",
                "does_not_permit": "paper trading, dry-run trading, live trading, or exchange order placement",
                "next_required_gate": "documented downstream Bot Factory gate",
            },
        )
    )


def gate_semantics_payload(*names: str) -> dict[str, Any]:
    selected = names or tuple(GATE_GLOSSARY)
    return {
        "factory": "bot_factory_gate_semantics",
        "schema_version": "gate_semantics_v1",
        "promotion_authorized_by_this_command": False,
        "paper_live_approval_by_name_allowed": False,
        "gates": {name: gate_semantics(name) for name in selected},
    }


def render_gate_glossary_markdown() -> str:
    lines = [
        "# Bot Factory Gate Glossary",
        "",
        "Every gate in this glossary is a local artifact-review result. No gate name starts paper, dry-run, live, canary, or exchange-facing execution.",
        "",
        "| gate | permits | does not permit | next required gate |",
        "| --- | --- | --- | --- |",
    ]
    for name, semantics in sorted(GATE_GLOSSARY.items()):
        lines.append(
            "| {name} | {permits} | {does_not_permit} | {next_required_gate} |".format(
                name=name,
                permits=semantics["permits"],
                does_not_permit=semantics["does_not_permit"],
                next_required_gate=semantics["next_required_gate"],
            )
        )
    lines.append("")
    return "\n".join(lines)
