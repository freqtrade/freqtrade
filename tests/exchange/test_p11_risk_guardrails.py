import os
import pytest
from pathlib import Path
from user_data.risk_guardrails.guardrails import RiskGuardrails


def test_risk_guardrails_disabled():
    config = {"risk_guardrails": {"enabled": False}}
    guardrails = RiskGuardrails(config)
    blocked, reason = guardrails.should_block_entry({})
    assert not blocked
    assert reason == "risk_guardrails_disabled"


def test_risk_guardrails_kill_switch(tmp_path):
    kill_switch = tmp_path / "KILL_SWITCH"
    config = {"risk_guardrails": {"enabled": True, "kill_switch": {"file": str(kill_switch)}}}
    guardrails = RiskGuardrails(config)

    # Not present
    blocked, reason = guardrails.should_block_entry({})
    assert not blocked

    # Present
    kill_switch.write_text("ON")
    blocked, reason = guardrails.should_block_entry({})
    assert blocked
    assert reason == "kill_switch_on"


def test_risk_guardrails_green_day_lock():
    config = {"risk_guardrails": {"enabled": True, "green_day_profit_ratio": 0.01}}
    guardrails = RiskGuardrails(config)

    # Below
    blocked, reason = guardrails.should_block_entry({"daily_profit_ratio": 0.005})
    assert not blocked

    # Above
    blocked, reason = guardrails.should_block_entry({"daily_profit_ratio": 0.011})
    assert blocked
    assert "green_day_lock" in reason


def test_risk_guardrails_max_daily_loss():
    config = {"risk_guardrails": {"enabled": True, "max_daily_loss_ratio": 0.01}}
    guardrails = RiskGuardrails(config)

    # Safe
    blocked, reason = guardrails.should_block_entry({"daily_profit_ratio": -0.005})
    assert not blocked

    # Loss Limit Hit
    blocked, reason = guardrails.should_block_entry({"daily_profit_ratio": -0.011})
    assert blocked
    assert "max_daily_loss" in reason


def test_risk_guardrails_env_overrides(monkeypatch):
    config = {"risk_guardrails": {"enabled": True, "green_day_profit_ratio": 0.01}}
    guardrails = RiskGuardrails(config)

    # Normal (not blocked)
    blocked, reason = guardrails.should_block_entry({"daily_profit_ratio": 0.0})
    assert not blocked

    # Override PROFIT to block (triggering green_day_lock)
    # Note: 0.015 > 0.01
    monkeypatch.setenv("RISK_FORCE_DAILY_PROFIT_RATIO", "0.015")
    blocked, reason = guardrails.should_block_entry({"daily_profit_ratio": 0.0})
    assert blocked
    assert "0.0150" in reason

    # Override LOSS to block (triggering max_daily_loss)
    # Given config max_daily_loss_ratio is not set in this subtest, let's reset config
    config_loss = {"risk_guardrails": {"enabled": True, "max_daily_loss_ratio": 0.01}}
    guardrails_loss = RiskGuardrails(config_loss)
    monkeypatch.delenv("RISK_FORCE_DAILY_PROFIT_RATIO")
    monkeypatch.setenv("RISK_FORCE_DAILY_LOSS_RATIO", "-0.015")
    blocked, reason = guardrails_loss.should_block_entry({"daily_profit_ratio": 0.0})
    assert blocked
    assert "max_daily_loss" in reason
    assert "-0.0150" in reason


def test_risk_guardrails_max_open():
    config = {"risk_guardrails": {"enabled": True, "max_open_positions": 3}}
    guardrails = RiskGuardrails(config)

    blocked, reason = guardrails.should_block_entry({"open_trades_count": 2})
    assert not blocked

    blocked, reason = guardrails.should_block_entry({"open_trades_count": 3})
    assert blocked
    assert "max_open_positions" in reason
