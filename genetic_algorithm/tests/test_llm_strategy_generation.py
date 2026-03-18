#!/usr/bin/env python3
"""
LLM Strategy Generation End-to-End Test

Tests that each LLM provider can generate valid StrategyGene objects
through the full pipeline: prompt → LLM API → JSON → parse → StrategyGene.

Usage:
    cd /home/periklis/projects/trading/freqtradeForkGA
    source .venv/bin/activate
    python genetic_algorithm/tests/test_llm_strategy_generation.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Minimal GA config needed by StrategyDesigner and StrategyParser
MINI_CONFIG = {
    'indicators': {
        'available': [
            'RSI', 'MACD', 'BBANDS', 'EMA', 'SMA', 'ADX', 'STOCH', 'ATR', 'CCI',
            'SUPERTREND', 'ICHIMOKU', 'DONCHIAN', 'PSAR', 'CMF', 'VROC',
            'CDL_ENGULFING', 'CDL_HAMMER', 'CDL_MORNINGSTAR', 'CDL_EVENINGSTAR', 'CDL_DOJI',
        ],
        'max_per_strategy': 6,
        'min_per_strategy': 1,
        'min_entry_conditions': 2,
        'min_exit_conditions': 1,
    },
    'strategy_constraints': {
        'timeframes': ['5m', '15m', '1h'],
        'stoploss_range': [-0.15, -0.03],
        'roi_range': [0.01, 0.10],
        'max_open_trades_range': [1, 5],
    },
}

# Provider configurations to test
PROVIDERS_TO_TEST = [
    {
        'name': 'Groq (llama-3.1-8b-instant)',
        'env_key': 'GROQ_API_KEY',
        'llm_config': {
            'enabled': True,
            'provider': 'groq',
            'model': 'llama-3.1-8b-instant',
            'temperature': 0.7,
            'max_retries': 3,
            'timeout': 60,
            'retry_delay': 5.0,
            'min_call_interval': 5.0,
            'max_call_interval': 15.0,
            'seed_ratio': 1.0,
            'immigrant_ratio': 1.0,
            'max_calls_per_generation': 50,
            'max_calls_per_run': 500,
        },
    },
    {
        'name': 'Groq (llama-3.3-70b)',
        'env_key': 'GROQ_API_KEY',
        'llm_config': {
            'enabled': True,
            'provider': 'groq',
            'model': 'llama-3.3-70b-versatile',
            'temperature': 0.7,
            'max_retries': 3,
            'timeout': 60,
            'retry_delay': 5.0,
            'min_call_interval': 5.0,
            'max_call_interval': 15.0,
            'seed_ratio': 1.0,
            'immigrant_ratio': 1.0,
            'max_calls_per_generation': 50,
            'max_calls_per_run': 500,
        },
    },
    {
        'name': 'Anthropic (claude-3-haiku)',
        'env_key': 'ANTHROPIC_API_KEY',
        'llm_config': {
            'enabled': True,
            'provider': 'anthropic',
            'model': 'claude-3-haiku-20240307',
            'temperature': 0.7,
            'max_retries': 2,
            'timeout': 60,
            'min_call_interval': 1.0,
            'max_call_interval': 5.0,
            'seed_ratio': 1.0,
            'immigrant_ratio': 1.0,
            'max_calls_per_generation': 50,
            'max_calls_per_run': 500,
        },
    },
    {
        'name': 'OpenAI (gpt-4o-mini)',
        'env_key': 'OPENAI_API_KEY',
        'llm_config': {
            'enabled': True,
            'provider': 'openai',
            'model': 'gpt-4o-mini',
            'temperature': 0.7,
            'max_retries': 2,
            'timeout': 60,
            'min_call_interval': 1.0,
            'max_call_interval': 5.0,
            'seed_ratio': 1.0,
            'immigrant_ratio': 1.0,
            'max_calls_per_generation': 50,
            'max_calls_per_run': 500,
        },
    },
    {
        'name': 'OpenAI (gpt-4o)',
        'env_key': 'OPENAI_API_KEY',
        'llm_config': {
            'enabled': True,
            'provider': 'openai',
            'model': 'gpt-4o',
            'temperature': 0.7,
            'max_retries': 2,
            'timeout': 60,
            'min_call_interval': 1.0,
            'max_call_interval': 5.0,
            'seed_ratio': 1.0,
            'immigrant_ratio': 1.0,
            'max_calls_per_generation': 50,
            'max_calls_per_run': 500,
        },
    },
    {
        'name': 'OpenAI (gpt-3.5-turbo)',
        'env_key': 'OPENAI_API_KEY',
        'llm_config': {
            'enabled': True,
            'provider': 'openai',
            'model': 'gpt-3.5-turbo',
            'temperature': 0.7,
            'max_retries': 2,
            'timeout': 60,
            'min_call_interval': 1.0,
            'max_call_interval': 5.0,
            'seed_ratio': 1.0,
            'immigrant_ratio': 1.0,
            'max_calls_per_generation': 50,
            'max_calls_per_run': 500,
        },
    },
]


def _build_full_config(llm_config: dict) -> dict:
    """Build full GA config with LLM settings merged in."""
    config = {**MINI_CONFIG}
    config['advanced'] = {
        'enable_llm': True,
        'llm': llm_config,
    }
    return config


def test_raw_api_call(provider_config: dict) -> dict:
    """Test 1: Raw API connectivity — can we call the LLM at all?"""
    from genetic_algorithm.llm.provider import LLMProviderFactory

    result = {'test': 'raw_api_call', 'status': 'SKIP', 'error': None, 'time_ms': None}

    api_key = os.environ.get(provider_config['env_key'], '')
    if not api_key:
        result['error'] = f"No {provider_config['env_key']} set"
        return result

    llm_cfg = provider_config['llm_config']
    try:
        provider = LLMProviderFactory.create({
            'provider': llm_cfg['provider'],
            'model': llm_cfg['model'],
            'api_key': api_key,
            'temperature': 0.5,
            'max_tokens': 200,
            'timeout': 30,
            'max_retries': 1,
        })
        start = time.time()
        resp = provider.generate("Respond with: {\"test\": true}", "Return only JSON.")
        elapsed = (time.time() - start) * 1000
        result['status'] = 'OK'
        result['time_ms'] = round(elapsed, 1)
        result['response_preview'] = resp[:120]
    except Exception as e:
        result['status'] = 'FAIL'
        result['error'] = str(e)[:200]

    return result


def test_json_generation(provider_config: dict) -> dict:
    """Test 2: Can the LLM produce valid JSON via generate_json?"""
    from genetic_algorithm.llm.provider import LLMProviderFactory

    result = {'test': 'json_generation', 'status': 'SKIP', 'error': None, 'time_ms': None}

    api_key = os.environ.get(provider_config['env_key'], '')
    if not api_key:
        result['error'] = f"No {provider_config['env_key']} set"
        return result

    llm_cfg = provider_config['llm_config']
    try:
        provider = LLMProviderFactory.create({
            'provider': llm_cfg['provider'],
            'model': llm_cfg['model'],
            'api_key': api_key,
            'temperature': 0.5,
            'max_tokens': 1000,
            'timeout': 45,
            'max_retries': 2,
        })
        start = time.time()
        data = provider.generate_json(
            'Generate a JSON object with: {"name": "test", "value": 42, "items": ["a","b"]}',
            "Return only valid JSON. No explanations."
        )
        elapsed = (time.time() - start) * 1000
        if data and isinstance(data, dict):
            result['status'] = 'OK'
            result['time_ms'] = round(elapsed, 1)
            result['json_keys'] = list(data.keys())
        else:
            result['status'] = 'FAIL'
            result['error'] = f"generate_json returned: {type(data).__name__} = {data}"
    except Exception as e:
        result['status'] = 'FAIL'
        result['error'] = str(e)[:200]

    return result


def test_strategy_prompt_and_parse(provider_config: dict) -> dict:
    """Test 3: Full pipeline — prompt builder → LLM → parser → StrategyGene."""
    from genetic_algorithm.llm.provider import LLMProviderFactory
    from genetic_algorithm.llm.prompts import StrategyPromptBuilder
    from genetic_algorithm.llm.parser import StrategyParser

    result = {
        'test': 'full_strategy_generation',
        'status': 'SKIP',
        'error': None,
        'time_ms': None,
        'strategy_details': None,
        'parse_fixes': None,
    }

    api_key = os.environ.get(provider_config['env_key'], '')
    if not api_key:
        result['error'] = f"No {provider_config['env_key']} set"
        return result

    config = _build_full_config(provider_config['llm_config'])
    llm_cfg = provider_config['llm_config']

    try:
        provider = LLMProviderFactory.create({
            'provider': llm_cfg['provider'],
            'model': llm_cfg['model'],
            'api_key': api_key,
            'temperature': 0.7,
            'max_tokens': 4096,
            'timeout': 60,
            'max_retries': 2,
        })

        prompt_builder = StrategyPromptBuilder(config)
        parser = StrategyParser(config)

        system_prompt = prompt_builder.build_system_prompt()
        user_prompt = prompt_builder.build_seed_prompt(strategy_style='trend_following')

        start = time.time()
        raw_json = provider.generate_json(user_prompt, system_prompt)
        elapsed = (time.time() - start) * 1000

        if raw_json is None:
            result['status'] = 'FAIL'
            result['error'] = 'LLM returned no valid JSON (generate_json returned None)'
            result['time_ms'] = round(elapsed, 1)
            return result

        # Handle array response
        if isinstance(raw_json, list):
            raw_json = raw_json[0] if raw_json else None
            if raw_json is None:
                result['status'] = 'FAIL'
                result['error'] = 'LLM returned empty array'
                return result

        result['raw_json_keys'] = list(raw_json.keys()) if isinstance(raw_json, dict) else str(type(raw_json))

        # Parse into StrategyGene
        gene, error_msg = parser.parse_with_feedback(raw_json, generation=0, individual_id=0)

        if gene is None:
            result['status'] = 'FAIL'
            result['error'] = f'Parser rejected LLM output: {error_msg}'
            result['time_ms'] = round(elapsed, 1)
            result['raw_json_sample'] = json.dumps(raw_json, indent=2)[:500]
            return result

        result['status'] = 'OK'
        result['time_ms'] = round(elapsed, 1)
        result['strategy_details'] = {
            'indicators': [f"{ind.type}({ind.instance_id})" for ind in gene.indicators],
            'entry_conditions': len(gene.entry_conditions),
            'exit_conditions': len(gene.exit_conditions),
            'timeframe': gene.timeframe,
            'stoploss': gene.stoploss,
            'trailing_stop': gene.trailing_stop,
        }
    except Exception as e:
        result['status'] = 'FAIL'
        result['error'] = str(e)[:300]

    return result


def test_designer_seed(provider_config: dict) -> dict:
    """Test 4: StrategyDesigner.generate_seed_strategies() end-to-end."""
    from genetic_algorithm.llm.designer import StrategyDesigner

    result = {
        'test': 'designer_seed_generation',
        'status': 'SKIP',
        'error': None,
        'time_ms': None,
        'strategies_generated': 0,
        'strategies_requested': 2,
    }

    api_key = os.environ.get(provider_config['env_key'], '')
    if not api_key:
        result['error'] = f"No {provider_config['env_key']} set"
        return result

    config = _build_full_config(provider_config['llm_config'])

    try:
        start = time.time()
        designer = StrategyDesigner(config)
        strategies = designer.generate_seed_strategies(count=2, generation=0, start_id=0)
        elapsed = (time.time() - start) * 1000

        result['strategies_generated'] = len(strategies)
        result['time_ms'] = round(elapsed, 1)
        result['designer_stats'] = {
            'total_requests': designer.stats['total_requests'],
            'successful': designer.stats['successful'],
            'failed': designer.stats['failed'],
        }

        if len(strategies) >= 1:
            result['status'] = 'OK'
            result['strategy_summaries'] = []
            for s in strategies:
                result['strategy_summaries'].append({
                    'indicators': [ind.type for ind in s.indicators],
                    'entry_conds': len(s.entry_conditions),
                    'exit_conds': len(s.exit_conditions),
                    'timeframe': s.timeframe,
                })
        else:
            result['status'] = 'FAIL'
            result['error'] = f'No strategies generated (requested 2)'
    except Exception as e:
        result['status'] = 'FAIL'
        result['error'] = str(e)[:300]

    return result


def print_test_result(result: dict, indent: int = 2):
    """Pretty-print a test result."""
    prefix = ' ' * indent
    icons = {'OK': '✅', 'FAIL': '❌', 'SKIP': '⏭️', 'WARN': '⚠️'}
    icon = icons.get(result['status'], '❓')

    print(f"{prefix}{icon} {result['test']}: {result['status']}")
    if result.get('time_ms'):
        print(f"{prefix}   Time: {result['time_ms']}ms")
    if result.get('error'):
        print(f"{prefix}   Error: {result['error']}")
    if result.get('response_preview'):
        print(f"{prefix}   Response: {result['response_preview']}")
    if result.get('json_keys'):
        print(f"{prefix}   JSON keys: {result['json_keys']}")
    if result.get('raw_json_keys'):
        print(f"{prefix}   Raw JSON keys: {result['raw_json_keys']}")
    if result.get('strategy_details'):
        d = result['strategy_details']
        print(f"{prefix}   Indicators: {d['indicators']}")
        print(f"{prefix}   Entry conditions: {d['entry_conditions']}, Exit conditions: {d['exit_conditions']}")
        print(f"{prefix}   Timeframe: {d['timeframe']}, Stoploss: {d['stoploss']}")
    if result.get('strategies_generated') is not None and result.get('strategies_requested'):
        print(f"{prefix}   Generated: {result['strategies_generated']}/{result['strategies_requested']}")
    if result.get('strategy_summaries'):
        for i, s in enumerate(result['strategy_summaries']):
            print(f"{prefix}   Strategy {i}: {s['indicators']} | "
                  f"{s['entry_conds']} entry, {s['exit_conds']} exit | {s['timeframe']}")
    if result.get('designer_stats'):
        ds = result['designer_stats']
        print(f"{prefix}   Designer stats: {ds['successful']} ok, {ds['failed']} failed, "
              f"{ds['total_requests']} total")
    if result.get('raw_json_sample'):
        print(f"{prefix}   Raw JSON (truncated):")
        for line in result['raw_json_sample'].split('\n')[:10]:
            print(f"{prefix}     {line}")


def main():
    print("=" * 72)
    print("LLM STRATEGY GENERATION — END-TO-END TEST")
    print("=" * 72)
    print()

    # Load .env
    from genetic_algorithm.llm.provider import _load_dotenv
    _load_dotenv()

    # Environment check
    print("ENVIRONMENT")
    print("-" * 50)
    for p in PROVIDERS_TO_TEST:
        key = os.environ.get(p['env_key'], '')
        status = '✅ SET' if key else '❌ NOT SET'
        masked = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else '(empty)'
        print(f"  {p['env_key']}: {status} {masked if key else ''}")
    print()

    all_results = {}

    for provider_cfg in PROVIDERS_TO_TEST:
        name = provider_cfg['name']
        print("=" * 72)
        print(f"TESTING: {name}")
        print("=" * 72)

        api_key = os.environ.get(provider_cfg['env_key'], '')
        if not api_key:
            print(f"  ⏭️  Skipped — no {provider_cfg['env_key']} configured")
            all_results[name] = 'SKIP'
            print()
            continue

        tests = [
            test_raw_api_call,
            test_json_generation,
            test_strategy_prompt_and_parse,
            test_designer_seed,
        ]

        provider_ok = True
        results_for_provider = []

        for test_fn in tests:
            result = test_fn(provider_cfg)
            results_for_provider.append(result)
            print_test_result(result)
            print()

            # Stop testing this provider on hard failure
            if result['status'] == 'FAIL':
                provider_ok = False
                # Still continue — we want to see which tests pass
                # but for API failures, skip the rest
                if result['test'] == 'raw_api_call':
                    print(f"    ⚠️  Skipping remaining tests (API not reachable)")
                    break

            # Longer delays for Groq to avoid token rate limits
            delay = 8 if 'groq' in provider_cfg['llm_config'].get('provider', '') else 2
            time.sleep(delay)

        all_results[name] = 'OK' if provider_ok else 'FAIL'

    # Summary
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for name, status in all_results.items():
        icon = {'OK': '✅', 'FAIL': '❌', 'SKIP': '⏭️'}.get(status, '❓')
        print(f"  {icon} {name}: {status}")

    ok_count = sum(1 for s in all_results.values() if s == 'OK')
    fail_count = sum(1 for s in all_results.values() if s == 'FAIL')
    skip_count = sum(1 for s in all_results.values() if s == 'SKIP')

    print()
    print(f"  Total: {ok_count} OK, {fail_count} FAIL, {skip_count} SKIP")

    if ok_count > 0:
        print()
        print("🎉 At least one provider can generate valid trading strategies!")
    if fail_count > 0:
        print()
        print("⚠️  Some providers failed. Check errors above for details.")

    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
