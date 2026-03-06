#!/usr/bin/env python3
"""
LLM Provider Connectivity Test

Tests each configured LLM provider independently and as a router.
Run this before starting long evolution runs to verify all providers work.

Usage:
    cd /home/kali/trading/freqtradeForkGA
    source .venv/bin/activate
    python genetic_algorithm/tests/test_llm_providers.py
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Provider test prompt  
TEST_PROMPT = """Generate a simple JSON trading strategy with these fields:
{
  "name": "test_strategy",
  "indicators": ["RSI"],
  "buy_condition": "RSI < 30",
  "sell_condition": "RSI > 70"
}
Return ONLY valid JSON, no explanation."""

PROVIDERS = [
    {
        'name': 'Groq',
        'provider': 'groq',
        'env_key': 'GROQ_API_KEY',
        'model': 'llama-3.3-70b-versatile',
    },
    {
        'name': 'OpenAI',
        'provider': 'openai',
        'env_key': 'OPENAI_API_KEY',
        'model': 'gpt-4o-mini',  # Cheapest: $0.15/1M input, $0.60/1M output
    },
    {
        'name': 'Anthropic',
        'provider': 'anthropic',
        'env_key': 'ANTHROPIC_API_KEY',
        'model': 'claude-3-haiku-20240307',  # Cheapest: $0.25/1M input, $1.25/1M output
    },
]


def test_single_provider(provider_config: dict) -> dict:
    """Test a single LLM provider.
    
    Returns:
        dict: Test result with status, response time, error message if any
    """
    from genetic_algorithm.llm.provider import LLMProviderFactory
    
    name = provider_config['name']
    env_key = provider_config['env_key']
    
    result = {
        'name': name,
        'status': 'unknown',
        'response_time_ms': None,
        'error': None,
        'api_key_present': False,
    }
    
    # Check API key
    api_key = os.environ.get(env_key, '')
    if not api_key:
        result['status'] = 'SKIPPED'
        result['error'] = f'No {env_key} environment variable set'
        return result
    
    result['api_key_present'] = True
    
    # Create provider config
    config = {
        'provider': provider_config['provider'],
        'model': provider_config['model'],
        'api_key': api_key,
        'temperature': 0.5,
        'max_tokens': 500,
        'timeout': 30,
        'max_retries': 1,
    }
    
    try:
        provider = LLMProviderFactory.create(config)
        
        start = time.time()
        response = provider.generate(
            TEST_PROMPT,
            system_prompt="You are a trading strategy generator. Return only valid JSON."
        )
        elapsed_ms = (time.time() - start) * 1000
        
        result['response_time_ms'] = round(elapsed_ms, 1)
        result['status'] = 'OK'
        
        # Quick validation - check if response contains JSON-like content
        if '{' in response and '}' in response:
            result['response_preview'] = response[:100] + '...' if len(response) > 100 else response
        else:
            result['status'] = 'WARN'
            result['error'] = 'Response does not appear to contain JSON'
            result['response_preview'] = response[:100]
            
    except Exception as e:
        result['status'] = 'FAIL'
        result['error'] = str(e)
    
    return result


def test_router(providers_list: list) -> dict:
    """Test the router with multiple providers.
    
    Returns:
        dict: Router test result
    """
    from genetic_algorithm.llm.router import LLMProviderRouter
    
    result = {
        'name': 'LLMProviderRouter',
        'status': 'unknown',
        'registered_providers': [],
        'skipped_providers': [],
        'response_time_ms': None,
        'error': None,
    }
    
    # Build router config
    config = {
        'temperature': 0.5,
        'max_tokens': 500,
        'timeout': 30,
        'max_retries': 1,
        'cooldown_seconds': 5,
        'providers_list': [],
    }
    
    # Add providers with API keys
    for p in providers_list:
        api_key = os.environ.get(p['env_key'], '')
        if api_key:
            config['providers_list'].append({
                'provider': p['provider'],
                'model': p['model'],
                'api_key': api_key,
            })
            result['registered_providers'].append(p['name'])
        else:
            result['skipped_providers'].append(f"{p['name']} (no {p['env_key']})")
    
    if not config['providers_list']:
        result['status'] = 'SKIPPED'
        result['error'] = 'No providers have API keys configured'
        return result
    
    try:
        router = LLMProviderRouter(config)
        
        start = time.time()
        response = router.generate(
            TEST_PROMPT,
            system_prompt="You are a trading strategy generator. Return only valid JSON."
        )
        elapsed_ms = (time.time() - start) * 1000
        
        result['response_time_ms'] = round(elapsed_ms, 1)
        result['status'] = 'OK'
        result['router_stats'] = router.get_router_stats()
        
        if '{' in response and '}' in response:
            result['response_preview'] = response[:100] + '...' if len(response) > 100 else response
        else:
            result['status'] = 'WARN'
            result['error'] = 'Response does not appear to contain JSON'
            
    except Exception as e:
        result['status'] = 'FAIL'
        result['error'] = str(e)
    
    return result


def test_failover() -> dict:
    """Test router failover by using a bad key for the first provider."""
    from genetic_algorithm.llm.router import LLMProviderRouter
    
    result = {
        'name': 'Failover Test',
        'status': 'unknown',
        'error': None,
    }
    
    # Find first working provider for fallback
    fallback_provider = None
    for p in PROVIDERS:
        if os.environ.get(p['env_key']):
            fallback_provider = p
            break
    
    if not fallback_provider:
        result['status'] = 'SKIPPED'
        result['error'] = 'No providers with API keys available'
        return result
    
    config = {
        'temperature': 0.5,
        'max_tokens': 500,
        'timeout': 10,
        'max_retries': 1,
        'cooldown_seconds': 1,
        'providers_list': [
            # Bad provider that should fail
            {
                'provider': 'groq',
                'api_key': 'bad-api-key-that-will-fail',
                'model': 'llama-3.3-70b-versatile',
            },
            # Good fallback provider
            {
                'provider': fallback_provider['provider'],
                'api_key': os.environ.get(fallback_provider['env_key']),
                'model': fallback_provider['model'],
            },
        ],
    }
    
    try:
        router = LLMProviderRouter(config)
        
        start = time.time()
        response = router.generate(
            "Say 'failover test successful' in JSON: {\"status\": \"...\"}",
            system_prompt="Return only valid JSON."
        )
        elapsed_ms = (time.time() - start) * 1000
        
        stats = router.get_router_stats()
        
        # Check if failover actually happened
        if stats['stats'].get('groq', {}).get('failures', 0) > 0:
            result['status'] = 'OK'
            result['message'] = f'First provider failed, fallback to {fallback_provider["name"]} succeeded'
            result['response_time_ms'] = round(elapsed_ms, 1)
        else:
            result['status'] = 'WARN'
            result['message'] = 'Expected first provider to fail but it succeeded'
            
    except Exception as e:
        result['status'] = 'FAIL'
        result['error'] = str(e)
    
    return result


def print_result(result: dict, indent: int = 0):
    """Pretty print a test result."""
    prefix = "  " * indent
    status_emoji = {
        'OK': '✅',
        'WARN': '⚠️',
        'FAIL': '❌',
        'SKIPPED': '⏭️',
        'unknown': '❓',
    }
    
    emoji = status_emoji.get(result['status'], '❓')
    print(f"{prefix}{emoji} {result['name']}: {result['status']}")
    
    if result.get('response_time_ms'):
        print(f"{prefix}   Response time: {result['response_time_ms']}ms")
    
    if result.get('error'):
        print(f"{prefix}   Error: {result['error']}")
    
    if result.get('message'):
        print(f"{prefix}   {result['message']}")
    
    if result.get('registered_providers'):
        print(f"{prefix}   Registered: {', '.join(result['registered_providers'])}")
    
    if result.get('skipped_providers'):
        print(f"{prefix}   Skipped: {', '.join(result['skipped_providers'])}")


def main():
    print("=" * 70)
    print("LLM PROVIDER CONNECTIVITY TEST")
    print("=" * 70)
    print()
    
    # Check environment
    print("1. ENVIRONMENT CHECK")
    print("-" * 40)
    for p in PROVIDERS:
        key = os.environ.get(p['env_key'], '')
        status = '✅ SET' if key else '❌ NOT SET'
        masked = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else '(empty)'
        print(f"   {p['env_key']}: {status} {masked if key else ''}")
    print()
    
    # Test individual providers
    print("2. INDIVIDUAL PROVIDER TESTS")
    print("-" * 40)
    provider_results = []
    for p in PROVIDERS:
        result = test_single_provider(p)
        provider_results.append(result)
        print_result(result, indent=1)
        print()
    
    # Test router
    print("3. ROUTER TEST (Multi-Provider)")
    print("-" * 40)
    router_result = test_router(PROVIDERS)
    print_result(router_result, indent=1)
    print()
    
    # Test failover
    print("4. FAILOVER TEST")
    print("-" * 40)
    failover_result = test_failover()
    print_result(failover_result, indent=1)
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    ok_count = sum(1 for r in provider_results if r['status'] == 'OK')
    fail_count = sum(1 for r in provider_results if r['status'] == 'FAIL')
    skip_count = sum(1 for r in provider_results if r['status'] == 'SKIPPED')
    
    print(f"   Individual Providers: {ok_count} OK, {fail_count} FAIL, {skip_count} SKIPPED")
    print(f"   Router: {router_result['status']}")
    print(f"   Failover: {failover_result['status']}")
    
    all_ok = (
        all(r['status'] in ('OK', 'SKIPPED') for r in provider_results)
        and router_result['status'] in ('OK', 'SKIPPED')
        and failover_result['status'] in ('OK', 'SKIPPED')
    )
    
    print()
    if all_ok:
        print("🎉 All tests passed! LLM providers are ready for evolution.")
    else:
        print("⚠️  Some tests failed. Review errors above before starting evolution.")
    
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
