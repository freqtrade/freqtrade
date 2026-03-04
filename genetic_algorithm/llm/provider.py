"""
LLM Provider Interface

Abstract base class and factory for LLM providers.
Supports Grok (xAI), OpenAI, and any OpenAI-compatible API.
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Environment variable names for API keys per provider
_ENV_KEY_MAP = {
    'grok': 'XAI_API_KEY',
    'xai': 'XAI_API_KEY',
    'groq': 'GROQ_API_KEY',
    'openai': 'OPENAI_API_KEY',
    'anthropic': 'ANTHROPIC_API_KEY',
    'claude': 'ANTHROPIC_API_KEY',
}


def _load_dotenv():
    """Load .env file from project root if it exists (no external dependency)."""
    # Walk up from this file to find .env in project root
    for parent in Path(__file__).resolve().parents:
        env_file = parent / '.env'
        if env_file.exists():
            try:
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#') or '=' not in line:
                            continue
                        key, _, value = line.partition('=')
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and value and key not in os.environ:
                            os.environ[key] = value
                logger.debug(f"Loaded .env from {env_file}")
            except Exception as e:
                logger.debug(f"Failed to load .env: {e}")
            return


# Load .env on module import
_load_dotenv()


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = config.get('model', '')
        self.base_url = config.get('base_url', '')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 4096)
        self.timeout = config.get('timeout', 60)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 2.0)
        
        # API key: config value → env var → empty
        self.api_key = config.get('api_key', '') or ''
        if not self.api_key:
            provider = config.get('provider', '').lower()
            env_var = _ENV_KEY_MAP.get(provider, f'{provider.upper()}_API_KEY')
            self.api_key = os.environ.get(env_var, '')
            if self.api_key:
                logger.info(f"API key loaded from ${env_var} environment variable")
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Generate a completion from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Generated text response
        """
        pass
    
    def generate_json(self, prompt: str, system_prompt: str = "") -> Optional[Dict]:
        """
        Generate a JSON response from the LLM.
        
        Retries on JSON parse failure only.  Network/HTTP retries are
        handled internally by ``generate()``, so we do **not** wrap
        ``generate()`` in a second retry loop to avoid max_retries²
        total attempts.
        
        Args:
            prompt: User prompt requesting JSON output
            system_prompt: Optional system prompt
            
        Returns:
            Parsed JSON dict, or None on failure
        """
        for attempt in range(self.max_retries):
            try:
                response = self.generate(prompt, system_prompt)
                json_str = self._extract_json(response)
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse failed (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
            except Exception as e:
                # generate() already retried network errors; if it still raised,
                # the endpoint is down — don't retry again here.
                logger.error(f"LLM generation failed: {e}")
                return None
        
        return None
    
    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from a response that may contain markdown fences."""
        text = text.strip()
        
        # Try to find JSON in markdown code blocks
        for fence in ['```json', '```']:
            if fence in text:
                start = text.index(fence) + len(fence)
                end = text.index('```', start) if '```' in text[start:] else len(text)
                return text[start:end].strip()
        
        # Try to find raw JSON object/array
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            first = text.find(start_char)
            last = text.rfind(end_char)
            if first != -1 and last != -1 and last > first:
                return text[first:last + 1]
        
        return text
    
    @property
    def provider_name(self) -> str:
        """Return the provider name for logging."""
        return self.__class__.__name__


class OpenAICompatibleProvider(LLMProvider):
    """
    Provider for OpenAI-compatible APIs (OpenAI, Grok/xAI, local servers, etc.)
    
    Works with any API that follows the OpenAI chat completions format.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not self.api_key:
            raise ValueError(f"API key required for {self.provider_name}")
    
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate using OpenAI-compatible chat completions API."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required for LLM providers. Install with: pip install httpx")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                    )
                    response.raise_for_status()
                    data = response.json()
                    return data['choices'][0]['message']['content']
            except httpx.HTTPStatusError as e:
                logger.warning(f"HTTP {e.response.status_code} from {self.provider_name} "
                             f"(attempt {attempt+1}/{self.max_retries})")
                if e.response.status_code == 429:
                    # Rate limit — exponential backoff
                    time.sleep(self.retry_delay * (2 ** attempt))
                elif e.response.status_code >= 500:
                    time.sleep(self.retry_delay)
                else:
                    raise
            except httpx.TimeoutException:
                logger.warning(f"Timeout from {self.provider_name} "
                             f"(attempt {attempt+1}/{self.max_retries})")
                time.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"Unexpected error from {self.provider_name}: {e}")
                raise
        
        raise RuntimeError(f"Failed after {self.max_retries} retries with {self.provider_name}")


class GrokProvider(OpenAICompatibleProvider):
    """xAI Grok API provider."""
    
    def __init__(self, config: Dict[str, Any]):
        if not config.get('base_url'):
            config['base_url'] = 'https://api.x.ai/v1'
        if not config.get('model'):
            config['model'] = 'grok-3-mini'
        super().__init__(config)


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI API provider."""
    
    def __init__(self, config: Dict[str, Any]):
        if not config.get('base_url'):
            config['base_url'] = 'https://api.openai.com/v1'
        if not config.get('model'):
            config['model'] = 'gpt-4o-mini'
        super().__init__(config)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not self.api_key:
            raise ValueError("API key required for Anthropic")
        if not self.base_url:
            self.base_url = 'https://api.anthropic.com/v1'
        if not self.model:
            self.model = 'claude-sonnet-4-20250514'
    
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate using Anthropic Messages API."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required for LLM providers. Install with: pip install httpx")
        
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        f"{self.base_url}/messages",
                        json=payload,
                        headers=headers,
                    )
                    response.raise_for_status()
                    data = response.json()
                    return data['content'][0]['text']
            except httpx.HTTPStatusError as e:
                logger.warning(f"HTTP {e.response.status_code} from Anthropic "
                             f"(attempt {attempt+1}/{self.max_retries})")
                if e.response.status_code == 429:
                    time.sleep(self.retry_delay * (2 ** attempt))
                elif e.response.status_code >= 500:
                    time.sleep(self.retry_delay)
                else:
                    raise
            except httpx.TimeoutException:
                logger.warning(f"Timeout from Anthropic (attempt {attempt+1}/{self.max_retries})")
                time.sleep(self.retry_delay)
        
        raise RuntimeError(f"Failed after {self.max_retries} retries with Anthropic")


class GroqProvider(OpenAICompatibleProvider):
    """Groq (groq.com) LPU inference provider — fast, OpenAI-compatible."""
    
    def __init__(self, config: Dict[str, Any]):
        if not config.get('base_url'):
            config['base_url'] = 'https://api.groq.com/openai/v1'
        if not config.get('model'):
            config['model'] = 'llama-3.3-70b-versatile'
        super().__init__(config)


class LocalProvider(OpenAICompatibleProvider):
    """Local LLM server (Ollama, llama.cpp, vLLM, etc.)"""
    
    def __init__(self, config: Dict[str, Any]):
        if not config.get('base_url'):
            config['base_url'] = 'http://localhost:11434/v1'
        if not config.get('model'):
            config['model'] = 'llama3'
        # Local servers often don't need an API key
        if not config.get('api_key'):
            config['api_key'] = 'not-needed'
        super().__init__(config)


# ============================================================================
# Provider Factory
# ============================================================================

PROVIDER_REGISTRY: Dict[str, type] = {
    'grok': GrokProvider,
    'xai': GrokProvider,
    'groq': GroqProvider,
    'openai': OpenAIProvider,
    'anthropic': AnthropicProvider,
    'claude': AnthropicProvider,
    'local': LocalProvider,
    'ollama': LocalProvider,
}


class LLMProviderFactory:
    """Factory for creating LLM providers from config."""
    
    @staticmethod
    def create(config: Dict[str, Any]) -> LLMProvider:
        """
        Create an LLM provider from configuration.
        
        Config should contain:
            provider: str - Provider name (grok, openai, anthropic, local)
            api_key: str - API key
            model: str - Model name (optional, uses provider default)
            base_url: str - API base URL (optional, uses provider default)
            temperature: float - Sampling temperature (default: 0.7)
            max_tokens: int - Max response tokens (default: 4096)
            
        Args:
            config: LLM configuration dict
            
        Returns:
            Configured LLMProvider instance
        """
        provider_name = config.get('provider', '').lower()
        
        if provider_name not in PROVIDER_REGISTRY:
            available = ', '.join(sorted(PROVIDER_REGISTRY.keys()))
            raise ValueError(f"Unknown LLM provider '{provider_name}'. "
                           f"Available: {available}")
        
        provider_cls = PROVIDER_REGISTRY[provider_name]
        provider = provider_cls(config)
        logger.info(f"Initialized LLM provider: {provider.provider_name} "
                   f"(model={provider.model})")
        return provider
    
    @staticmethod
    def register(name: str, provider_cls: type):
        """Register a custom provider class."""
        PROVIDER_REGISTRY[name.lower()] = provider_cls
