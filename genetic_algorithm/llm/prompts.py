"""
LLM Prompt Templates for Strategy Generation

Constructs structured prompts that guide LLMs to produce valid StrategyGene-
compatible JSON output. Prompts encode the full grammar of indicators,
conditions, operators, and risk parameters.
"""

import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Operator descriptions for the LLM
CONDITION_OPERATORS = {
    '>': 'Value is above threshold',
    '<': 'Value is below threshold',
    'cross_above': 'Value crosses above threshold (was below, now above)',
    'cross_below': 'Value crosses below threshold (was above, now below)',
    'increasing': 'Value has been increasing over lookback periods',
    'decreasing': 'Value has been decreasing over lookback periods',
    'between': 'Value is between threshold (lower) and threshold_upper',
    'value_above_ago': 'Current value is above value from lookback periods ago',
}

# Indicator descriptions with valid parameter ranges and typical condition usage
INDICATOR_REFERENCE = {
    'RSI': {
        'description': 'Relative Strength Index - momentum oscillator (0-100)',
        'parameters': {'period': '7-21'},
        'typical_entry': {'operator': '<', 'threshold': '20-40 (oversold)'},
        'typical_exit': {'operator': '>', 'threshold': '60-80 (overbought)'},
    },
    'MACD': {
        'description': 'Moving Average Convergence/Divergence - trend/momentum',
        'parameters': {'fast_period': '8-21', 'slow_period': '21-50', 'signal_period': '5-14'},
        'typical_entry': {'operator': 'cross_above', 'threshold': '0 (signal line cross)'},
        'typical_exit': {'operator': 'cross_below', 'threshold': '0'},
    },
    'BBANDS': {
        'description': 'Bollinger Bands - volatility bands around SMA',
        'parameters': {'period': '15-30', 'std_dev': '1.5-3.0'},
        'typical_entry': {'operator': '<', 'threshold': '-1.0 to 0 (near lower band)'},
        'typical_exit': {'operator': '>', 'threshold': '0.5 to 1.0 (near upper band)'},
    },
    'EMA': {
        'description': 'Exponential Moving Average - trend direction',
        'parameters': {'period': '5-50'},
        'typical_entry': {'operator': 'cross_above', 'threshold': '0 (price crosses above EMA)'},
        'typical_exit': {'operator': 'cross_below', 'threshold': '0'},
    },
    'SMA': {
        'description': 'Simple Moving Average - trend direction',
        'parameters': {'period': '10-100'},
        'typical_entry': {'operator': 'cross_above', 'threshold': '0'},
        'typical_exit': {'operator': 'cross_below', 'threshold': '0'},
    },
    'ADX': {
        'description': 'Average Directional Index - trend strength (0-100)',
        'parameters': {'period': '10-20'},
        'typical_entry': {'operator': '>', 'threshold': '20-40 (trending)'},
        'typical_exit': {'operator': '<', 'threshold': '20 (trend weakening)'},
    },
    'SUPERTREND': {
        'description': 'Supertrend - trend following with ATR-based bands',
        'parameters': {'period': '7-14', 'multiplier': '2.0-4.0'},
        'typical_entry': {'operator': '>', 'threshold': '0 (bullish)'},
        'typical_exit': {'operator': '<', 'threshold': '0 (bearish)'},
    },
    'ICHIMOKU': {
        'description': 'Ichimoku Cloud - multi-signal trend system',
        'parameters': {'tenkan_period': '7-12', 'kijun_period': '20-30', 'senkou_b_period': '40-60'},
        'typical_entry': {'operator': '>', 'threshold': '0 (above cloud)'},
        'typical_exit': {'operator': '<', 'threshold': '0 (below cloud)'},
    },
    'DONCHIAN': {
        'description': 'Donchian Channel - breakout trading',
        'parameters': {'period': '10-30'},
        'typical_entry': {'operator': '>', 'threshold': '0.8 (near upper channel)'},
        'typical_exit': {'operator': '<', 'threshold': '0.2 (near lower channel)'},
    },
    'PSAR': {
        'description': 'Parabolic SAR - trend reversal detection',
        'parameters': {'acceleration': '0.01-0.05', 'maximum': '0.1-0.3'},
        'typical_entry': {'operator': '>', 'threshold': '0 (bullish)'},
        'typical_exit': {'operator': '<', 'threshold': '0 (bearish)'},
    },
    'CMF': {
        'description': 'Chaikin Money Flow - volume-weighted accumulation/distribution',
        'parameters': {'period': '10-30'},
        'typical_entry': {'operator': '>', 'threshold': '0.05 (buying pressure)'},
        'typical_exit': {'operator': '<', 'threshold': '-0.05 (selling pressure)'},
    },
    'VROC': {
        'description': 'Volume Rate of Change - volume momentum',
        'parameters': {'period': '5-20'},
        'typical_entry': {'operator': '>', 'threshold': '50 (volume surge)'},
        'typical_exit': {'operator': '<', 'threshold': '-20 (volume decline)'},
    },
    'CDL_ENGULFING': {
        'description': 'Engulfing candlestick pattern',
        'parameters': {},
        'typical_entry': {'operator': '>', 'threshold': '0 (bullish engulfing)'},
        'typical_exit': {'operator': '<', 'threshold': '0 (bearish engulfing)'},
    },
    'CDL_HAMMER': {
        'description': 'Hammer candlestick pattern (reversal)',
        'parameters': {},
        'typical_entry': {'operator': '>', 'threshold': '0 (hammer detected)'},
        'typical_exit': {'operator': '<', 'threshold': '0'},
    },
    'CDL_MORNINGSTAR': {
        'description': 'Morning Star pattern (bullish reversal)',
        'parameters': {},
        'typical_entry': {'operator': '>', 'threshold': '0 (pattern detected)'},
        'typical_exit': {'operator': '<', 'threshold': '0'},
    },
    'CDL_EVENINGSTAR': {
        'description': 'Evening Star pattern (bearish reversal)',
        'parameters': {},
        'typical_entry': {'operator': '<', 'threshold': '0'},
        'typical_exit': {'operator': '>', 'threshold': '0 (pattern detected)'},
    },
    'CDL_DOJI': {
        'description': 'Doji candlestick pattern (indecision)',
        'parameters': {},
        'typical_entry': {'operator': '>', 'threshold': '0'},
        'typical_exit': {'operator': '>', 'threshold': '0'},
    },
}


class StrategyPromptBuilder:
    """
    Builds structured prompts for LLM-based strategy generation.
    
    Creates prompts that:
    1. Describe available indicators, conditions, and risk parameters
    2. Request valid StrategyGene-compatible JSON output
    3. Encode trading domain knowledge (trend following, mean reversion, etc.)
    4. Can optionally include context from previous evolution results
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.indicator_config = config.get('indicators', {})
        self.strategy_constraints = config.get('strategy_constraints', {})
        self.available_indicators = self.indicator_config.get('available', list(INDICATOR_REFERENCE.keys()))
        self.available_timeframes = self.strategy_constraints.get('timeframes', ['5m', '15m', '1h'])
        self.llm_config = config.get('advanced', {}).get('llm', {})
        
        # Build indicator reference filtered to available indicators
        self.indicator_ref = {
            k: v for k, v in INDICATOR_REFERENCE.items()
            if k in self.available_indicators
        }
    
    def build_system_prompt(self) -> str:
        """Build the system prompt that sets the LLM's role and constraints."""
        return """You are an expert quantitative trading strategy designer. You create trading strategies for cryptocurrency markets by selecting technical indicators and combining them with entry/exit conditions.

Your strategies MUST:
1. Use multiple uncorrelated indicators for robust signals (at least 2 entry conditions)
2. Include both entry AND exit conditions (at least 1 exit condition)
3. Use proper risk management (stop-loss, ROI targets)
4. Be designed to generalize across market conditions, NOT overfit to specific patterns
5. Output ONLY valid JSON matching the exact schema provided

Key principles:
- Combine trend filters (ADX, SUPERTREND, EMA) with oscillators (RSI, MACD, STOCH)
- Add volume confirmation (CMF, VROC) when possible
- Use conservative thresholds that work across different volatility regimes
- Prefer cross_above/cross_below for entry timing over static thresholds
- Set stop-loss between -5% and -15%, not too tight (whipsaws) or too loose (large draws)"""
    
    def build_seed_prompt(self, strategy_style: Optional[str] = None,
                          market_context: Optional[str] = None,
                          num_strategies: int = 1) -> str:
        """
        Build a prompt for generating seed strategies for initial population.
        
        Args:
            strategy_style: Optional hint like 'trend_following', 'mean_reversion', 'breakout'
            market_context: Optional market description for context
            num_strategies: Number of strategies to generate
            
        Returns:
            Formatted prompt string
        """
        # Build indicator reference section
        indicator_docs = self._format_indicator_reference()
        
        # Build schema section
        schema = self._format_output_schema()
        
        # Build constraints
        constraints = self._format_constraints()
        
        # Style guidance
        style_guidance = ""
        if strategy_style:
            style_guidance = f"\n\nSTRATEGY STYLE HINT: Design a {strategy_style} strategy. "
            style_hints = {
                'trend_following': "Use trend indicators (ADX, SUPERTREND, EMA) as primary signals. "
                    "Enter on confirmed trends, exit when trend weakens. "
                    "Use wider stops to ride trends.",
                'mean_reversion': "Use oscillators (RSI, BBANDS, STOCH) to detect oversold/overbought. "
                    "Enter when price deviates from mean, exit on return. "
                    "Use tighter stops as mean reversion strategies have shorter holding.",
                'breakout': "Use channel indicators (DONCHIAN, BBANDS) and volume (CMF, VROC). "
                    "Enter on breakout with volume confirmation, exit on failure. "
                    "Set stops just below breakout level.",
                'momentum': "Use momentum indicators (MACD, RSI, VROC). "
                    "Enter on increasing momentum, exit on momentum divergence. "
                    "Use trailing stops to capture extended moves.",
                'volatility': "Use ATR-based indicators (SUPERTREND, BBANDS). "
                    "Enter during volatility expansion, exit on contraction. "
                    "Adapt position sizing to volatility.",
            }
            style_guidance += style_hints.get(strategy_style, "")
        
        market_guidance = ""
        if market_context:
            market_guidance = f"\n\nMARKET CONTEXT: {market_context}"
        
        count_instruction = ""
        if num_strategies > 1:
            count_instruction = (
                f"\n\nGenerate exactly {num_strategies} DIFFERENT strategies as a JSON array. "
                "Each strategy should use a distinct combination of indicators and logic."
            )
        else:
            count_instruction = "\n\nGenerate exactly 1 strategy as a JSON object."
        
        return f"""Design a cryptocurrency trading strategy using the available indicators below.

{indicator_docs}

AVAILABLE CONDITION OPERATORS:
{self._format_operators()}

{constraints}

{schema}
{style_guidance}{market_guidance}{count_instruction}

IMPORTANT: 
- Every indicator referenced in conditions MUST be in the indicators list
- Use instance_id (e.g., "RSI_0") to reference indicators in conditions
- Return ONLY valid JSON, no explanations or comments"""

    def build_immigrant_prompt(self, 
                               top_performers: Optional[List[Dict]] = None,
                               weaknesses: Optional[List[str]] = None,
                               feedback: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a prompt for generating immigrant strategies during evolution.
        
        Can include context from current evolution state to guide the LLM
        toward filling gaps in the population.
        
        Args:
            top_performers: Summary of current top strategies for context
            weaknesses: List of identified weaknesses to address
            feedback: Performance feedback, feature importance, and evolution progress
            
        Returns:
            Formatted prompt string
        """
        indicator_docs = self._format_indicator_reference()
        schema = self._format_output_schema()
        constraints = self._format_constraints()
        
        context = ""
        if top_performers:
            context += "\n\nCURRENT EVOLUTION CONTEXT - Top performing strategies:"
            for i, perf in enumerate(top_performers[:3], 1):
                context += (f"\n  Strategy {i}: fitness={perf.get('fitness', 'N/A'):.4f}, "
                          f"indicators={perf.get('indicators', [])}, "
                          f"profit={perf.get('profit', 'N/A'):.1f}%, "
                          f"drawdown={perf.get('max_drawdown', 'N/A'):.1f}%")
        
        weakness_guidance = ""
        if weaknesses:
            weakness_guidance = (
                "\n\nIDENTIFIED GAPS to address with your new strategy:\n"
                + "\n".join(f"  - {w}" for w in weaknesses)
            )
        
        # Build feedback section from LLM performance history
        feedback_section = ""
        if feedback:
            feedback_section = self._format_feedback_context(feedback)
        
        return f"""Design a NOVEL cryptocurrency trading strategy that would complement existing strategies in a genetic algorithm population.

{indicator_docs}

AVAILABLE CONDITION OPERATORS:
{self._format_operators()}

{constraints}

{schema}
{context}{weakness_guidance}{feedback_section}

Design a strategy that is DIFFERENT from the top performers above. 
Use a different combination of indicators and trading logic.
Focus on robustness and generalization over raw profit.

Generate exactly 1 strategy as a JSON object.
Return ONLY valid JSON, no explanations."""

    def _format_feedback_context(self, feedback: Dict[str, Any]) -> str:
        """
        Format performance feedback into prompt-friendly text.
        
        Includes:
        1. How previous LLM strategies performed (learn from successes/failures)
        2. Feature importance data (which indicators the GA selects for)
        3. Evolution progress (generation, fitness trend, diversity)
        """
        sections = []
        
        # --- Section 1: LLM strategy performance results ---
        llm_results = feedback.get('llm_strategy_results', [])
        if llm_results:
            lines = ["\n\nPERFORMANCE FEEDBACK - Your previous strategies' results:"]
            
            # Separate into successes and failures
            good = [r for r in llm_results if r.get('fitness', 0) > 0.3]
            bad = [r for r in llm_results if r.get('fitness', 0) <= 0.3]
            
            if good:
                lines.append("  SUCCESSFUL strategies (learn from these):")
                for r in good[:3]:
                    lines.append(
                        f"    Gen {r['generation']}: fitness={r['fitness']:.4f}, "
                        f"profit={r['profit']:.1f}%, drawdown={r['max_drawdown']:.1f}%, "
                        f"win_rate={r['win_rate']:.0f}%, trades={r['num_trades']}, "
                        f"indicators={r['indicators']}"
                    )
            
            if bad:
                lines.append("  FAILED strategies (avoid these patterns):")
                for r in bad[:3]:
                    reasons = []
                    if r.get('num_trades', 0) == 0:
                        reasons.append("no trades generated")
                    if r.get('max_drawdown', 0) > 0.20:
                        reasons.append("excessive drawdown")
                    if r.get('win_rate', 0) < 30:
                        reasons.append("low win rate")
                    if r.get('profit', 0) < -5:
                        reasons.append("large loss")
                    reason_str = f" (issues: {', '.join(reasons)})" if reasons else ""
                    lines.append(
                        f"    Gen {r['generation']}: fitness={r['fitness']:.4f}, "
                        f"indicators={r['indicators']}{reason_str}"
                    )
            
            # LLM vs Random comparison
            llm_vs = feedback.get('llm_vs_random', {})
            if llm_vs:
                avg_llm = llm_vs.get('avg_llm_fitness', 0)
                avg_rand = llm_vs.get('avg_random_fitness', 0)
                diff = avg_llm - avg_rand
                if diff > 0:
                    lines.append(f"  Your strategies are OUTPERFORMING random by {diff:.4f} fitness on average. Keep innovating!")
                elif diff < -0.05:
                    lines.append(f"  Your strategies are UNDERPERFORMING random by {abs(diff):.4f}. Try different approaches!")
                else:
                    lines.append(f"  Your strategies perform similarly to random. Be bolder with indicator combinations!")
            
            sections.append("\n".join(lines))
        
        # --- Section 2: Feature importance (what the GA selects for) ---
        feature_data = feedback.get('feature_importance', [])
        if feature_data:
            lines = ["\n\nFEATURE IMPORTANCE - Indicators the evolution process values most:"]
            for f in feature_data:
                score = f.get('importance_score', 0)
                marker = "***" if score > 0.3 else "**" if score > 0.1 else "*"
                lines.append(
                    f"  {marker} {f['indicator']}: importance={score:+.4f}, "
                    f"avg_fitness_when_present={f['avg_fitness']:.4f}"
                )
            
            # Top condition patterns
            patterns = feedback.get('top_condition_patterns', [])
            if patterns:
                lines.append("  Best condition patterns:")
                for p in patterns[:3]:
                    lines.append(f"    {p['pattern']} (score={p['score']:+.4f})")
            
            lines.append("  Use HIGH-importance indicators as building blocks. Avoid or recombine LOW-importance ones.")
            sections.append("\n".join(lines))
        
        # --- Section 3: Evolution progress context ---
        progress = feedback.get('evolution_progress', {})
        if progress:
            lines = ["\n\nEVOLUTION PROGRESS:"]
            gen = progress.get('generation', '?')
            total = progress.get('total_generations', '?')
            best = progress.get('best_fitness', 0)
            lines.append(f"  Generation: {gen}/{total}")
            lines.append(f"  Best fitness so far: {best:.4f}")
            
            if progress.get('plateau_generations', 0) > 3:
                lines.append(f"  WARNING: Evolution has plateaued for {progress['plateau_generations']} generations!")
                lines.append("  Try a radically different approach - unusual indicator combinations or unconventional thresholds.")
            
            diversity = progress.get('diversity', None)
            if diversity is not None:
                if diversity < 0.3:
                    lines.append(f"  Population diversity is LOW ({diversity:.2f}). Introduce novel, diverse strategies!")
                elif diversity > 0.7:
                    lines.append(f"  Population diversity is HIGH ({diversity:.2f}). Focus on quality over novelty.")
            
            sections.append("\n".join(lines))
        
        return "".join(sections)
    
    def _format_indicator_reference(self) -> str:
        """Format the indicator reference for inclusion in prompts.""" 
        lines = ["AVAILABLE INDICATORS:"]
        for name, info in self.indicator_ref.items():
            params = info.get('parameters', {})
            param_str = ", ".join(f"{k}: {v}" for k, v in params.items()) if params else "none"
            lines.append(f"  {name}: {info['description']}")
            lines.append(f"    Parameters: {param_str}")
            lines.append(f"    Entry example: {info['typical_entry']}")
            lines.append(f"    Exit example: {info['typical_exit']}")
        return "\n".join(lines)
    
    def _format_operators(self) -> str:
        """Format available condition operators."""
        return "\n".join(f"  '{op}': {desc}" for op, desc in CONDITION_OPERATORS.items())
    
    def _format_constraints(self) -> str:
        """Format strategy constraints from config."""
        min_entry = self.indicator_config.get('min_entry_conditions', 2)
        max_entry = self.indicator_config.get('max_entry_conditions', 4)
        min_exit = self.indicator_config.get('min_exit_conditions', 1)
        min_ind = self.indicator_config.get('min_per_strategy', 2)
        max_ind = self.indicator_config.get('max_per_strategy', 5)
        sl_range = self.strategy_constraints.get('stoploss_range', [-0.20, -0.05])
        roi_range = self.strategy_constraints.get('roi_range', [0.01, 0.10])
        tfs = self.available_timeframes
        
        return f"""CONSTRAINTS:
  - Indicators: {min_ind} to {max_ind} per strategy
  - Entry conditions: {min_entry} to {max_entry} (AND logic)
  - Exit conditions: at least {min_exit}
  - Stop-loss: {sl_range[0]} to {sl_range[1]} (negative fraction)
  - ROI targets: {roi_range[0]} to {roi_range[1]} (positive fraction)
  - Timeframes: {tfs}
  - max_open_trades: 1 to 10"""
    
    def _format_output_schema(self) -> str:
        """Format the expected JSON output schema."""
        return """OUTPUT JSON SCHEMA:
{
  "indicators": [
    {
      "type": "RSI",           // Must be from AVAILABLE INDICATORS
      "instance_id": "RSI_0",  // Unique: TYPE_N (e.g., RSI_0, EMA_0, EMA_1)
      "parameters": {"period": 14},
      "weight": 1.0,
      "timeframe": null        // null = base timeframe, or "1h"/"4h" for informative
    }
  ],
  "entry_conditions": [
    {
      "indicator": "RSI_0",    // Must match an instance_id from indicators
      "operator": "<",         // From AVAILABLE CONDITION OPERATORS
      "threshold": 30,
      "logic": "AND",          // Use "AND" for all conditions
      "threshold_upper": 0,    // Only used with "between" operator
      "lookback": 3            // Only used with "increasing"/"decreasing"/"value_above_ago"
    }
  ],
  "exit_conditions": [
    {
      "indicator": "RSI_0",
      "operator": ">",
      "threshold": 70,
      "logic": "AND",
      "threshold_upper": 0,
      "lookback": 3
    }
  ],
  "timeframe": "15m",         // From available timeframes
  "stoploss": -0.10,          // Negative fraction
  "minimal_roi": {
    "0": 0.05,                // Immediate ROI target
    "30": 0.03,               // After 30 minutes
    "60": 0.01                // After 60 minutes  
  },
  "max_open_trades": 3,
  "trailing_stop": false,
  "trailing_stop_positive": null,
  "trailing_stop_positive_offset": null
}"""


# Strategy style pool for diverse seed generation
STRATEGY_STYLES = [
    'trend_following',
    'mean_reversion',
    'breakout',
    'momentum',
    'volatility',
]


def get_diverse_styles(count: int) -> List[str]:
    """
    Get a diverse set of strategy styles for seed generation.
    
    Distributes evenly across styles, cycling if count > len(styles).
    """
    styles = []
    for i in range(count):
        styles.append(STRATEGY_STYLES[i % len(STRATEGY_STYLES)])
    return styles
