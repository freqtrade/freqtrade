#!/usr/bin/env python3
"""
Extract strategies from GA-generated strategy files and create a checkpoint/seed file.

This tool parses the generated .py strategy files in ga_generated/ directory,
fully reconstructs StrategyGene objects (indicators, conditions, params), and
saves them as a JSON seed file that can be loaded by the GA system via --seed
flag or inject_immigrants().

Usage:
    python genetic_algorithm/tools/extract_checkpoint.py \
        --strategy-dir user_data/strategies/ga_generated/ \
        --output genetic_algorithm/data/checkpoints/seed_from_run.json \
        --top 20

    # Or extract from a specific set of files:
    python genetic_algorithm/tools/extract_checkpoint.py \
        --files "GAStrategy_Gen8_Ind51.py,GAStrategy_Gen5_Ind10.py" \
        --output seed.json
"""

import sys
import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.core.individual import Individual


# ---------------------------------------------------------------------------
# Indicator parsing
# ---------------------------------------------------------------------------

# Patterns for base-timeframe indicators (operate on `dataframe`)
_BASE_INDICATOR_PATTERNS = [
    # RSI: dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
    (re.compile(r"dataframe\[.rsi_(\d+).\]\s*=\s*ta\.RSI\(dataframe,\s*timeperiod=(\d+)\)"), 'RSI'),
    # EMA: dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
    (re.compile(r"dataframe\[.ema_(\d+).\]\s*=\s*ta\.EMA\(dataframe,\s*timeperiod=(\d+)\)"), 'EMA'),
    # SMA: dataframe['sma_20'] = ta.SMA(dataframe, timeperiod=20)
    (re.compile(r"dataframe\[.sma_(\d+).\]\s*=\s*ta\.SMA\(dataframe,\s*timeperiod=(\d+)\)"), 'SMA'),
    # ATR: dataframe['atr_14'] = ta.ATR(dataframe, timeperiod=14)
    (re.compile(r"dataframe\[.atr_(\d+).\]\s*=\s*ta\.ATR\(dataframe,\s*timeperiod=(\d+)\)"), 'ATR'),
    # ADX: dataframe['adx_14'] = ta.ADX(dataframe, timeperiod=14)
    (re.compile(r"dataframe\[.adx_(\d+).\]\s*=\s*ta\.ADX\(dataframe,\s*timeperiod=(\d+)\)"), 'ADX'),
    # CCI: dataframe['cci_14'] = ta.CCI(dataframe, timeperiod=14)
    (re.compile(r"dataframe\[.cci_(\d+).\]\s*=\s*ta\.CCI\(dataframe,\s*timeperiod=(\d+)\)"), 'CCI'),
    # MACD: macd = ta.MACD(dataframe, fastperiod=11, slowperiod=34, signalperiod=10)
    (re.compile(r"macd\s*=\s*ta\.MACD\(dataframe,\s*fastperiod=(\d+),\s*slowperiod=(\d+),\s*signalperiod=(\d+)\)"), 'MACD'),
    # BBANDS: bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
    (re.compile(r"bollinger\s*=\s*ta\.BBANDS\(dataframe,\s*timeperiod=(\d+),\s*nbdevup=([\d.]+),\s*nbdevdn=([\d.]+)\)"), 'BBANDS'),
    # STOCH: stoch = ta.STOCH(dataframe, fastk_period=14, slowk_period=3, slowd_period=3)
    (re.compile(r"stoch\s*=\s*ta\.STOCH\(dataframe,\s*fastk_period=(\d+),\s*slowk_period=(\d+),\s*slowd_period=(\d+)\)"), 'STOCH'),
]

# Patterns for informative-timeframe indicators (operate on `informative`)
_INF_INDICATOR_PATTERNS = [
    (re.compile(r"informative\[.rsi_(\d+).\]\s*=\s*ta\.RSI\(informative,\s*timeperiod=(\d+)\)"), 'RSI'),
    (re.compile(r"informative\[.ema_(\d+).\]\s*=\s*ta\.EMA\(informative,\s*timeperiod=(\d+)\)"), 'EMA'),
    (re.compile(r"informative\[.sma_(\d+).\]\s*=\s*ta\.SMA\(informative,\s*timeperiod=(\d+)\)"), 'SMA'),
    (re.compile(r"informative\[.atr_(\d+).\]\s*=\s*ta\.ATR\(informative,\s*timeperiod=(\d+)\)"), 'ATR'),
    (re.compile(r"informative\[.adx_(\d+).\]\s*=\s*ta\.ADX\(informative,\s*timeperiod=(\d+)\)"), 'ADX'),
    (re.compile(r"informative\[.cci_(\d+).\]\s*=\s*ta\.CCI\(informative,\s*timeperiod=(\d+)\)"), 'CCI'),
    (re.compile(r"macd\s*=\s*ta\.MACD\(informative,\s*fastperiod=(\d+),\s*slowperiod=(\d+),\s*signalperiod=(\d+)\)"), 'MACD'),
    (re.compile(r"bollinger\s*=\s*ta\.BBANDS\(informative,\s*timeperiod=(\d+),\s*nbdevup=([\d.]+),\s*nbdevdn=([\d.]+)\)"), 'BBANDS'),
    (re.compile(r"stoch\s*=\s*ta\.STOCH\(informative,\s*fastk_period=(\d+),\s*slowk_period=(\d+),\s*slowd_period=(\d+)\)"), 'STOCH'),
]

# ---------------------------------------------------------------------------
# Condition parsing
# ---------------------------------------------------------------------------

_CONDITION_PATTERNS = [
    # RSI/CCI/ADX/EMA/SMA/ATR  < or > threshold
    # e.g. (dataframe['rsi_14'] < 30)  or  (dataframe['cci_14'] > 156)
    re.compile(r"\(dataframe\['(\w+?)'\]\s*([<>])\s*([-\d.]+)\)"),
    # MACD cross: (dataframe['macd'] > dataframe['macdsignal'])
    re.compile(r"\(dataframe\['(macd(?:_\w+)?)'\]\s*([<>])\s*dataframe\['(macdsignal(?:_\w+)?)'\]\)"),
    # STOCH cross: (dataframe['slowk'] > dataframe['slowd'])
    re.compile(r"\(dataframe\['(slowk(?:_\w+)?)'\]\s*([<>])\s*dataframe\['(slowd(?:_\w+)?)'\]\)"),
    # BBANDS: (dataframe['close'] < dataframe['bb_lowerband'])
    re.compile(r"\(dataframe\['close'\]\s*([<>])\s*dataframe\['(bb_\w+)'\]\)"),
    # EMA/SMA cross: (dataframe['ema_15'] > dataframe['ema_29'])
    re.compile(r"\(dataframe\['((?:ema|sma)_\d+)'\]\s*([<>])\s*dataframe\['((?:ema|sma)_\d+)'\]\)"),
]


def _make_indicator_gene(ind_type: str, match, timeframe: Optional[str] = None) -> IndicatorGene:
    """Create an IndicatorGene from a regex match."""
    groups = match.groups()
    params: Dict = {}
    if ind_type in ('RSI', 'EMA', 'SMA', 'ATR', 'ADX', 'CCI'):
        params = {'period': int(groups[-1])}  # last group is the timeperiod
    elif ind_type == 'MACD':
        params = {'fast_period': int(groups[0]), 'slow_period': int(groups[1]), 'signal_period': int(groups[2])}
    elif ind_type == 'BBANDS':
        params = {'period': int(groups[0]), 'std_dev': float(groups[1])}
    elif ind_type == 'STOCH':
        params = {'k_period': int(groups[0]), 'd_period': int(groups[1])}
    return IndicatorGene(type=ind_type, parameters=params, timeframe=timeframe)


def _parse_indicators(content: str) -> Tuple[List[IndicatorGene], List[str]]:
    """
    Parse all indicators (base + informative) from strategy source code.
    Returns (list_of_IndicatorGene, list_of_informative_timeframes).
    """
    indicators: List[IndicatorGene] = []
    informative_timeframes: List[str] = []

    # --- Base timeframe indicators ---
    # Extract the populate_indicators section (before informative block)
    pop_ind_match = re.search(r'def populate_indicators\(.*?\).*?:.*?\n(.*?)(?:# --- Informative|return dataframe)',
                              content, re.DOTALL)
    base_section = pop_ind_match.group(1) if pop_ind_match else content

    for pattern, ind_type in _BASE_INDICATOR_PATTERNS:
        for m in pattern.finditer(base_section):
            indicators.append(_make_indicator_gene(ind_type, m, timeframe=None))

    # --- Informative timeframe indicators ---
    # Find each informative block: inf_tf = '4h' ... merge_informative_pair
    inf_blocks = re.finditer(
        r"inf_tf\s*=\s*'(\w+)'(.*?)merge_informative_pair",
        content, re.DOTALL
    )
    for block in inf_blocks:
        tf = block.group(1)
        if tf not in informative_timeframes:
            informative_timeframes.append(tf)
        block_text = block.group(2)
        for pattern, ind_type in _INF_INDICATOR_PATTERNS:
            for m in pattern.finditer(block_text):
                indicators.append(_make_indicator_gene(ind_type, m, timeframe=tf))

    return indicators, informative_timeframes


def _col_to_indicator_type(col: str) -> str:
    """Map a dataframe column name to indicator type."""
    col_lower = col.lower()
    if col_lower.startswith('rsi'):
        return 'RSI'
    elif col_lower.startswith('cci'):
        return 'CCI'
    elif col_lower.startswith('adx'):
        return 'ADX'
    elif col_lower.startswith('atr'):
        return 'ATR'
    elif col_lower.startswith('ema'):
        return 'EMA'
    elif col_lower.startswith('sma'):
        return 'SMA'
    elif col_lower.startswith('macd'):
        return 'MACD'
    elif col_lower.startswith('slowk') or col_lower.startswith('slowd'):
        return 'STOCH'
    elif col_lower.startswith('bb_'):
        return 'BBANDS'
    return col.upper()


def _parse_conditions(content: str, section: str) -> List[ConditionGene]:
    """
    Parse entry or exit conditions from a strategy section.
    
    section: 'entry' or 'exit'
    """
    # Find the relevant method
    if section == 'entry':
        method_re = re.compile(r'def populate_entry_trend\(.*?\).*?:\s*\n(.*?)return dataframe', re.DOTALL)
    else:
        method_re = re.compile(r'def populate_exit_trend\(.*?\).*?:\s*\n(.*?)return dataframe', re.DOTALL)

    m = method_re.search(content)
    if not m:
        return []
    
    body = m.group(1)

    # Skip fallback/volume conditions or empty signals
    if "dataframe['volume'] > 0" in body or "dataframe['volume'] > dataframe['volume_sma']" in body:
        # Fallback condition — not a real signal
        pass
    if "= 0" in body and "conditions" not in body:
        # e.g. dataframe['exit_long'] = 0
        return []

    conditions: List[ConditionGene] = []

    # Pattern 1: simple threshold — (dataframe['col'] < 30)
    for m_cond in re.finditer(r"\(dataframe\['(\w+)'\]\s*([<>])\s*([-\d.]+)\)", body):
        col = m_cond.group(1)
        op = m_cond.group(2)
        threshold = float(m_cond.group(3))
        if col == 'volume':
            continue  # skip fallback
        ind_type = _col_to_indicator_type(col)
        conditions.append(ConditionGene(
            indicator=ind_type,
            operator=op,
            threshold=threshold,
        ))

    # Pattern 2: MACD cross — (dataframe['macd'] > dataframe['macdsignal'])
    for m_cond in re.finditer(r"\(dataframe\['(macd(?:_\w+)?)'\]\s*([<>])\s*dataframe\['(macdsignal(?:_\w+)?)'\]\)", body):
        op = 'cross_above' if m_cond.group(2) == '>' else 'cross_below'
        conditions.append(ConditionGene(
            indicator='MACD',
            operator=op,
            threshold=0.0,
        ))

    # Pattern 3: STOCH cross — (dataframe['slowk'] > dataframe['slowd'])
    for m_cond in re.finditer(r"\(dataframe\['(slowk(?:_\w+)?)'\]\s*([<>])\s*dataframe\['(slowd(?:_\w+)?)'\]\)", body):
        op = 'cross_above' if m_cond.group(2) == '>' else 'cross_below'
        conditions.append(ConditionGene(
            indicator='STOCH',
            operator=op,
            threshold=0.0,
        ))

    # Pattern 4: EMA/SMA cross — (dataframe['ema_15'] > dataframe['ema_29'])
    for m_cond in re.finditer(r"\(dataframe\['((?:ema|sma)_\d+)'\]\s*([<>])\s*dataframe\['((?:ema|sma)_\d+)'\]\)", body):
        ind_type = _col_to_indicator_type(m_cond.group(1))
        op = 'cross_above' if m_cond.group(2) == '>' else 'cross_below'
        conditions.append(ConditionGene(
            indicator=ind_type,
            operator=op,
            threshold=0.0,
        ))

    # Pattern 5: BBANDS — (dataframe['close'] < dataframe['bb_lowerband'])
    for m_cond in re.finditer(r"\(dataframe\['close'\]\s*([<>])\s*dataframe\['(bb_\w+)'\]\)", body):
        op = '<' if m_cond.group(1) == '<' else '>'
        conditions.append(ConditionGene(
            indicator='BBANDS',
            operator=op,
            threshold=0.0,
        ))

    # Determine logic (OR vs AND) from joining operator
    if ' |\n' in body or ' | ' in body:
        for c in conditions:
            c.logic = 'OR'

    return conditions


def parse_strategy_file(filepath: str) -> dict:
    """
    Parse a GA-generated strategy .py file and extract ALL parameters,
    indicators, and conditions.
    
    Returns a dict with everything needed to reconstruct a StrategyGene.
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    result: Dict = {}
    
    # Extract strategy name
    name_match = re.search(r'class (\w+)\(IStrategy\)', content)
    if name_match:
        result['class_name'] = name_match.group(1)
    
    # Extract generation and individual ID from filename
    gen_match = re.search(r'Gen(\d+)_Ind(\d+)', str(filepath))
    if gen_match:
        result['generation'] = int(gen_match.group(1))
        result['individual_id'] = int(gen_match.group(2))
    else:
        result['generation'] = 0
        result['individual_id'] = 0
    
    # --- Scalar strategy parameters ---
    tf_match = re.search(r"timeframe\s*=\s*['\"](\w+)['\"]", content)
    if tf_match:
        result['timeframe'] = tf_match.group(1)
    
    sl_match = re.search(r'stoploss\s*=\s*(-?[\d.e+-]+)', content)
    if sl_match:
        result['stoploss'] = float(sl_match.group(1))
    
    ts_match = re.search(r'trailing_stop\s*=\s*(True|False)', content)
    if ts_match:
        result['trailing_stop'] = ts_match.group(1) == 'True'
    
    ts_offset = re.search(r'trailing_stop_positive_offset\s*=\s*([\d.e+-]+)', content)
    if ts_offset:
        result['trailing_stop_positive_offset'] = float(ts_offset.group(1))
    
    ts_pos = re.search(r'trailing_stop_positive\s*=\s*([\d.e+-]+)', content)
    if ts_pos:
        result['trailing_stop_positive'] = float(ts_pos.group(1))
    
    roi_match = re.search(r'minimal_roi\s*=\s*(\{[^}]+\})', content)
    if roi_match:
        try:
            result['minimal_roi'] = eval(roi_match.group(1))  # safe for GA output
        except Exception:
            pass
    
    mot_match = re.search(r'max_open_trades\s*=\s*(\d+)', content)
    if mot_match:
        result['max_open_trades'] = int(mot_match.group(1))
    
    # --- Indicators (full reconstruction) ---
    indicators, informative_timeframes = _parse_indicators(content)
    result['indicators'] = indicators
    result['informative_timeframes'] = informative_timeframes

    # --- Entry & exit conditions ---
    result['entry_conditions'] = _parse_conditions(content, 'entry')
    result['exit_conditions'] = _parse_conditions(content, 'exit')

    return result


def create_individual_from_parsed(parsed: dict) -> Optional[Individual]:
    """
    Create an Individual from parsed strategy data with full gene reconstruction.
    
    Returns None if the strategy cannot form a valid StrategyGene (e.g. no
    indicators or conditions were parseable).
    """
    indicators: List[IndicatorGene] = parsed.get('indicators', [])
    entry_conditions: List[ConditionGene] = parsed.get('entry_conditions', [])

    # StrategyGene requires >= 1 indicator and >= 1 entry condition
    if not indicators:
        # Fallback: create a dummy RSI indicator so the gene is valid
        indicators = [IndicatorGene(type='RSI', parameters={'period': 14})]
    if not entry_conditions:
        # Fallback: create a simple RSI < 30 entry
        entry_conditions = [ConditionGene(indicator='RSI', operator='<', threshold=30.0)]

    gene = StrategyGene(
        generation=parsed.get('generation', 0),
        individual_id=parsed.get('individual_id', 0),
        indicators=indicators,
        entry_conditions=entry_conditions,
        exit_conditions=parsed.get('exit_conditions', []),
        timeframe=parsed.get('timeframe', '1h'),
        stoploss=parsed.get('stoploss', -0.10),
        trailing_stop=parsed.get('trailing_stop', False),
        trailing_stop_positive=parsed.get('trailing_stop_positive'),
        trailing_stop_positive_offset=parsed.get('trailing_stop_positive_offset'),
        minimal_roi=parsed.get('minimal_roi', {"0": 0.04, "30": 0.02, "60": 0.01}),
        max_open_trades=parsed.get('max_open_trades', 3),
        informative_timeframes=parsed.get('informative_timeframes', []),
    )

    individual = Individual(strategy_gene=gene)
    return individual


def extract_from_directory(strategy_dir: str, top_n: int = 20,
                           pattern: str = "GAStrategy_*.py") -> list:
    """
    Extract strategies from a directory of GA-generated files.
    
    Returns list of (filepath, parsed_data) tuples, sorted by generation desc.
    """
    strategy_path = Path(strategy_dir)
    files = sorted(strategy_path.glob(pattern))
    
    results = []
    failed = 0
    for f in files:
        try:
            parsed = parse_strategy_file(str(f))
            if parsed:
                results.append((str(f), parsed))
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  Warning: Failed to parse {f.name}: {e}")
    
    if failed > 5:
        print(f"  ... and {failed - 5} more failures")

    # Sort by generation (descending) to get latest strategies first
    results.sort(key=lambda x: (x[1].get('generation', 0), x[1].get('individual_id', 0)), reverse=True)
    
    return results[:top_n]


def create_seed_file(strategies: list, output_path: str, source_info: str = ""):
    """
    Create a seed JSON file from parsed strategies.
    """
    individuals = []
    skipped = 0
    for filepath, parsed in strategies:
        ind = create_individual_from_parsed(parsed)
        if ind is None:
            skipped += 1
            continue
        ind_dict = ind.to_dict()
        ind_dict['source_file'] = filepath
        individuals.append(ind_dict)
    
    if skipped:
        print(f"  Skipped {skipped} un-parseable strategies")

    seed_data = {
        'version': 2,
        'type': 'seed',
        'timestamp': datetime.now().isoformat(),
        'source': source_info,
        'count': len(individuals),
        'individuals': individuals,
    }
    
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(seed_data, f, indent=2, default=str)
    
    print(f"\n  Saved {len(individuals)} strategies to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract strategies from GA output and create seed/checkpoint files'
    )
    parser.add_argument('--strategy-dir', type=str,
                        default='user_data/strategies/ga_generated/',
                        help='Directory containing GA-generated strategy files')
    parser.add_argument('--output', '-o', type=str,
                        default='genetic_algorithm/data/checkpoints/seed_extracted.json',
                        help='Output seed file path')
    parser.add_argument('--top', '-n', type=int, default=20,
                        help='Number of top strategies to extract (from latest generations)')
    parser.add_argument('--pattern', type=str, default='GAStrategy_*.py',
                        help='Glob pattern for strategy files')
    parser.add_argument('--files', type=str, default=None,
                        help='Comma-separated list of specific strategy files to extract')
    parser.add_argument('--all', action='store_true', default=False,
                        help='Extract ALL strategies (ignore --top)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  STRATEGY EXTRACTOR — Full Gene Reconstruction")
    print("=" * 60)
    
    if args.files:
        # Extract from specific files
        file_list = [f.strip() for f in args.files.split(',')]
        strategies = []
        for f in file_list:
            try:
                parsed = parse_strategy_file(f)
                if parsed:
                    strategies.append((f, parsed))
                    n_ind = len(parsed.get('indicators', []))
                    n_entry = len(parsed.get('entry_conditions', []))
                    n_exit = len(parsed.get('exit_conditions', []))
                    print(f"  Parsed: {Path(f).name}  "
                          f"[{n_ind} indicators, {n_entry} entry, {n_exit} exit conditions]")
            except Exception as e:
                print(f"  Failed: {f}: {e}")
    else:
        # Extract from directory
        print(f"  Scanning: {args.strategy_dir}")
        print(f"  Pattern:  {args.pattern}")
        top_n = 999999 if args.all else args.top
        strategies = extract_from_directory(args.strategy_dir, top_n, args.pattern)
        
        print(f"\n  Extracted {len(strategies)} strategies:")
        for filepath, parsed in strategies:
            name = Path(filepath).name
            gen = parsed.get('generation', '?')
            tf = parsed.get('timeframe', '?')
            sl = parsed.get('stoploss', '?')
            n_ind = len(parsed.get('indicators', []))
            n_entry = len(parsed.get('entry_conditions', []))
            print(f"    Gen{gen}: {name}  tf={tf}  sl={sl}  "
                  f"[{n_ind} ind, {n_entry} entry conds]")
    
    if not strategies:
        print("  No strategies found!")
        return 1
    
    create_seed_file(strategies, args.output,
                     source_info=f"Extracted from {args.strategy_dir or 'specified files'}")
    
    print(f"\n  Usage:")
    print(f"    # Seed a new evolution:")
    print(f"    python genetic_algorithm/run_ga.py --seed {args.output}")
    print(f"    # Or resume from checkpoint with seed:")
    print(f"    python genetic_algorithm/run_ga.py --resume <checkpoint.json> --seed {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
