"""
Indicator Library and Components

Comprehensive library of technical indicators and their configurations
for strategy generation. Adapted from GAFreqTrade for freqtradeForkGA.
"""

import random
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class IndicatorConfig:
    """Configuration for a technical indicator"""
    name: str
    key: str
    calculation_template: str
    params: Dict[str, Tuple[float, float, float]]  # (min, max, default)
    column: str
    type: str  # 'momentum', 'trend', 'volatility'


class IndicatorLibrary:
    """
    Library of available technical indicators with their configurations
    
    Each indicator includes:
    - Calculation code template
    - Parameter ranges (min, max, default)
    - Column names for referencing
    - Indicator type classification
    """
    
    INDICATORS = {
        'rsi': IndicatorConfig(
            name='RSI',
            key='rsi',
            calculation_template='dataframe["rsi"] = ta.RSI(dataframe, timeperiod={period})',
            params={'period': (7, 21, 14)},
            column='rsi',
            type='momentum'
        ),
        
        'macd': IndicatorConfig(
            name='MACD',
            key='macd',
            calculation_template='''macd = ta.MACD(dataframe, fastperiod={fast}, slowperiod={slow}, signalperiod={signal})
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]''',
            params={
                'fast': (8, 16, 12),
                'slow': (20, 30, 26),
                'signal': (7, 12, 9)
            },
            column='macd',
            type='trend'
        ),
        
        'bb': IndicatorConfig(
            name='Bollinger Bands',
            key='bb',
            calculation_template='''bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window={period}, stds={std})
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]
        dataframe["bb_percent"] = (dataframe["close"] - dataframe["bb_lowerband"]) / (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        dataframe["bb_width"] = (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]''',
            params={
                'period': (15, 25, 20),
                'std': (1.5, 2.5, 2.0)
            },
            column='bb_middleband',
            type='volatility'
        ),
        
        'ema': IndicatorConfig(
            name='EMA',
            key='ema',
            calculation_template='dataframe["ema_{period}"] = ta.EMA(dataframe, timeperiod={period})',
            params={'period': (5, 50, 20)},
            column='ema_{period}',
            type='trend'
        ),
        
        'sma': IndicatorConfig(
            name='SMA',
            key='sma',
            calculation_template='dataframe["sma_{period}"] = ta.SMA(dataframe, timeperiod={period})',
            params={'period': (10, 100, 50)},
            column='sma_{period}',
            type='trend'
        ),
        
        'adx': IndicatorConfig(
            name='ADX',
            key='adx',
            calculation_template='dataframe["adx"] = ta.ADX(dataframe, timeperiod={period})',
            params={'period': (10, 20, 14)},
            column='adx',
            type='trend'
        ),
        
        'cci': IndicatorConfig(
            name='CCI',
            key='cci',
            calculation_template='dataframe["cci"] = ta.CCI(dataframe, timeperiod={period})',
            params={'period': (10, 30, 20)},
            column='cci',
            type='momentum'
        ),
        
        'mfi': IndicatorConfig(
            name='MFI',
            key='mfi',
            calculation_template='dataframe["mfi"] = ta.MFI(dataframe, timeperiod={period})',
            params={'period': (10, 20, 14)},
            column='mfi',
            type='momentum'
        ),
        
        'stoch': IndicatorConfig(
            name='Stochastic',
            key='stoch',
            calculation_template='''stoch = ta.STOCH(dataframe, fastk_period={fastk}, slowk_period={slowk}, slowd_period={slowd})
        dataframe["slowk"] = stoch["slowk"]
        dataframe["slowd"] = stoch["slowd"]''',
            params={
                'fastk': (3, 7, 5),
                'slowk': (2, 5, 3),
                'slowd': (2, 5, 3)
            },
            column='slowk',
            type='momentum'
        ),
        
        'atr': IndicatorConfig(
            name='ATR',
            key='atr',
            calculation_template='dataframe["atr"] = ta.ATR(dataframe, timeperiod={period})',
            params={'period': (10, 20, 14)},
            column='atr',
            type='volatility'
        ),
    }
    
    @classmethod
    def get_all_indicators(cls) -> Dict[str, IndicatorConfig]:
        """Get all available indicators"""
        return cls.INDICATORS
    
    @classmethod
    def get_indicator(cls, key: str) -> IndicatorConfig:
        """Get a specific indicator by key"""
        return cls.INDICATORS.get(key)
    
    @classmethod
    def get_random_indicators(cls, min_count: int = 2, max_count: int = 6, 
                            allowed_keys: List[str] = None) -> List[Dict[str, Any]]:
        """
        Select random indicators with randomized parameters
        
        Args:
            min_count: Minimum number of indicators
            max_count: Maximum number of indicators
            allowed_keys: List of allowed indicator keys (None = all)
            
        Returns:
            List of indicator dictionaries with selected parameters
        """
        available_keys = list(cls.INDICATORS.keys())
        if allowed_keys:
            available_keys = [k for k in available_keys if k in allowed_keys]
        
        count = random.randint(min_count, min(max_count, len(available_keys)))
        selected_keys = random.sample(available_keys, count)
        
        indicators = []
        for key in selected_keys:
            indicator_config = cls.INDICATORS[key]
            
            # Randomize parameters within ranges
            params = {}
            for param_name, (min_val, max_val, default) in indicator_config.params.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = random.randint(int(min_val), int(max_val))
                else:
                    params[param_name] = round(random.uniform(float(min_val), float(max_val)), 2)
            
            indicators.append({
                'key': key,
                'name': indicator_config.name,
                'calculation': indicator_config.calculation_template,
                'params': indicator_config.params,
                'selected_params': params,
                'column': indicator_config.column,
                'type': indicator_config.type
            })
        
        return indicators
    
    @classmethod
    def get_indicators_by_type(cls, indicator_type: str) -> List[str]:
        """Get all indicators of a specific type"""
        return [key for key, config in cls.INDICATORS.items() if config.type == indicator_type]


class ConditionTemplates:
    """
    Templates for generating entry/exit conditions based on indicators
    
    Each indicator has a set of condition templates that can be used
    to create buy and sell signals.
    """
    
    TEMPLATES = {
        'rsi': {
            'buy': [
                'dataframe["rsi"] < {buy_rsi_threshold}',
                'dataframe["rsi"].shift(1) > dataframe["rsi"]',  # RSI falling
                '(dataframe["rsi"] < {buy_rsi_threshold}) & (dataframe["rsi"].shift(1) >= {buy_rsi_threshold})',  # Crossed below
            ],
            'sell': [
                'dataframe["rsi"] > {sell_rsi_threshold}',
                'dataframe["rsi"].shift(1) < dataframe["rsi"]',  # RSI rising
                '(dataframe["rsi"] > {sell_rsi_threshold}) & (dataframe["rsi"].shift(1) <= {sell_rsi_threshold})',  # Crossed above
            ]
        },
        'macd': {
            'buy': [
                'dataframe["macd"] > dataframe["macdsignal"]',
                'dataframe["macdhist"] > 0',
                '(qtpylib.crossed_above(dataframe["macd"], dataframe["macdsignal"]))',
            ],
            'sell': [
                'dataframe["macd"] < dataframe["macdsignal"]',
                'dataframe["macdhist"] < 0',
                '(qtpylib.crossed_below(dataframe["macd"], dataframe["macdsignal"]))',
            ]
        },
        'bb': {
            'buy': [
                'dataframe["close"] < dataframe["bb_lowerband"]',
                'dataframe["bb_percent"] < {bb_buy_threshold}',
                '(qtpylib.crossed_above(dataframe["close"], dataframe["bb_lowerband"]))',
            ],
            'sell': [
                'dataframe["close"] > dataframe["bb_upperband"]',
                'dataframe["bb_percent"] > {bb_sell_threshold}',
                '(qtpylib.crossed_below(dataframe["close"], dataframe["bb_upperband"]))',
            ]
        },
        'ema': {
            'buy': [
                'dataframe["close"] > dataframe["ema_{ema_period}"]',
                '(qtpylib.crossed_above(dataframe["close"], dataframe["ema_{ema_period}"]))',
            ],
            'sell': [
                'dataframe["close"] < dataframe["ema_{ema_period}"]',
                '(qtpylib.crossed_below(dataframe["close"], dataframe["ema_{ema_period}"]))',
            ]
        },
        'sma': {
            'buy': [
                'dataframe["close"] > dataframe["sma_{sma_period}"]',
                '(qtpylib.crossed_above(dataframe["close"], dataframe["sma_{sma_period}"]))',
            ],
            'sell': [
                'dataframe["close"] < dataframe["sma_{sma_period}"]',
                '(qtpylib.crossed_below(dataframe["close"], dataframe["sma_{sma_period}"]))',
            ]
        },
        'adx': {
            'buy': [
                'dataframe["adx"] > {adx_threshold}',
            ],
            'sell': [
                'dataframe["adx"] < {adx_threshold}',
            ]
        },
        'cci': {
            'buy': [
                'dataframe["cci"] < {cci_buy_threshold}',
                '(dataframe["cci"] < {cci_buy_threshold}) & (dataframe["cci"].shift(1) >= {cci_buy_threshold})',
            ],
            'sell': [
                'dataframe["cci"] > {cci_sell_threshold}',
                '(dataframe["cci"] > {cci_sell_threshold}) & (dataframe["cci"].shift(1) <= {cci_sell_threshold})',
            ]
        },
        'mfi': {
            'buy': [
                'dataframe["mfi"] < {mfi_buy_threshold}',
            ],
            'sell': [
                'dataframe["mfi"] > {mfi_sell_threshold}',
            ]
        },
        'stoch': {
            'buy': [
                'dataframe["slowk"] < {stoch_buy_threshold}',
                '(qtpylib.crossed_above(dataframe["slowk"], dataframe["slowd"]))',
                '(dataframe["slowk"] < {stoch_buy_threshold}) & (dataframe["slowd"] < {stoch_buy_threshold})',
            ],
            'sell': [
                'dataframe["slowk"] > {stoch_sell_threshold}',
                '(qtpylib.crossed_below(dataframe["slowk"], dataframe["slowd"]))',
                '(dataframe["slowk"] > {stoch_sell_threshold}) & (dataframe["slowd"] > {stoch_sell_threshold})',
            ]
        }
    }
    
    # Default threshold values for indicators
    DEFAULT_THRESHOLDS = {
        'buy_rsi_threshold': (20, 40, 30),  # (min, max, default)
        'sell_rsi_threshold': (60, 80, 70),
        'bb_buy_threshold': (0.0, 0.3, 0.1),
        'bb_sell_threshold': (0.7, 1.0, 0.9),
        'adx_threshold': (20, 40, 25),
        'cci_buy_threshold': (-200, -100, -150),
        'cci_sell_threshold': (100, 200, 150),
        'mfi_buy_threshold': (10, 30, 20),
        'mfi_sell_threshold': (70, 90, 80),
        'stoch_buy_threshold': (10, 30, 20),
        'stoch_sell_threshold': (70, 90, 80),
    }
    
    @classmethod
    def get_templates(cls, indicator_key: str, signal_type: str) -> List[str]:
        """
        Get condition templates for an indicator
        
        Args:
            indicator_key: Indicator key (e.g., 'rsi', 'macd')
            signal_type: 'buy' or 'sell'
            
        Returns:
            List of condition template strings
        """
        return cls.TEMPLATES.get(indicator_key, {}).get(signal_type, [])
    
    @classmethod
    def get_random_threshold(cls, threshold_name: str) -> float:
        """Get a random value within the threshold range"""
        if threshold_name in cls.DEFAULT_THRESHOLDS:
            min_val, max_val, default = cls.DEFAULT_THRESHOLDS[threshold_name]
            if isinstance(min_val, int):
                return random.randint(int(min_val), int(max_val))
            else:
                return round(random.uniform(float(min_val), float(max_val)), 2)
        return 0
