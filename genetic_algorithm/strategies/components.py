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
        
        # === NEW INDICATORS ===
        
        'supertrend': IndicatorConfig(
            name='SuperTrend',
            key='supertrend',
            calculation_template='''# SuperTrend calculation
        hl2 = (dataframe["high"] + dataframe["low"]) / 2
        atr_st = ta.ATR(dataframe, timeperiod={period})
        upperband = hl2 + ({multiplier} * atr_st)
        lowerband = hl2 - ({multiplier} * atr_st)
        
        supertrend = [True] * len(dataframe)
        for i in range(1, len(dataframe)):
            if dataframe["close"].iloc[i] > upperband.iloc[i-1]:
                supertrend[i] = True
            elif dataframe["close"].iloc[i] < lowerband.iloc[i-1]:
                supertrend[i] = False
            else:
                supertrend[i] = supertrend[i-1]
                if supertrend[i] and lowerband.iloc[i] < lowerband.iloc[i-1]:
                    lowerband.iloc[i] = lowerband.iloc[i-1]
                if not supertrend[i] and upperband.iloc[i] > upperband.iloc[i-1]:
                    upperband.iloc[i] = upperband.iloc[i-1]
        
        dataframe["supertrend"] = supertrend
        dataframe["supertrend_upper"] = upperband
        dataframe["supertrend_lower"] = lowerband''',
            params={
                'period': (7, 14, 10),
                'multiplier': (2.0, 4.0, 3.0)
            },
            column='supertrend',
            type='trend'
        ),
        
        'ichimoku': IndicatorConfig(
            name='Ichimoku',
            key='ichimoku',
            calculation_template='''# Ichimoku Cloud
        from freqtrade.vendor.qtpylib.indicators import ichimoku
        ichi = ichimoku(dataframe, tenkan={tenkan_period}, kijun={kijun_period}, senkou={senkou_b_period})
        dataframe["tenkan_sen"] = ichi["tenkan_sen"]
        dataframe["kijun_sen"] = ichi["kijun_sen"]
        dataframe["senkou_span_a"] = ichi["senkou_span_a"]
        dataframe["senkou_span_b"] = ichi["senkou_span_b"]
        dataframe["cloud_green"] = ichi["cloud_green"]
        dataframe["cloud_red"] = ichi["cloud_red"]''',
            params={
                'tenkan_period': (7, 12, 9),
                'kijun_period': (20, 30, 26),
                'senkou_b_period': (40, 60, 52)
            },
            column='tenkan_sen',
            type='trend'
        ),
        
        'donchian': IndicatorConfig(
            name='Donchian',
            key='donchian',
            calculation_template='''# Donchian Channels
        dataframe["donchian_upper"] = dataframe["high"].rolling({period}).max()
        dataframe["donchian_lower"] = dataframe["low"].rolling({period}).min()
        dataframe["donchian_mid"] = (dataframe["donchian_upper"] + dataframe["donchian_lower"]) / 2''',
            params={'period': (10, 30, 20)},
            column='donchian_mid',
            type='trend'
        ),
        
        'vwap': IndicatorConfig(
            name='VWAP',
            key='vwap',
            calculation_template='''# Volume Weighted Average Price
        typical_price = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3
        dataframe["vwap"] = (typical_price * dataframe["volume"]).cumsum() / dataframe["volume"].cumsum()''',
            params={},
            column='vwap',
            type='trend'
        ),
        
        'cmf': IndicatorConfig(
            name='CMF',
            key='cmf',
            calculation_template='''# Chaikin Money Flow
        mfv = ((dataframe["close"] - dataframe["low"]) - (dataframe["high"] - dataframe["close"])) / (dataframe["high"] - dataframe["low"])
        mfv = mfv.fillna(0) * dataframe["volume"]
        dataframe["cmf"] = mfv.rolling({period}).sum() / dataframe["volume"].rolling({period}).sum()''',
            params={'period': (10, 25, 20)},
            column='cmf',
            type='momentum'
        ),
        
        'vroc': IndicatorConfig(
            name='VROC',
            key='vroc',
            calculation_template='''# Volume Rate of Change
        dataframe["vroc"] = ((dataframe["volume"] - dataframe["volume"].shift({period})) / dataframe["volume"].shift({period})) * 100''',
            params={'period': (5, 20, 12)},
            column='vroc',
            type='momentum'
        ),
        
        'psar': IndicatorConfig(
            name='PSAR',
            key='psar',
            calculation_template='dataframe["psar"] = ta.SAR(dataframe, acceleration={acceleration}, maximum={maximum})',
            params={
                'acceleration': (0.01, 0.05, 0.02),
                'maximum': (0.1, 0.3, 0.2)
            },
            column='psar',
            type='trend'
        ),
        
        # === CANDLESTICK PATTERNS ===
        
        'cdl_engulfing': IndicatorConfig(
            name='Engulfing',
            key='cdl_engulfing',
            calculation_template='dataframe["cdl_engulfing"] = ta.CDLENGULFING(dataframe)',
            params={},
            column='cdl_engulfing',
            type='pattern'
        ),
        
        'cdl_hammer': IndicatorConfig(
            name='Hammer',
            key='cdl_hammer',
            calculation_template='dataframe["cdl_hammer"] = ta.CDLHAMMER(dataframe)',
            params={},
            column='cdl_hammer',
            type='pattern'
        ),
        
        'cdl_doji': IndicatorConfig(
            name='Doji',
            key='cdl_doji',
            calculation_template='dataframe["cdl_doji"] = ta.CDLDOJI(dataframe)',
            params={},
            column='cdl_doji',
            type='pattern'
        ),
        
        'cdl_morningstar': IndicatorConfig(
            name='Morning Star',
            key='cdl_morningstar',
            calculation_template='dataframe["cdl_morningstar"] = ta.CDLMORNINGSTAR(dataframe, penetration={penetration})',
            params={'penetration': (0.0, 0.3, 0.0)},
            column='cdl_morningstar',
            type='pattern'
        ),
        
        'cdl_eveningstar': IndicatorConfig(
            name='Evening Star',
            key='cdl_eveningstar',
            calculation_template='dataframe["cdl_eveningstar"] = ta.CDLEVENINGSTAR(dataframe, penetration={penetration})',
            params={'penetration': (0.0, 0.3, 0.0)},
            column='cdl_eveningstar',
            type='pattern'
        ),
        
        'cdl_shootingstar': IndicatorConfig(
            name='Shooting Star',
            key='cdl_shootingstar',
            calculation_template='dataframe["cdl_shootingstar"] = ta.CDLSHOOTINGSTAR(dataframe)',
            params={},
            column='cdl_shootingstar',
            type='pattern'
        ),
        
        'cdl_harami': IndicatorConfig(
            name='Harami',
            key='cdl_harami',
            calculation_template='dataframe["cdl_harami"] = ta.CDLHARAMI(dataframe)',
            params={},
            column='cdl_harami',
            type='pattern'
        ),
        
        'cdl_piercing': IndicatorConfig(
            name='Piercing',
            key='cdl_piercing',
            calculation_template='dataframe["cdl_piercing"] = ta.CDLPIERCING(dataframe)',
            params={},
            column='cdl_piercing',
            type='pattern'
        ),
        
        'cdl_darkcloud': IndicatorConfig(
            name='Dark Cloud',
            key='cdl_darkcloud',
            calculation_template='dataframe["cdl_darkcloud"] = ta.CDLDARKCLOUDCOVER(dataframe)',
            params={},
            column='cdl_darkcloud',
            type='pattern'
        ),
        
        'cdl_3whitesoldiers': IndicatorConfig(
            name='Three White Soldiers',
            key='cdl_3whitesoldiers',
            calculation_template='dataframe["cdl_3whitesoldiers"] = ta.CDL3WHITESOLDIERS(dataframe)',
            params={},
            column='cdl_3whitesoldiers',
            type='pattern'
        ),
        
        'cdl_3blackcrows': IndicatorConfig(
            name='Three Black Crows',
            key='cdl_3blackcrows',
            calculation_template='dataframe["cdl_3blackcrows"] = ta.CDL3BLACKCROWS(dataframe)',
            params={},
            column='cdl_3blackcrows',
            type='pattern'
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
        },
        
        # === NEW INDICATOR TEMPLATES ===
        
        'supertrend': {
            'buy': [
                'dataframe["supertrend"] == True',
                '(dataframe["supertrend"] == True) & (dataframe["supertrend"].shift(1) == False)',  # Trend flip to bullish
                'dataframe["close"] > dataframe["supertrend_lower"]',
            ],
            'sell': [
                'dataframe["supertrend"] == False',
                '(dataframe["supertrend"] == False) & (dataframe["supertrend"].shift(1) == True)',  # Trend flip to bearish
                'dataframe["close"] < dataframe["supertrend_upper"]',
            ]
        },
        
        'ichimoku': {
            'buy': [
                'dataframe["close"] > dataframe["senkou_span_a"]',
                'dataframe["tenkan_sen"] > dataframe["kijun_sen"]',
                '(qtpylib.crossed_above(dataframe["tenkan_sen"], dataframe["kijun_sen"]))',
                'dataframe["cloud_green"] == True',
            ],
            'sell': [
                'dataframe["close"] < dataframe["senkou_span_b"]',
                'dataframe["tenkan_sen"] < dataframe["kijun_sen"]',
                '(qtpylib.crossed_below(dataframe["tenkan_sen"], dataframe["kijun_sen"]))',
                'dataframe["cloud_red"] == True',
            ]
        },
        
        'donchian': {
            'buy': [
                '(qtpylib.crossed_above(dataframe["close"], dataframe["donchian_upper"]))',  # Breakout above
                'dataframe["close"] > dataframe["donchian_mid"]',
            ],
            'sell': [
                '(qtpylib.crossed_below(dataframe["close"], dataframe["donchian_lower"]))',  # Breakout below
                'dataframe["close"] < dataframe["donchian_mid"]',
            ]
        },
        
        'vwap': {
            'buy': [
                '(qtpylib.crossed_above(dataframe["close"], dataframe["vwap"]))',
                'dataframe["close"] > dataframe["vwap"]',
            ],
            'sell': [
                '(qtpylib.crossed_below(dataframe["close"], dataframe["vwap"]))',
                'dataframe["close"] < dataframe["vwap"]',
            ]
        },
        
        'cmf': {
            'buy': [
                'dataframe["cmf"] > {cmf_buy_threshold}',
                '(dataframe["cmf"] > 0) & (dataframe["cmf"].shift(1) <= 0)',  # Crossed above zero
            ],
            'sell': [
                'dataframe["cmf"] < {cmf_sell_threshold}',
                '(dataframe["cmf"] < 0) & (dataframe["cmf"].shift(1) >= 0)',  # Crossed below zero
            ]
        },
        
        'vroc': {
            'buy': [
                'dataframe["vroc"] > {vroc_threshold}',  # Volume spike
            ],
            'sell': [
                'dataframe["vroc"] < -{vroc_threshold}',  # Volume drop
            ]
        },
        
        'psar': {
            'buy': [
                'dataframe["close"] > dataframe["psar"]',
                '(dataframe["close"] > dataframe["psar"]) & (dataframe["close"].shift(1) <= dataframe["psar"].shift(1))',  # Flip bullish
            ],
            'sell': [
                'dataframe["close"] < dataframe["psar"]',
                '(dataframe["close"] < dataframe["psar"]) & (dataframe["close"].shift(1) >= dataframe["psar"].shift(1))',  # Flip bearish
            ]
        },
        
        # === CANDLESTICK PATTERN TEMPLATES ===
        # TALib CDL* returns: >0 bullish, <0 bearish, 0 no pattern
        
        'cdl_engulfing': {
            'buy': [
                'dataframe["cdl_engulfing"] > 0',  # Bullish engulfing
            ],
            'sell': [
                'dataframe["cdl_engulfing"] < 0',  # Bearish engulfing
            ]
        },
        
        'cdl_hammer': {
            'buy': [
                'dataframe["cdl_hammer"] != 0',  # Hammer (bullish reversal)
            ],
            'sell': []  # Hammer is bullish only
        },
        
        'cdl_doji': {
            'buy': [
                'dataframe["cdl_doji"] != 0',  # Doji (indecision, watch for reversal)
            ],
            'sell': [
                'dataframe["cdl_doji"] != 0',  # Can be exit signal too
            ]
        },
        
        'cdl_morningstar': {
            'buy': [
                'dataframe["cdl_morningstar"] != 0',  # Morning star (bullish reversal)
            ],
            'sell': []  # Morning star is bullish only
        },
        
        'cdl_eveningstar': {
            'buy': [],  # Evening star is bearish only
            'sell': [
                'dataframe["cdl_eveningstar"] != 0',  # Evening star (bearish reversal)
            ]
        },
        
        'cdl_shootingstar': {
            'buy': [],  # Shooting star is bearish only
            'sell': [
                'dataframe["cdl_shootingstar"] != 0',  # Shooting star (bearish reversal)
            ]
        },
        
        'cdl_harami': {
            'buy': [
                'dataframe["cdl_harami"] > 0',  # Bullish harami
            ],
            'sell': [
                'dataframe["cdl_harami"] < 0',  # Bearish harami
            ]
        },
        
        'cdl_piercing': {
            'buy': [
                'dataframe["cdl_piercing"] != 0',  # Piercing line (bullish)
            ],
            'sell': []  # Piercing is bullish only
        },
        
        'cdl_darkcloud': {
            'buy': [],  # Dark cloud is bearish only
            'sell': [
                'dataframe["cdl_darkcloud"] != 0',  # Dark cloud cover (bearish)
            ]
        },
        
        'cdl_3whitesoldiers': {
            'buy': [
                'dataframe["cdl_3whitesoldiers"] != 0',  # Three white soldiers (strong bullish)
            ],
            'sell': []  # 3WS is bullish only
        },
        
        'cdl_3blackcrows': {
            'buy': [],  # 3BC is bearish only
            'sell': [
                'dataframe["cdl_3blackcrows"] != 0',  # Three black crows (strong bearish)
            ]
        },
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
        # New indicator thresholds
        'cmf_buy_threshold': (0.05, 0.2, 0.1),
        'cmf_sell_threshold': (-0.2, -0.05, -0.1),
        'vroc_threshold': (50, 200, 100),  # Volume rate of change spike threshold
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
