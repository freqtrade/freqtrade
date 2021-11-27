# pragma pylint: disable=missing-docstring, W0212, line-too-long, C0103, C0330, unused-argument
import logging

import pytest

from freqtrade.data.history import get_timerange
from freqtrade.enums import SellType
from freqtrade.optimize.backtesting import Backtesting
from tests.conftest import patch_exchange
from tests.optimize import (BTContainer, BTrade, _build_backtest_dataframe,
                            _get_frame_time_from_offset, tests_timeframe)


# Test 0: Sell with signal sell in candle 3
# Test with Stop-loss at 1%
tc0 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4975, 4987, 6172, 0, 0],  # enter trade (signal on last candle)
    [2, 4987, 5012, 4986, 4986, 6172, 0, 0],  # exit with stoploss hit
    [3, 5010, 5010, 4980, 5010, 6172, 0, 1],
    [4, 5010, 5011, 4977, 4995, 6172, 0, 0],
    [5, 4995, 4995, 4950, 4950, 6172, 0, 0]],
    stop_loss=-0.01, roi={"0": 1}, profit_perc=0.002, use_sell_signal=True,
    trades=[BTrade(sell_reason=SellType.SELL_SIGNAL, open_tick=1, close_tick=4)]
)

# Test 1: Stop-Loss Triggered 1% loss
# Test with Stop-loss at 1%
tc1 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4975, 4987, 6172, 0, 0],  # enter trade (signal on last candle)
    [2, 4987, 5012, 4600, 4600, 6172, 0, 0],  # exit with stoploss hit
    [3, 4975, 5000, 4975, 4977, 6172, 0, 0],
    [4, 4977, 4995, 4977, 4995, 6172, 0, 0],
    [5, 4995, 4995, 4950, 4950, 6172, 0, 0]],
    stop_loss=-0.01, roi={"0": 1}, profit_perc=-0.01,
    trades=[BTrade(sell_reason=SellType.STOP_LOSS, open_tick=1, close_tick=2)]
)


# Test 2: Minus 4% Low, minus 1% close
# Test with Stop-Loss at 3%
tc2 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4975, 4987, 6172, 0, 0],  # enter trade (signal on last candle)
    [2, 4987, 5012, 4962, 4975, 6172, 0, 0],
    [3, 4975, 5000, 4800, 4962, 6172, 0, 0],  # exit with stoploss hit
    [4, 4962, 4987, 4937, 4950, 6172, 0, 0],
    [5, 4950, 4975, 4925, 4950, 6172, 0, 0]],
    stop_loss=-0.03, roi={"0": 1}, profit_perc=-0.03,
    trades=[BTrade(sell_reason=SellType.STOP_LOSS, open_tick=1, close_tick=3)]
)


# Test 3: Multiple trades.
#         Candle drops 4%, Recovers 1%.
#         Entry Criteria Met
#         Candle drops 20%
#  Trade-A: Stop-Loss Triggered 2% Loss
#           Trade-B: Stop-Loss Triggered 2% Loss
tc3 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4975, 4987, 6172, 0, 0],  # enter trade (signal on last candle)
    [2, 4987, 5012, 4800, 4975, 6172, 0, 0],  # exit with stoploss hit
    [3, 4975, 5000, 4950, 4962, 6172, 1, 0],
    [4, 4975, 5000, 4950, 4962, 6172, 0, 0],  # enter trade 2 (signal on last candle)
    [5, 4962, 4987, 4000, 4000, 6172, 0, 0],  # exit with stoploss hit
    [6, 4950, 4975, 4950, 4950, 6172, 0, 0]],
    stop_loss=-0.02, roi={"0": 1}, profit_perc=-0.04,
    trades=[BTrade(sell_reason=SellType.STOP_LOSS, open_tick=1, close_tick=2),
            BTrade(sell_reason=SellType.STOP_LOSS, open_tick=4, close_tick=5)]
)

# Test 4: Minus 3% / recovery +15%
# Candle Data for test 3 – Candle drops 3% Closed 15% up
# Test with Stop-loss at 2% ROI 6%
# Stop-Loss Triggered 2% Loss
tc4 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4975, 4987, 6172, 0, 0],  # enter trade (signal on last candle)
    [2, 4987, 5750, 4850, 5750, 6172, 0, 0],  # Exit with stoploss hit
    [3, 4975, 5000, 4950, 4962, 6172, 0, 0],
    [4, 4962, 4987, 4937, 4950, 6172, 0, 0],
    [5, 4950, 4975, 4925, 4950, 6172, 0, 0]],
    stop_loss=-0.02, roi={"0": 0.06}, profit_perc=-0.02,
    trades=[BTrade(sell_reason=SellType.STOP_LOSS, open_tick=1, close_tick=2)]
)

# Test 5: Drops 0.5% Closes +20%, ROI triggers 3% Gain
# stop-loss: 1%, ROI: 3%
tc5 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4980, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4980, 4987, 6172, 0, 0],  # enter trade (signal on last candle)
    [2, 4987, 5025, 4975, 4987, 6172, 0, 0],
    [3, 4975, 6000, 4975, 6000, 6172, 0, 0],  # ROI
    [4, 4962, 4987, 4962, 4972, 6172, 0, 0],
    [5, 4950, 4975, 4925, 4950, 6172, 0, 0]],
    stop_loss=-0.01, roi={"0": 0.03}, profit_perc=0.03,
    trades=[BTrade(sell_reason=SellType.ROI, open_tick=1, close_tick=3)]
)

# Test 6: Drops 3% / Recovers 6% Positive / Closes 1% positve, Stop-Loss triggers 2% Loss
# stop-loss: 2% ROI: 5%
tc6 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4975, 4987, 6172, 0, 0],  # enter trade (signal on last candle)
    [2, 4987, 5300, 4850, 5050, 6172, 0, 0],  # Exit with stoploss
    [3, 4975, 5000, 4950, 4962, 6172, 0, 0],
    [4, 4962, 4987, 4950, 4950, 6172, 0, 0],
    [5, 4950, 4975, 4925, 4950, 6172, 0, 0]],
    stop_loss=-0.02, roi={"0": 0.05}, profit_perc=-0.02,
    trades=[BTrade(sell_reason=SellType.STOP_LOSS, open_tick=1, close_tick=2)]
)

# Test 7: 6% Positive / 1% Negative / Close 1% Positve, ROI Triggers 3% Gain
# stop-loss: 2% ROI: 3%
tc7 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4975, 4987, 6172, 0, 0],
    [2, 4987, 5300, 4950, 5050, 6172, 0, 0],
    [3, 4975, 5000, 4950, 4962, 6172, 0, 0],
    [4, 4962, 4987, 4950, 4950, 6172, 0, 0],
    [5, 4950, 4975, 4925, 4950, 6172, 0, 0]],
    stop_loss=-0.02, roi={"0": 0.03}, profit_perc=0.03,
    trades=[BTrade(sell_reason=SellType.ROI, open_tick=1, close_tick=2)]
)


# Test 8: trailing_stop should raise so candle 3 causes a stoploss.
# stop-loss: 10%, ROI: 10% (should not apply), stoploss adjusted in candle 2
tc8 = BTContainer(data=[
    # D   O     H     L    C     V    B  S
    [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
    [1, 5000, 5050, 4950, 5000, 6172, 0, 0],
    [2, 5000, 5250, 4750, 4850, 6172, 0, 0],
    [3, 4850, 5050, 4650, 4750, 6172, 0, 0],
    [4, 4750, 4950, 4350, 4750, 6172, 0, 0]],
    stop_loss=-0.10, roi={"0": 0.10}, profit_perc=-0.055, trailing_stop=True,
    trades=[BTrade(sell_reason=SellType.TRAILING_STOP_LOSS, open_tick=1, close_tick=3)]
)


# Test 9: trailing_stop should raise - high and low in same candle.
# stop-loss: 10%, ROI: 10% (should not apply), stoploss adjusted in candle 3
tc9 = BTContainer(data=[
    # D   O     H     L     C    V    B  S
    [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
    [1, 5000, 5050, 4950, 5000, 6172, 0, 0],
    [2, 5000, 5050, 4950, 5000, 6172, 0, 0],
    [3, 5000, 5200, 4550, 4850, 6172, 0, 0],
    [4, 4750, 4950, 4350, 4750, 6172, 0, 0]],
    stop_loss=-0.10, roi={"0": 0.10}, profit_perc=-0.064, trailing_stop=True,
    trades=[BTrade(sell_reason=SellType.TRAILING_STOP_LOSS, open_tick=1, close_tick=3)]
)

# Test 10: trailing_stop should raise so candle 3 causes a stoploss
# without applying trailing_stop_positive since stoploss_offset is at 10%.
# stop-loss: 10%, ROI: 10% (should not apply), stoploss adjusted candle 2
tc10 = BTContainer(data=[
    # D   O     H     L     C    V    B  S
    [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
    [1, 5000, 5100, 4950, 5100, 6172, 0, 0],
    [2, 5100, 5251, 5100, 5100, 6172, 0, 0],
    [3, 4850, 5050, 4650, 4750, 6172, 0, 0],
    [4, 4750, 4950, 4350, 4750, 6172, 0, 0]],
    stop_loss=-0.10, roi={"0": 0.10}, profit_perc=-0.1, trailing_stop=True,
    trailing_only_offset_is_reached=True, trailing_stop_positive_offset=0.10,
    trailing_stop_positive=0.03,
    trades=[BTrade(sell_reason=SellType.STOP_LOSS, open_tick=1, close_tick=4)]
)

# Test 11: trailing_stop should raise so candle 3 causes a stoploss
# applying a positive trailing stop of 3% since stop_positive_offset is reached.
# stop-loss: 10%, ROI: 10% (should not apply), stoploss adjusted candle 2
tc11 = BTContainer(data=[
    # D   O     H     L     C    V    B  S
    [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
    [1, 5000, 5100, 4950, 5100, 6172, 0, 0],
    [2, 5100, 5251, 5100, 5100, 6172, 0, 0],
    [3, 5000, 5150, 4650, 4750, 6172, 0, 0],
    [4, 4750, 4950, 4350, 4750, 6172, 0, 0]],
    stop_loss=-0.10, roi={"0": 0.10}, profit_perc=0.019, trailing_stop=True,
    trailing_only_offset_is_reached=True, trailing_stop_positive_offset=0.05,
    trailing_stop_positive=0.03,
    trades=[BTrade(sell_reason=SellType.TRAILING_STOP_LOSS, open_tick=1, close_tick=3)]
)

# Test 12: trailing_stop should raise in candle 2 and cause a stoploss in the same candle
# applying a positive trailing stop of 3% since stop_positive_offset is reached.
# stop-loss: 10%, ROI: 10% (should not apply), stoploss adjusted candle 2
tc12 = BTContainer(data=[
    # D   O     H     L     C    V    B  S
    [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
    [1, 5000, 5100, 4950, 5100, 6172, 0, 0],
    [2, 5100, 5251, 4650, 5100, 6172, 0, 0],
    [3, 4850, 5050, 4650, 4750, 6172, 0, 0],
    [4, 4750, 4950, 4350, 4750, 6172, 0, 0]],
    stop_loss=-0.10, roi={"0": 0.10}, profit_perc=0.019, trailing_stop=True,
    trailing_only_offset_is_reached=True, trailing_stop_positive_offset=0.05,
    trailing_stop_positive=0.03,
    trades=[BTrade(sell_reason=SellType.TRAILING_STOP_LOSS, open_tick=1, close_tick=2)]
)

# Test 13: Buy and sell ROI on same candle
# stop-loss: 10% (should not apply), ROI: 1%
tc13 = BTContainer(data=[
    # D   O     H     L     C    V    B  S
    [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
    [1, 5000, 5100, 4950, 5100, 6172, 0, 0],
    [2, 5100, 5251, 4850, 5100, 6172, 0, 0],
    [3, 4850, 5050, 4750, 4750, 6172, 0, 0],
    [4, 4750, 4950, 4750, 4750, 6172, 0, 0]],
    stop_loss=-0.10, roi={"0": 0.01}, profit_perc=0.01,
    trades=[BTrade(sell_reason=SellType.ROI, open_tick=1, close_tick=1)]
)

# Test 14 - Buy and Stoploss on same candle
# stop-loss: 5%, ROI: 10% (should not apply)
tc14 = BTContainer(data=[
    # D   O     H     L     C    V    B  S
    [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
    [1, 5000, 5100, 4600, 5100, 6172, 0, 0],
    [2, 5100, 5251, 4850, 5100, 6172, 0, 0],
    [3, 4850, 5050, 4750, 4750, 6172, 0, 0],
    [4, 4750, 4950, 4350, 4750, 6172, 0, 0]],
    stop_loss=-0.05, roi={"0": 0.10}, profit_perc=-0.05,
    trades=[BTrade(sell_reason=SellType.STOP_LOSS, open_tick=1, close_tick=1)]
)


# Test 15 - Buy and ROI on same candle, followed by buy and Stoploss on next candle
# stop-loss: 5%, ROI: 10% (should not apply)
tc15 = BTContainer(data=[
    # D   O     H     L     C    V    B  S
    [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
    [1, 5000, 5100, 4900, 5100, 6172, 1, 0],
    [2, 5100, 5251, 4650, 5100, 6172, 0, 0],
    [3, 4850, 5050, 4750, 4750, 6172, 0, 0],
    [4, 4750, 4950, 4350, 4750, 6172, 0, 0]],
    stop_loss=-0.05, roi={"0": 0.01}, profit_perc=-0.04,
    trades=[BTrade(sell_reason=SellType.ROI, open_tick=1, close_tick=1),
            BTrade(sell_reason=SellType.STOP_LOSS, open_tick=2, close_tick=2)]
)

# Test 16: Buy, hold for 65 min, then forcesell using roi=-1
# Causes negative profit even though sell-reason is ROI.
# stop-loss: 10%, ROI: 10% (should not apply), -100% after 65 minutes (limits trade duration)
tc16 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4975, 4987, 6172, 0, 0],
    [2, 4987, 5300, 4950, 5050, 6172, 0, 0],
    [3, 4975, 5000, 4940, 4962, 6172, 0, 0],  # ForceSell on ROI (roi=-1)
    [4, 4962, 4987, 4950, 4950, 6172, 0, 0],
    [5, 4950, 4975, 4925, 4950, 6172, 0, 0]],
    stop_loss=-0.10, roi={"0": 0.10, "65": -1}, profit_perc=-0.012,
    trades=[BTrade(sell_reason=SellType.ROI, open_tick=1, close_tick=3)]
)

# Test 17: Buy, hold for 120 mins, then forcesell using roi=-1
# Causes negative profit even though sell-reason is ROI.
# stop-loss: 10%, ROI: 10% (should not apply), -100% after 100 minutes (limits trade duration)
# Uses open as sell-rate (special case) - since the roi-time is a multiple of the timeframe.
tc17 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4975, 4987, 6172, 0, 0],
    [2, 4987, 5300, 4950, 5050, 6172, 0, 0],
    [3, 4980, 5000, 4940, 4962, 6172, 0, 0],  # ForceSell on ROI (roi=-1)
    [4, 4962, 4987, 4950, 4950, 6172, 0, 0],
    [5, 4950, 4975, 4925, 4950, 6172, 0, 0]],
    stop_loss=-0.10, roi={"0": 0.10, "120": -1}, profit_perc=-0.004,
    trades=[BTrade(sell_reason=SellType.ROI, open_tick=1, close_tick=3)]
)


# Test 18: Buy, hold for 120 mins, then drop ROI to 1%, causing a sell in candle 3.
# stop-loss: 10%, ROI: 10% (should not apply), -100% after 100 minutes (limits trade duration)
# uses open_rate as sell-price
tc18 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4975, 4987, 6172, 0, 0],
    [2, 4987, 5300, 4950, 5200, 6172, 0, 0],
    [3, 5200, 5220, 4940, 4962, 6172, 0, 0],  # Sell on ROI (sells on open)
    [4, 4962, 4987, 4950, 4950, 6172, 0, 0],
    [5, 4950, 4975, 4925, 4950, 6172, 0, 0]],
    stop_loss=-0.10, roi={"0": 0.10, "120": 0.01}, profit_perc=0.04,
    trades=[BTrade(sell_reason=SellType.ROI, open_tick=1, close_tick=3)]
)

# Test 19: Buy, hold for 119 mins, then drop ROI to 1%, causing a sell in candle 3.
# stop-loss: 10%, ROI: 10% (should not apply), -100% after 100 minutes (limits trade duration)
# uses calculated ROI (1%) as sell rate, otherwise identical to tc18
tc19 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4975, 4987, 6172, 0, 0],
    [2, 4987, 5300, 4950, 5200, 6172, 0, 0],
    [3, 5000, 5300, 4940, 4962, 6172, 0, 0],  # Sell on ROI
    [4, 4962, 4987, 4950, 4950, 6172, 0, 0],
    [5, 4550, 4975, 4550, 4950, 6172, 0, 0]],
    stop_loss=-0.10, roi={"0": 0.10, "120": 0.01}, profit_perc=0.01,
    trades=[BTrade(sell_reason=SellType.ROI, open_tick=1, close_tick=3)]
)

# Test 20: Buy, hold for 119 mins, then drop ROI to 1%, causing a sell in candle 3.
# stop-loss: 10%, ROI: 10% (should not apply), -100% after 100 minutes (limits trade duration)
# uses calculated ROI (1%) as sell rate, otherwise identical to tc18
tc20 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4975, 4987, 6172, 0, 0],
    [2, 4987, 5300, 4950, 5200, 6172, 0, 0],
    [3, 5200, 5300, 4940, 4962, 6172, 0, 0],  # Sell on ROI
    [4, 4962, 4987, 4950, 4950, 6172, 0, 0],
    [5, 4925, 4975, 4925, 4950, 6172, 0, 0]],
    stop_loss=-0.10, roi={"0": 0.10, "119": 0.01}, profit_perc=0.01,
    trades=[BTrade(sell_reason=SellType.ROI, open_tick=1, close_tick=3)]
)

# Test 21: trailing_stop ROI collision.
# Roi should trigger before Trailing stop - otherwise Trailing stop profits can be > ROI
# which cannot happen in reality
# stop-loss: 10%, ROI: 4%, Trailing stop adjusted at the sell candle
tc21 = BTContainer(data=[
    # D   O     H     L     C    V    B  S
    [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
    [1, 5000, 5100, 4950, 5100, 6172, 0, 0],
    [2, 5100, 5251, 4650, 5100, 6172, 0, 0],
    [3, 4850, 5050, 4650, 4750, 6172, 0, 0],
    [4, 4750, 4950, 4350, 4750, 6172, 0, 0]],
    stop_loss=-0.10, roi={"0": 0.04}, profit_perc=0.04, trailing_stop=True,
    trailing_only_offset_is_reached=True, trailing_stop_positive_offset=0.05,
    trailing_stop_positive=0.03,
    trades=[BTrade(sell_reason=SellType.ROI, open_tick=1, close_tick=2)]
)

# Test 22: trailing_stop Raises in candle 2 - but ROI applies at the same time.
# applying a positive trailing stop of 3% - ROI should apply before trailing stop.
# stop-loss: 10%, ROI: 4%, stoploss adjusted candle 2
tc22 = BTContainer(data=[
    # D   O     H     L     C    V    B  S
    [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
    [1, 5000, 5100, 4950, 5100, 6172, 0, 0],
    [2, 5100, 5251, 5100, 5100, 6172, 0, 0],
    [3, 4850, 5050, 4650, 4750, 6172, 0, 0],
    [4, 4750, 4950, 4350, 4750, 6172, 0, 0]],
    stop_loss=-0.10, roi={"0": 0.04}, profit_perc=0.04, trailing_stop=True,
    trailing_only_offset_is_reached=True, trailing_stop_positive_offset=0.05,
    trailing_stop_positive=0.03,
    trades=[BTrade(sell_reason=SellType.ROI, open_tick=1, close_tick=2)]
)

# Test 23: trailing_stop Raises in candle 2 (does not trigger)
# applying a positive trailing stop of 3% since stop_positive_offset is reached.
# ROI is changed after this to 4%, dropping ROI below trailing_stop_positive, causing a sell
# in the candle after the raised stoploss candle with ROI reason.
# Stoploss would trigger in this candle too, but it's no longer relevant.
# stop-loss: 10%, ROI: 4%, stoploss adjusted candle 2, ROI adjusted in candle 3 (causing the sell)
tc23 = BTContainer(data=[
    # D   O     H     L     C    V    B  S
    [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
    [1, 5000, 5100, 4950, 5100, 6172, 0, 0],
    [2, 5100, 5251, 5100, 5100, 6172, 0, 0],
    [3, 4850, 5251, 4650, 4750, 6172, 0, 0],
    [4, 4750, 4950, 4350, 4750, 6172, 0, 0]],
    stop_loss=-0.10, roi={"0": 0.1, "119": 0.03}, profit_perc=0.03, trailing_stop=True,
    trailing_only_offset_is_reached=True, trailing_stop_positive_offset=0.05,
    trailing_stop_positive=0.03,
    trades=[BTrade(sell_reason=SellType.ROI, open_tick=1, close_tick=3)]
)

# Test 24: Sell with signal sell in candle 3 (stoploss also triggers on this candle)
# Stoploss at 1%.
# Stoploss wins over Sell-signal (because sell-signal is acted on in the next candle)
tc24 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4975, 4987, 6172, 0, 0],  # enter trade (signal on last candle)
    [2, 4987, 5012, 4986, 4986, 6172, 0, 0],
    [3, 5010, 5010, 4855, 5010, 6172, 0, 1],  # Triggers stoploss + sellsignal
    [4, 5010, 5010, 4977, 4995, 6172, 0, 0],
    [5, 4995, 4995, 4950, 4950, 6172, 0, 0]],
    stop_loss=-0.01, roi={"0": 1}, profit_perc=-0.01, use_sell_signal=True,
    trades=[BTrade(sell_reason=SellType.STOP_LOSS, open_tick=1, close_tick=3)]
)

# Test 25: Sell with signal sell in candle 3 (stoploss also triggers on this candle)
# Stoploss at 1%.
# Sell-signal wins over stoploss
tc25 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4975, 4987, 6172, 0, 0],  # enter trade (signal on last candle)
    [2, 4987, 5012, 4986, 4986, 6172, 0, 0],
    [3, 5010, 5010, 4986, 5010, 6172, 0, 1],
    [4, 5010, 5010, 4855, 4995, 6172, 0, 0],  # Triggers stoploss + sellsignal acted on
    [5, 4995, 4995, 4950, 4950, 6172, 0, 0]],
    stop_loss=-0.01, roi={"0": 1}, profit_perc=0.002, use_sell_signal=True,
    trades=[BTrade(sell_reason=SellType.SELL_SIGNAL, open_tick=1, close_tick=4)]
)

# Test 26: Sell with signal sell in candle 3 (ROI at signal candle)
# Stoploss at 10% (irrelevant), ROI at 5% (will trigger)
# Sell-signal wins over stoploss
tc26 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4975, 4987, 6172, 0, 0],  # enter trade (signal on last candle)
    [2, 4987, 5012, 4986, 4986, 6172, 0, 0],
    [3, 5010, 5251, 4986, 5010, 6172, 0, 1],  # Triggers ROI, sell-signal
    [4, 5010, 5010, 4855, 4995, 6172, 0, 0],
    [5, 4995, 4995, 4950, 4950, 6172, 0, 0]],
    stop_loss=-0.10, roi={"0": 0.05}, profit_perc=0.05, use_sell_signal=True,
    trades=[BTrade(sell_reason=SellType.ROI, open_tick=1, close_tick=3)]
)

# Test 27: Sell with signal sell in candle 3 (ROI at signal candle)
# Stoploss at 10% (irrelevant), ROI at 5% (will trigger) - Wins over Sell-signal
# TODO: figure out if sell-signal should win over ROI
# Sell-signal wins over stoploss
tc27 = BTContainer(data=[
    # D  O     H     L     C     V    B  S
    [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
    [1, 5000, 5025, 4975, 4987, 6172, 0, 0],  # enter trade (signal on last candle)
    [2, 4987, 5012, 4986, 4986, 6172, 0, 0],
    [3, 5010, 5012, 4986, 5010, 6172, 0, 1],  # sell-signal
    [4, 5010, 5251, 4855, 4995, 6172, 0, 0],  # Triggers ROI, sell-signal acted on
    [5, 4995, 4995, 4950, 4950, 6172, 0, 0]],
    stop_loss=-0.10, roi={"0": 0.05}, profit_perc=0.05, use_sell_signal=True,
    trades=[BTrade(sell_reason=SellType.ROI, open_tick=1, close_tick=4)]
)

# Test 28: trailing_stop should raise so candle 3 causes a stoploss
# Same case than tc11 - but candle 3 "gaps down" - the stoploss will be above the candle,
# therefore "open" will be used
# stop-loss: 10%, ROI: 10% (should not apply), stoploss adjusted candle 2
tc28 = BTContainer(data=[
    # D   O     H     L     C    V    B  S
    [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
    [1, 5000, 5100, 4950, 5100, 6172, 0, 0],
    [2, 5100, 5251, 5100, 5100, 6172, 0, 0],
    [3, 4850, 5050, 4650, 4750, 6172, 0, 0],
    [4, 4750, 4950, 4350, 4750, 6172, 0, 0]],
    stop_loss=-0.10, roi={"0": 0.10}, profit_perc=-0.03, trailing_stop=True,
    trailing_only_offset_is_reached=True, trailing_stop_positive_offset=0.05,
    trailing_stop_positive=0.03,
    trades=[BTrade(sell_reason=SellType.TRAILING_STOP_LOSS, open_tick=1, close_tick=3)]
)

# Test 29: trailing_stop should be triggered by low of next candle, without adjusting stoploss using
# high of stoploss candle.
# stop-loss: 10%, ROI: 10% (should not apply)
tc29 = BTContainer(data=[
    # D   O     H     L     C    V    B  S
    [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
    [1, 5000, 5050, 5000, 5000, 6172, 0, 0],    # enter trade (signal on last candle)
    [2, 4900, 5250, 4500, 5100, 6172, 0, 0],    # Triggers trailing-stoploss
    [3, 5100, 5100, 4650, 4750, 6172, 0, 0],
    [4, 4750, 4950, 4350, 4750, 6172, 0, 0]],
    stop_loss=-0.10, roi={"0": 0.10}, profit_perc=-0.02, trailing_stop=True,
    trailing_stop_positive=0.03,
    trades=[BTrade(sell_reason=SellType.TRAILING_STOP_LOSS, open_tick=1, close_tick=2)]
)

# Test 30: trailing_stop should be triggered immediately on trade open candle.
# stop-loss: 10%, ROI: 10% (should not apply)
tc30 = BTContainer(data=[
    # D   O     H     L     C    V    B  S
    [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
    [1, 5000, 5500, 4900, 4900, 6172, 0, 0],    # enter trade (signal on last candle) and stop
    [2, 4900, 5250, 4500, 5100, 6172, 0, 0],
    [3, 5100, 5100, 4650, 4750, 6172, 0, 0],
    [4, 4750, 4950, 4350, 4750, 6172, 0, 0]],
    stop_loss=-0.10, roi={"0": 0.10}, profit_perc=-0.01, trailing_stop=True,
    trailing_stop_positive=0.01,
    trades=[BTrade(sell_reason=SellType.TRAILING_STOP_LOSS, open_tick=1, close_tick=1)]
)

# Test 31: trailing_stop should be triggered immediately on trade open candle.
# stop-loss: 10%, ROI: 10% (should not apply)
tc31 = BTContainer(data=[
    # D   O     H     L     C    V    B  S
    [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
    [1, 5000, 5500, 4900, 4900, 6172, 0, 0],    # enter trade (signal on last candle) and stop
    [2, 4900, 5250, 4500, 5100, 6172, 0, 0],
    [3, 5100, 5100, 4650, 4750, 6172, 0, 0],
    [4, 4750, 4950, 4350, 4750, 6172, 0, 0]],
    stop_loss=-0.10, roi={"0": 0.10}, profit_perc=0.01, trailing_stop=True,
    trailing_only_offset_is_reached=True, trailing_stop_positive_offset=0.02,
    trailing_stop_positive=0.01,
    trades=[BTrade(sell_reason=SellType.TRAILING_STOP_LOSS, open_tick=1, close_tick=1)]
)

# Test 32: trailing_stop should be triggered immediately on trade open candle.
# stop-loss: 1%, ROI: 10% (should not apply)
tc32 = BTContainer(data=[
    # D   O     H     L     C    V    B  S
    [0, 5000, 5050, 4950, 5000, 6172, 1, 0],
    [1, 5000, 5500, 4951, 5000, 6172, 0, 0],    # enter trade (signal on last candle) and stop
    [2, 4900, 5250, 4500, 5100, 6172, 0, 0],
    [3, 5100, 5100, 4650, 4750, 6172, 0, 0],
    [4, 4750, 4950, 4350, 4750, 6172, 0, 0]],
    stop_loss=-0.01, roi={"0": 0.10}, profit_perc=-0.01, trailing_stop=True,
    trailing_only_offset_is_reached=True, trailing_stop_positive_offset=0.02,
    trailing_stop_positive=0.01, use_custom_stoploss=True,
    trades=[BTrade(sell_reason=SellType.TRAILING_STOP_LOSS, open_tick=1, close_tick=1)]
)

# Test 33: trailing_stop should be triggered immediately on trade open candle.
# stop-loss: 1%, ROI: 10% (should not apply)
tc33 = BTContainer(data=[
    # D   O     H     L     C    V    B  S   BT
    [0, 5000, 5050, 4950, 5000, 6172, 1, 0, 'buy_signal_01'],
    [1, 5000, 5500, 4951, 5000, 6172, 0, 0, None],    # enter trade (signal on last candle) and stop
    [2, 4900, 5250, 4500, 5100, 6172, 0, 0, None],
    [3, 5100, 5100, 4650, 4750, 6172, 0, 0, None],
    [4, 4750, 4950, 4350, 4750, 6172, 0, 0, None]],
    stop_loss=-0.01, roi={"0": 0.10}, profit_perc=-0.01, trailing_stop=True,
    trailing_only_offset_is_reached=True, trailing_stop_positive_offset=0.02,
    trailing_stop_positive=0.01, use_custom_stoploss=True,
    trades=[BTrade(
        sell_reason=SellType.TRAILING_STOP_LOSS,
        open_tick=1,
        close_tick=1,
        buy_tag='buy_signal_01'
    )]
)

TESTS = [
    tc0,
    tc1,
    tc2,
    tc3,
    tc4,
    tc5,
    tc6,
    tc7,
    tc8,
    tc9,
    tc10,
    tc11,
    tc12,
    tc13,
    tc14,
    tc15,
    tc16,
    tc17,
    tc18,
    tc19,
    tc20,
    tc21,
    tc22,
    tc23,
    tc24,
    tc25,
    tc26,
    tc27,
    tc28,
    tc29,
    tc30,
    tc31,
    tc32,
    tc33,
]


@pytest.mark.parametrize("data", TESTS)
def test_backtest_results(default_conf, fee, mocker, caplog, data) -> None:
    """
    run functional tests
    """
    default_conf["stoploss"] = data.stop_loss
    default_conf["minimal_roi"] = data.roi
    default_conf["timeframe"] = tests_timeframe
    default_conf["trailing_stop"] = data.trailing_stop
    default_conf["trailing_only_offset_is_reached"] = data.trailing_only_offset_is_reached
    # Only add this to configuration If it's necessary
    if data.trailing_stop_positive is not None:
        default_conf["trailing_stop_positive"] = data.trailing_stop_positive
    default_conf["trailing_stop_positive_offset"] = data.trailing_stop_positive_offset
    default_conf["use_sell_signal"] = data.use_sell_signal

    mocker.patch("freqtrade.exchange.Exchange.get_fee", return_value=0.0)
    mocker.patch("freqtrade.exchange.Exchange.get_min_pair_stake_amount", return_value=0.00001)
    patch_exchange(mocker)
    frame = _build_backtest_dataframe(data.data)
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.required_startup = 0
    backtesting.strategy.advise_buy = lambda a, m: frame
    backtesting.strategy.advise_sell = lambda a, m: frame
    backtesting.strategy.use_custom_stoploss = data.use_custom_stoploss
    caplog.set_level(logging.DEBUG)

    pair = "UNITTEST/BTC"
    # Dummy data as we mock the analyze functions
    data_processed = {pair: frame.copy()}
    min_date, max_date = get_timerange({pair: frame})
    result = backtesting.backtest(
        processed=data_processed,
        start_date=min_date,
        end_date=max_date,
        max_open_trades=10,
    )

    results = result['results']
    assert len(results) == len(data.trades)
    assert round(results["profit_ratio"].sum(), 3) == round(data.profit_perc, 3)

    for c, trade in enumerate(data.trades):
        res = results.iloc[c]
        assert res.sell_reason == trade.sell_reason.value
        assert res.buy_tag == trade.buy_tag
        assert res.open_date == _get_frame_time_from_offset(trade.open_tick)
        assert res.close_date == _get_frame_time_from_offset(trade.close_tick)
