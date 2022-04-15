from wao.wao_strategy_controller import WAOStrategyController
import time
import datetime
brain = 'Freq_Strategy000'
mode = "test"
coin = "ETH"
sell_reason = 'sell_signal'
current_time = str(datetime.datetime.now()).replace('.', '+')

controller = WAOStrategyController(brain)


controller.on_buy_signal(current_time, mode, coin, brain)

time.sleep(10)

current_time = str(datetime.datetime.now()).replace('.', '+')

controller.on_sell_signal(sell_reason, current_time, mode, coin, brain)
