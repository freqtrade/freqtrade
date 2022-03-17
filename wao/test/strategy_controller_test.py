from wao.strategy_controller import StrategyController
import time

brain = 'Freq_Strategy000'
mode = "test"
coin = "ETH"
sell_reason = 'sell_signal'
current_time = 'todo_current_time_format'#get now

controller = StrategyController(brain)


controller.on_buy_signal(current_time, mode, coin, brain)
time.sleep(1)

current_time = 'todo_current_time_format'#get now

controller.on_sell_signal(sell_reason, current_time, mode, coin, brain)
