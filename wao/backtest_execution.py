

class BacktestExecution:

    def __init__(self, brain, coin, type, timeout_hours, timestamp=None, is_finished=False):
        self.brain = brain
        self.coin = coin
        self.type = type
        self.timeout_hours = timeout_hours
        self.timestamp = timestamp
        self.is_finished = is_finished

    def finish(self):
        self.is_finished = True
