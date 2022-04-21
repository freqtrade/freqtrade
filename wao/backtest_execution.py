

class BacktestExecution:

    def __init__(self, brain, coin, type, timeout_hours, timestamp=None):
        self.brain = brain
        self.coin = coin
        self.type = type
        self.timeout_hours = timeout_hours
        self.timestamp = timestamp

    def __str__(self):
        return f'(brain={self.brain}, coin={self.coin}, type={self.type}, timeout_hours={self.timeout_hours}, timestamp={self.timestamp})\n'
