from freqtrade.resolvers.freqaimodel_resolver import FreqaiModelResolver


class CustomModel:
    """
    A bridge between the user defined IFreqaiModel class
    and the strategy.
    """

    def __init__(self, config):

        self.bridge = FreqaiModelResolver.load_freqaimodel(config)
