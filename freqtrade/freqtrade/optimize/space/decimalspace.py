from optuna.distributions import FloatDistribution


class SKDecimal(FloatDistribution):
    def __init__(
        self,
        low: float,
        high: float,
        *,
        step: float | None = None,
        decimals: int | None = None,
        name=None,
    ):
        """
        FloatDistribution with a fixed step size.
        Only one of step or decimals can be set.
        :param low: lower bound
        :param high: upper bound
        :param step: step size (e.g. 0.001)
        :param decimals: number of decimal places to round to (e.g. 3)
        :param name: name of the distribution
        """
        if decimals is not None and step is not None:
            raise ValueError("You can only set one of decimals or step")
        if decimals is None and step is None:
            raise ValueError("You must set one of decimals or step")
        # Convert decimals to step
        self.step = step or (1 / 10**decimals if decimals else 1)
        self.name = name

        super().__init__(
            low=round(low, decimals) if decimals else low,
            high=round(high, decimals) if decimals else high,
            step=self.step,
        )
