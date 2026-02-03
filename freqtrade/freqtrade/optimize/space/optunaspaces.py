from collections.abc import Sequence
from typing import Any, Protocol

from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution


class DimensionProtocol(Protocol):
    name: str


class ft_CategoricalDistribution(CategoricalDistribution):
    def __init__(
        self,
        categories: Sequence[Any],
        name: str,
        **kwargs,
    ):
        self.name = name
        self.categories = categories
        # if len(categories) <= 1:
        #     raise Exception(f"need at least 2 categories for {name}")
        return super().__init__(categories)

    def __repr__(self):
        return f"CategoricalDistribution({self.categories})"


class ft_IntDistribution(IntDistribution):
    def __init__(
        self,
        low: int | float,
        high: int | float,
        name: str,
        **kwargs,
    ):
        self.name = name
        self.low = int(low)
        self.high = int(high)
        return super().__init__(self.low, self.high, **kwargs)

    def __repr__(self):
        return f"IntDistribution(low={self.low}, high={self.high})"


class ft_FloatDistribution(FloatDistribution):
    def __init__(
        self,
        low: float,
        high: float,
        name: str,
        **kwargs,
    ):
        self.name = name
        self.low = low
        self.high = high
        return super().__init__(low, high, **kwargs)

    def __repr__(self):
        return f"FloatDistribution(low={self.low}, high={self.high}, step={self.step})"
