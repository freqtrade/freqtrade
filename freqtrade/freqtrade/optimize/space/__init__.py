from .decimalspace import SKDecimal
from .optunaspaces import (
    DimensionProtocol,
    ft_CategoricalDistribution,
    ft_FloatDistribution,
    ft_IntDistribution,
)


# Alias for the distribution classes
Dimension = DimensionProtocol
Categorical = ft_CategoricalDistribution
Integer = ft_IntDistribution
Real = ft_FloatDistribution

__all__ = ["Categorical", "Dimension", "Integer", "Real", "SKDecimal"]
