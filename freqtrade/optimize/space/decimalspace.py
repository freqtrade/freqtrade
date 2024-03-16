import numpy as np
from skopt.space import Integer


class SKDecimal(Integer):

    def __init__(self, low, high, decimals=3, prior="uniform", base=10, transform=None,
                 name=None, dtype=np.int64):
        self.decimals = decimals

        self.pow_dot_one = pow(0.1, self.decimals)
        self.pow_ten = pow(10, self.decimals)

        _low = int(low * self.pow_ten)
        _high = int(high * self.pow_ten)
        # trunc to precision to avoid points out of space
        self.low_orig = round(_low * self.pow_dot_one, self.decimals)
        self.high_orig = round(_high * self.pow_dot_one, self.decimals)

        super().__init__(_low, _high, prior, base, transform, name, dtype)

    def __repr__(self):
        return (
            f"Decimal(low={self.low_orig}, high={self.high_orig}, decimals={self.decimals}, "
            f"prior='{self.prior}', transform='{self.transform_}')"
        )

    def __contains__(self, point):
        if isinstance(point, list):
            point = np.array(point)
        return self.low_orig <= point <= self.high_orig

    def transform(self, Xt):
        return super().transform([int(v * self.pow_ten) for v in Xt])

    def inverse_transform(self, Xt):
        res = super().inverse_transform(Xt)
        # equivalent to [round(x * pow(0.1, self.decimals), self.decimals) for x in res]
        return [int(v) / self.pow_ten for v in res]
