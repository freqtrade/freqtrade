import numpy as np
from skopt.space import Integer


class SKDecimal(Integer):

    def __init__(self, low, high, decimals=3, prior="uniform", base=10, transform=None,
                 name=None, dtype=np.int64):
        self.decimals = decimals
        _low = int(low * pow(10, self.decimals))
        _high = int(high * pow(10, self.decimals))
        # trunc to precision to avoid points out of space
        self.low_orig = round(_low * pow(0.1, self.decimals), self.decimals)
        self.high_orig = round(_high * pow(0.1, self.decimals), self.decimals)

        super().__init__(_low, _high, prior, base, transform, name, dtype)

    def __repr__(self):
        return "Decimal(low={}, high={}, decimals={}, prior='{}', transform='{}')".format(
            self.low_orig, self.high_orig, self.decimals, self.prior, self.transform_)

    def __contains__(self, point):
        if isinstance(point, list):
            point = np.array(point)
        return self.low_orig <= point <= self.high_orig

    def transform(self, Xt):
        aa = [int(x * pow(10, self.decimals)) for x in Xt]
        return super().transform(aa)

    def inverse_transform(self, Xt):
        res = super().inverse_transform(Xt)
        return [round(x * pow(0.1, self.decimals), self.decimals) for x in res]
