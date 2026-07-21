# freqtrade previously vendored a copy of qtpylib's indicators
# (https://github.com/ranaroussi/qtpylib, Apache-2.0). That copy is identical to
# the one shipped by technical, so we re-export from there to avoid maintaining a
# duplicate. Existing imports such as
# `from freqtrade.vendor.qtpylib.indicators import crossed_above` keep working,
# but emit a FutureWarning pointing at technical.

import warnings

from technical.qtpylib import *  # noqa: F403


warnings.warn(
    "'freqtrade.vendor.qtpylib.indicators' is deprecated and will be removed in a future "
    "release. Import from technical instead - either the module "
    "(`from technical import qtpylib`) or a single "
    "function (`from technical.qtpylib import crossed_above`).",
    FutureWarning,
    stacklevel=2,
)
