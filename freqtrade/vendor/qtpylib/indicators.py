# freqtrade previously vendored a copy of qtpylib's indicators
# (https://github.com/ranaroussi/qtpylib, Apache-2.0). That copy is identical to
# the one shipped by technical, so we can re-export to avoid maintaining a duplicate.
# Existing imports such as
# `from freqtrade.vendor.qtpylib.indicators import crossed_above` keep working.

from technical.vendor.qtpylib.indicators import *  # noqa: F403
