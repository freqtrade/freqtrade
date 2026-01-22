import ccxt

# Ensure we don't patch twice if not needed, though safe to check
if "icicibreeze" not in ccxt.exchanges:
    ccxt.exchanges.append("icicibreeze")

if not hasattr(ccxt, "icicibreeze"):

    class icicibreeze(ccxt.Exchange):
        def describe(self):
            return self.deep_extend(
                super().describe(),
                {
                    "id": "icicibreeze",
                    "name": "IciciBreeze",
                    "has": {
                        "fetchOHLCV": True,
                        "fetchTicker": True,
                        "createOrder": True,
                        "cancelOrder": True,
                        "fetchOrder": True,
                        "fetchOpenOrders": True,
                        "fetchClosedOrders": True,
                        "fetchMyTrades": True,
                        "fetchBalance": True,
                    },
                    "timeframes": {"1m": "1minute", "5m": "5minute", "1d": "1day"},
                },
            )

    setattr(ccxt, "icicibreeze", icicibreeze)
