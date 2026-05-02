from __future__ import annotations

import os
import sys


def main() -> None:
    _force_threaded_dns_resolver()

    from freqtrade.main import main as freqtrade_main

    freqtrade_main(sys.argv[1:])


def _force_threaded_dns_resolver() -> None:
    enabled = os.environ.get("BOT_FACTORY_FORCE_THREADED_DNS", "1").lower()
    if enabled in {"0", "false", "no"}:
        return

    try:
        import aiohttp.connector as connector
        import aiohttp.resolver as resolver
    except Exception:
        return

    resolver.DefaultResolver = resolver.ThreadedResolver
    connector.DefaultResolver = resolver.ThreadedResolver


if __name__ == "__main__":
    main()
