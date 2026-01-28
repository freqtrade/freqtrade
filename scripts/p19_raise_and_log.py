#!/usr/bin/env python3
import logging
import sys

# Configure basic logging to stderr to capture output
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("p19_test")


def trigger_error():
    try:
        raise RuntimeError("p19_intentional_error_for_traceback_verification")
    except Exception:
        logger.exception(
            "P19: Intentional exception triggered for verification payload={'test': 'true'}"
        )


if __name__ == "__main__":
    print("Triggering exception...")
    trigger_error()
    print("Done.")
