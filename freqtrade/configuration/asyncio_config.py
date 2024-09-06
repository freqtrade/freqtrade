import sys


def asyncio_setup() -> None:  # pragma: no cover
    # Set eventloop for win32 setups

    if sys.platform == "win32":
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
