import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def start_trading(args: Dict[str, Any]) -> int:
    """
    Main entry point for trading mode
    """
    # Import here to avoid loading worker module when it's not used
    from freqtrade.worker import Worker

    # Create and run worker
    worker = None
    try:
        worker = Worker(args)
        worker.run()
    except Exception as e:
        logger.error(str(e))
        logger.exception("Fatal exception!")
    except KeyboardInterrupt:
        logger.info('SIGINT received, aborting ...')
    finally:
        if worker:
            logger.info("worker found ... calling exit")
            worker.exit()
    return 0


def start_webserver(args: Dict[str, Any]) -> None:
    """
    Main entry point for webserver mode
    """
    from freqtrade.rpc.api_server import ApiServer
    from freqtrade.configuration import Configuration
    from freqtrade.enums import RunMode

    # Initialize configuration
    config = Configuration(args, RunMode.WEBSERVER).get_config()
    ApiServer(config, standalone=True)
