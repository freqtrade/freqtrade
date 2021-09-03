from typing import Any, Dict

from freqtrade.enums import RunMode


def start_webserver(args: Dict[str, Any]) -> None:
    """
    Main entry point for webserver mode
    """
    from freqtrade.configuration import Configuration
    from freqtrade.rpc.api_server import ApiServer

    # Initialize configuration
    config = Configuration(args, RunMode.WEBSERVER).get_config()
    api_server = ApiServer(config, standalone=True)
    api_server.start_api()
