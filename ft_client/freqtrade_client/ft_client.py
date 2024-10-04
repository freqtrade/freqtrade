import argparse
import inspect
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import rapidjson

from freqtrade_client import __version__
from freqtrade_client.ft_rest_client import FtRestClient


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ft_rest_client")


def add_arguments(args: Any = None):
    parser = argparse.ArgumentParser(
        prog="freqtrade-client",
        description="Client for the freqtrade REST API",
    )
    parser.add_argument(
        "command", help="Positional argument defining the command to execute.", nargs="?"
    )
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--show",
        help="Show possible methods with this client",
        dest="show",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-c",
        "--config",
        help="Specify configuration file (default: %(default)s). ",
        dest="config",
        type=str,
        metavar="PATH",
        default="config.json",
    )

    parser.add_argument(
        "command_arguments",
        help="Positional arguments for the parameters for [command]",
        nargs="*",
        default=[],
    )

    pargs = parser.parse_args(args)
    return vars(pargs)


def load_config(configfile):
    file = Path(configfile)
    if file.is_file():
        with file.open("r") as f:
            config = rapidjson.load(
                f, parse_mode=rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS
            )
        return config
    else:
        logger.warning(f"Could not load config file {file}.")
        sys.exit(1)


def print_commands():
    # Print dynamic help for the different commands using the commands doc-strings
    client = FtRestClient(None)
    print("Possible commands:\n")
    for x, _ in inspect.getmembers(client):
        if not x.startswith("_"):
            doc = re.sub(":return:.*", "", getattr(client, x).__doc__, flags=re.MULTILINE).rstrip()
            print(f"{x}\n\t{doc}\n")


def main_exec(parsed: dict[str, Any]):
    if parsed.get("show"):
        print_commands()
        sys.exit()

    config = load_config(parsed["config"])
    url = config.get("api_server", {}).get("listen_ip_address", "127.0.0.1")
    port = config.get("api_server", {}).get("listen_port", "8080")
    username = config.get("api_server", {}).get("username")
    password = config.get("api_server", {}).get("password")

    server_url = f"http://{url}:{port}"
    client = FtRestClient(server_url, username, password)

    m = [x for x, y in inspect.getmembers(client) if not x.startswith("_")]
    command = parsed["command"]
    if command not in m:
        logger.error(f"Command {command} not defined")
        print_commands()
        return

    # Split arguments with = into key/value pairs
    kwargs = {x.split("=")[0]: x.split("=")[1] for x in parsed["command_arguments"] if "=" in x}
    args = [x for x in parsed["command_arguments"] if "=" not in x]
    try:
        res = getattr(client, command)(*args, **kwargs)
        print(json.dumps(res))
    except TypeError as e:
        logger.error(f"Error executing command {command}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal Error executing command {command}: {e}")
        sys.exit(1)


def main():
    """
    Main entry point for the client
    """
    args = add_arguments()
    main_exec(args)
