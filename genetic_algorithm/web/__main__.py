"""
Entry point for the GA web dashboard.

Usage:
    python -m genetic_algorithm.web                           # defaults
    python -m genetic_algorithm.web --host 0.0.0.0 --port 8501  # remote access
    python -m genetic_algorithm.web --config path/to/config.yaml  # from GA config
"""

import argparse
import sys

from genetic_algorithm.web.config import WebConfig
from genetic_algorithm.web.server import start_server


def main():
    parser = argparse.ArgumentParser(description="GA Evolution Web Dashboard")
    parser.add_argument("--host", type=str, default=None, help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Bind port (default: 8501)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to GA config YAML (reads web_dashboard section)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    args = parser.parse_args()

    web_config = WebConfig()

    # Load from GA config if provided
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            ga_config = yaml.safe_load(f)
        web_section = ga_config.get('web_dashboard', {})
        if web_section:
            web_config = WebConfig.from_dict(web_section)

    # CLI overrides
    if args.host:
        web_config.host = args.host
    if args.port:
        web_config.port = args.port
    if args.no_browser:
        web_config.open_browser = False

    print(f"Starting GA Dashboard at http://{web_config.host}:{web_config.port}")
    start_server(web_config=web_config)


if __name__ == "__main__":
    main()
