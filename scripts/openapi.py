#!/usr/bin/env python3
import argparse
import json

from freqtrade.rpc.api_server import ApiServer


def main(indent: int):
    openapi_dict = ApiServer({"api_server": {}}).get_open_api_json()
    print(json.dumps(openapi_dict, indent=indent), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simply outputs the OpenAPI JSON of the REST API")
    parser.add_argument("-i", "--indent", type=int, default=2)

    args = parser.parse_args()

    main(args.indent)
