"""Script to extract the configuration json schema from config_schema.py file."""

from pathlib import Path

import rapidjson

from freqtrade.configuration.config_schema import CONF_SCHEMA


def extract_config_json_schema():
    schema_filename = Path(__file__).parent / "schema.json"
    with schema_filename.open("w") as f:
        rapidjson.dump(CONF_SCHEMA, f, indent=2)


if __name__ == "__main__":
    extract_config_json_schema()
