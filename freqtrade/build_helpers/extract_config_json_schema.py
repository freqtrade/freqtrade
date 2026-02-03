"""Script to extract the configuration json schema from config_schema.py file."""

from pathlib import Path

import rapidjson


def extract_config_json_schema():
    try:
        # Try to import from the installed package
        from freqtrade.config_schema import CONF_SCHEMA
    except ImportError:
        # If freqtrade is not installed, add the parent directory to sys.path
        # to import directly from the source
        import sys

        script_dir = Path(__file__).parent
        freqtrade_dir = script_dir.parent
        sys.path.insert(0, str(freqtrade_dir))

        # Now try to import from the source
        from freqtrade.config_schema import CONF_SCHEMA

    schema_filename = Path(__file__).parent / "schema.json"
    with schema_filename.open("w") as f:
        rapidjson.dump(CONF_SCHEMA, f, indent=2)


if __name__ == "__main__":
    extract_config_json_schema()
