from pathlib import Path

import rapidjson

from freqtrade.configuration.config_schema import CONF_SCHEMA


schema_filename = Path(Path(__file__).parent.parent / "schema.json")
with schema_filename.open("w") as f:
    rapidjson.dump(CONF_SCHEMA, f, indent=2)
