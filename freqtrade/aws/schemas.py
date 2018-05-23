# defines the schema to submit a new strategy to the system
__SUBMIT_STRATEGY_SCHEMA__ = {
  "$id": "http://example.com/example.json",
  "type": "object",
  "definitions": {},
  "$schema": "http://json-schema.org/draft-07/schema#",
  "properties": {
    "user": {
      "$id": "/properties/user",
      "type": "string",
      "title": "The User Schema ",
      "default": "",
      "examples": [
        "GCU4LW2XXZW3A3FM2XZJTEJHNWHTWDKY2DIJLCZJ5ULVZ4K7LZ7D23TG"
      ]
    },
    "description": {
      "$id": "/properties/description",
      "type": "string",
      "title": "The Description Schema ",
      "default": "",
      "examples": [
        "simple test strategy"
      ]
    },
    "exchange": {
      "$id": "/properties/exchange",
      "type": "object",
      "properties": {
        "name": {
          "$id": "/properties/exchange/properties/name",
          "type": "string",
          "title": "The Name Schema ",
          "default": "",
          "examples": [
            "binance"
          ]
        },
        "stake": {
          "$id": "/properties/exchange/properties/stake",
          "type": "string",
          "title": "The Stake Schema ",
          "default": "",
          "examples": [
            "usdt"
          ]
        },
        "pairs": {
          "$id": "/properties/exchange/properties/pairs",
          "type": "array",
          "items": {
            "$id": "/properties/exchange/properties/pairs/items",
            "type": "string",
            "title": "The 0th Schema ",
            "default": "",
            "examples": [
              "btc/usdt"
            ]
          }
        }
      }
    },
    "name": {
      "$id": "/properties/name",
      "type": "string",
      "title": "The Name Schema ",
      "default": "",
      "examples": [
        "MyFancyTestStrategy"
      ]
    },
    "content": {
      "$id": "/properties/content",
      "type": "string",
      "title": "The Content Schema ",
      "default": "",
      "examples": [
        "IyAtLS0gRG8gbm90IHJlbW92ZSB0aGVzZSBsaWJzIC0tLQpmcm9tIGZyZXF0cmFkZS5zdHJhdGVneS5pbnRlcmZhY2UgaW1wb3J0IElTdHJhdGVneQpmcm9tIHR5cGluZyBpbXBvcnQgRGljdCwgTGlzdApmcm9tIGh5cGVyb3B0IGltcG9ydCBocApmcm9tIGZ1bmN0b29scyBpbXBvcnQgcmVkdWNlCmZyb20gcGFuZGFzIGltcG9ydCBEYXRhRnJhbWUKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQoKaW1wb3J0IHRhbGliLmFic3RyYWN0IGFzIHRhCmltcG9ydCBmcmVxdHJhZGUudmVuZG9yLnF0cHlsaWIuaW5kaWNhdG9ycyBhcyBxdHB5bGliCgpjbGFzcyBNeUZhbmN5VGVzdFN0cmF0ZWd5KElTdHJhdGVneSk6CiAgICBtaW5pbWFsX3JvaSA9IHsKICAgICAgICAiMCI6IDAuNQogICAgfQogICAgc3RvcGxvc3MgPSAtMC4yCiAgICB0aWNrZXJfaW50ZXJ2YWwgPSAnNW0nCgogICAgZGVmIHBvcHVsYXRlX2luZGljYXRvcnMoc2VsZiwgZGF0YWZyYW1lOiBEYXRhRnJhbWUpIC0-IERhdGFGcmFtZToKICAgICAgICBtYWNkID0gdGEuTUFDRChkYXRhZnJhbWUpCiAgICAgICAgZGF0YWZyYW1lWydtYVNob3J0J10gPSB0YS5FTUEoZGF0YWZyYW1lLCB0aW1lcGVyaW9kPTgpCiAgICAgICAgZGF0YWZyYW1lWydtYU1lZGl1bSddID0gdGEuRU1BKGRhdGFmcmFtZSwgdGltZXBlcmlvZD0yMSkKICAgICAgICByZXR1cm4gZGF0YWZyYW1lCgogICAgZGVmIHBvcHVsYXRlX2J1eV90cmVuZChzZWxmLCBkYXRhZnJhbWU6IERhdGFGcmFtZSkgLT4gRGF0YUZyYW1lOgogICAgICAgIGRhdGFmcmFtZS5sb2NbCiAgICAgICAgICAgICgKICAgICAgICAgICAgICAgIHF0cHlsaWIuY3Jvc3NlZF9hYm92ZShkYXRhZnJhbWVbJ21hU2hvcnQnXSwgZGF0YWZyYW1lWydtYU1lZGl1bSddKQogICAgICAgICAgICApLAogICAgICAgICAgICAnYnV5J10gPSAxCgogICAgICAgIHJldHVybiBkYXRhZnJhbWUKCiAgICBkZWYgcG9wdWxhdGVfc2VsbF90cmVuZChzZWxmLCBkYXRhZnJhbWU6IERhdGFGcmFtZSkgLT4gRGF0YUZyYW1lOgogICAgICAgIGRhdGFmcmFtZS5sb2NbCiAgICAgICAgICAgICgKICAgICAgICAgICAgICAgIHF0cHlsaWIuY3Jvc3NlZF9hYm92ZShkYXRhZnJhbWVbJ21hTWVkaXVtJ10sIGRhdGFmcmFtZVsnbWFTaG9ydCddKQogICAgICAgICAgICApLAogICAgICAgICAgICAnc2VsbCddID0gMQogICAgICAgIHJldHVybiBkYXRhZnJhbWUKCgogICAgICAgIA=="
      ]
    },
    "public": {
      "$id": "/properties/public",
      "type": "boolean",
      "title": "The Public Schema ",
      "default": False,
      "examples": [
        False
      ]
    }
  }
}
