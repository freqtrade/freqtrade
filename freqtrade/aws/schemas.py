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
            "title": "the associated Isaac user",
            "default": "",
            "examples": [
                "GCU4LW2XXZW3A3FM2XZJTEJHNWHTWDKY2DIJLCZJ5ULVZ4K7LZ7D23TG"
            ]
        },
        "description": {
            "$id": "/properties/description",
            "type": "string",
            "title": "a brief description",
            "default": "",
            "examples": [
                "simple test strategy"
            ]
        },
        "name": {
            "$id": "/properties/name",
            "type": "string",
            "title": "the name of your strategy",
            "default": "",
            "examples": [
                "TestStrategy"
            ]
        },
        "public": {
            "$id": "/properties/public",
            "type": "boolean",
            "title": "Will this strategy be public",
            "default": "false",
            "examples": [
                "true",
                "false"
            ]
        },
        "content": {
            "$id": "/properties/content",
            "type": "string",
            "title": "The Content Schema ",
            "default": "",
            "examples": [
                "IyAtLS0gRG8gbm90IHJlbW92ZSB0aGVzZSBsaWJzIC0tLQpmcm9tIGZyZXF0cmFkZS5zdHJhdGVneS5pbnRlcmZhY2UgaW1wb3J0IElTdHJhdGVneQpmcm9tIHR5cGluZyBpbXBvcnQgRGljdCwgTGlzdApmcm9tIGh5cGVyb3B0IGltcG9ydCBocApmcm9tIGZ1bmN0b29scyBpbXBvcnQgcmVkdWNlCmZyb20gcGFuZGFzIGltcG9ydCBEYXRhRnJhbWUKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQoKaW1wb3J0IHRhbGliLmFic3RyYWN0IGFzIHRhCmltcG9ydCBmcmVxdHJhZGUudmVuZG9yLnF0cHlsaWIuaW5kaWNhdG9ycyBhcyBxdHB5bGliCgpjbGFzcyBUZXN0U3RyYXRlZ3koSVN0cmF0ZWd5KToKICAgIG1pbmltYWxfcm9pID0gewogICAgICAgICIwIjogMC41CiAgICB9CiAgICBzdG9wbG9zcyA9IC0wLjIKICAgIHRpY2tlcl9pbnRlcnZhbCA9ICc1bScKCiAgICBkZWYgcG9wdWxhdGVfaW5kaWNhdG9ycyhzZWxmLCBkYXRhZnJhbWU6IERhdGFGcmFtZSkgLT4gRGF0YUZyYW1lOgogICAgICAgIG1hY2QgPSB0YS5NQUNEKGRhdGFmcmFtZSkKICAgICAgICBkYXRhZnJhbWVbJ21hU2hvcnQnXSA9IHRhLkVNQShkYXRhZnJhbWUsIHRpbWVwZXJpb2Q9OCkKICAgICAgICBkYXRhZnJhbWVbJ21hTWVkaXVtJ10gPSB0YS5FTUEoZGF0YWZyYW1lLCB0aW1lcGVyaW9kPTIxKQogICAgICAgIHJldHVybiBkYXRhZnJhbWUKCiAgICBkZWYgcG9wdWxhdGVfYnV5X3RyZW5kKHNlbGYsIGRhdGFmcmFtZTogRGF0YUZyYW1lKSAtPiBEYXRhRnJhbWU6CiAgICAgICAgZGF0YWZyYW1lLmxvY1sKICAgICAgICAgICAgKAogICAgICAgICAgICAgICAgcXRweWxpYi5jcm9zc2VkX2Fib3ZlKGRhdGFmcmFtZVsnbWFTaG9ydCddLCBkYXRhZnJhbWVbJ21hTWVkaXVtJ10pCiAgICAgICAgICAgICksCiAgICAgICAgICAgICdidXknXSA9IDEKCiAgICAgICAgcmV0dXJuIGRhdGFmcmFtZQoKICAgIGRlZiBwb3B1bGF0ZV9zZWxsX3RyZW5kKHNlbGYsIGRhdGFmcmFtZTogRGF0YUZyYW1lKSAtPiBEYXRhRnJhbWU6CiAgICAgICAgZGF0YWZyYW1lLmxvY1sKICAgICAgICAgICAgKAogICAgICAgICAgICAgICAgcXRweWxpYi5jcm9zc2VkX2Fib3ZlKGRhdGFmcmFtZVsnbWFNZWRpdW0nXSwgZGF0YWZyYW1lWydtYVNob3J0J10pCiAgICAgICAgICAgICksCiAgICAgICAgICAgICdzZWxsJ10gPSAxCiAgICAgICAgcmV0dXJuIGRhdGFmcmFtZQoKCiAgICAgICAg"
            ]
        }
    }
}
