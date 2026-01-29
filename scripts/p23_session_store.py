#!/usr/bin/env python3
"""
P23 Session Store Utility
Reads session token from STDIN (safe) or Argument (unsafe) and writes to
secure storage with strict permissions.

Usage:
  echo "token" | python3 scripts/p23_session_store.py --stdin
"""

import argparse
import os
import stat
import sys
import tempfile
from pathlib import Path

DEFAULT_SECRET_PATH = "user_data/secrets/breeze_session_token"


def validate_token(token: str) -> str | None:
    if not token:
        return "Empty token"
    # Normalize
    token = token.strip()
    if len(token) < 10:
        return "Token too short (min 10 chars)"
    if not token.isascii():
        return "Token contains non-ascii characters"
    return None


def secure_write(path_str: str, token: str) -> None:
    target_path = Path(path_str)
    # Ensure parent dir exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file first
    fd, tmp_nt = tempfile.mkstemp(dir=target_path.parent, text=True)
    tmp_path = Path(tmp_nt)

    try:
        with os.fdopen(fd, "w") as f:
            f.write(token)

        # Enforce 0400 (owner read-only)
        tmp_path.chmod(stat.S_IRUSR)  # 0400

        # Atomic move
        tmp_path.replace(target_path)
        print(f"Token written securely to {target_path}")
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise e


def main():
    parser = argparse.ArgumentParser(description="Secure Session Store")
    parser.add_argument("--stdin", action="store_true", help="Read token from stdin")
    parser.add_argument(
        "--token", help="Token value (DISCOURAGED - appears in process list)", default=None
    )
    parser.add_argument("--path", default=DEFAULT_SECRET_PATH, help="Target path")

    args = parser.parse_args()

    token = ""
    if args.stdin:
        # Read from stdin
        token = sys.stdin.read()
    elif args.token:
        print("WARNING: Passing token via argument is insecure.")
        token = args.token
    else:
        print("Error: Must provide --stdin or --token")
        sys.exit(1)

    token = token.strip()

    error = validate_token(token)
    if error:
        print(f"Error: {error}")
        sys.exit(1)

    try:
        secure_write(args.path, token)
    except Exception as e:
        print(f"Error writing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
