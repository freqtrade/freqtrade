#!/usr/bin/env python3
"""
Atomic Command Runner with Locking.
Usage: python with_lock.py --lock <lockfile> --cmd "<command>"
Ensures only one instance of <command> runs for the given <lockfile>.
"""

import argparse
import fcntl
import logging
import subprocess
import sys
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("with_lock")


def main():
    parser = argparse.ArgumentParser(description="Run command with exclusive lock.")
    parser.add_argument("--lock", required=True, help="Path to lock file")
    parser.add_argument("--cmd", required=True, help="Command to run")
    args = parser.parse_args()

    lock_path = args.lock

    # Ensure lock directory exists
    lock_dir = os.path.dirname(lock_path)
    if lock_dir and not os.path.exists(lock_dir):
        try:
            os.makedirs(lock_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create lock directory: {e}", exc_info=True)
            sys.exit(1)

    try:
        lock_fd = open(lock_path, "w")
    except Exception as e:
        logger.error(f"Failed to open lock file: {e}", exc_info=True)
        sys.exit(1)

    try:
        # Try to acquire exclusive, non-blocking lock
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        logger.warning(f"Could not acquire lock on {lock_path}. Another instance is running.")
        sys.exit(1)

    # Lock acquired
    logger.info(f"Lock acquired on {lock_path}. Running: {args.cmd}")

    try:
        # Run command
        # Use shell=True to allow complex commands
        ret = subprocess.call(args.cmd, shell=True)
        logger.info(f"Command finished with exit code {ret}")
        sys.exit(ret)
    except Exception as e:
        logger.error(f"Failed to run command: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Unlock is automatic on file close/process exit, but being explicit is good
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


if __name__ == "__main__":
    main()
