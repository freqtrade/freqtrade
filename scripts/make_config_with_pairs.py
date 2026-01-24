import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("make_config_with_pairs")


def _load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        logger.error("File not found: %s", path)
        raise
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", path, exc)
        raise


def _write_json(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("Wrote derived config to %s", path)
    except OSError as exc:
        logger.error("Failed to write JSON to %s: %s", path, exc)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create config with generated pairs whitelist.")
    parser.add_argument("--base", "--base-config", required=True, help="Base config JSON path")
    parser.add_argument("--pairs", required=True, help="Pairs JSON path")
    parser.add_argument("--out", "--out-config", required=True, help="Output config path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_path = Path(args.base)
    pairs_path = Path(args.pairs)
    out_path = Path(args.out)

    config = _load_json(base_path)
    pairs = _load_json(pairs_path)
    if not isinstance(pairs, list):
        logger.error("Pairs file must contain a list of pairs.")
        raise SystemExit(1)

    exchange = config.get("exchange")
    if not isinstance(exchange, dict):
        logger.error("Base config is missing exchange configuration.")
        raise SystemExit(1)

    exchange["pair_whitelist"] = pairs
    config["exchange"] = exchange
    _write_json(out_path, config)


if __name__ == "__main__":
    main()
