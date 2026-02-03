from freqtrade.constants import Config


def get_dry_run_wallet(config: Config) -> int | float:
    """
    Return dry-run wallet balance in stake currency from configuration.
    This setup also supports dictionary mode for dry-run-wallet.
    """
    if isinstance(_start_cap := config["dry_run_wallet"], float | int):
        return _start_cap
    else:
        return _start_cap.get(config["stake_currency"], 0.0)
