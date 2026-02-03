import os
import subprocess  # noqa: S404, RUF100
import sys
from io import StringIO
from pathlib import Path


def _write_partial_file(filename: str, content: str):
    with Path(filename).open("w") as f:
        f.write(f"``` output\n{content}\n```\n")


def _get_help_output(parser) -> str:
    """Capture the help output from a parser."""
    output = StringIO()
    parser.print_help(file=output)
    return output.getvalue()


def extract_command_partials():
    # Set terminal width to 80 columns for consistent output formatting
    os.environ["COLUMNS"] = "80"

    # Import Arguments here to avoid circular imports and ensure COLUMNS is set
    from freqtrade.commands.arguments import Arguments

    subcommands = [
        "trade",
        "create-userdir",
        "new-config",
        "show-config",
        "new-strategy",
        "download-data",
        "convert-data",
        "convert-trade-data",
        "trades-to-ohlcv",
        "list-data",
        "backtesting",
        "backtesting-show",
        "backtesting-analysis",
        "edge",
        "hyperopt",
        "hyperopt-list",
        "hyperopt-show",
        "list-exchanges",
        "list-markets",
        "list-pairs",
        "list-strategies",
        "list-hyperoptloss",
        "list-freqaimodels",
        "list-timeframes",
        "show-trades",
        "test-pairlist",
        "convert-db",
        "install-ui",
        "plot-dataframe",
        "plot-profit",
        "webserver",
        "strategy-updater",
        "lookahead-analysis",
        "recursive-analysis",
    ]

    # Build the Arguments class to get the parser with all subcommands
    args = Arguments(None)
    args._build_subcommands()

    # Get main help output
    main_help = _get_help_output(args.parser)
    _write_partial_file("docs/commands/main.md", main_help)

    # Get subparsers from the main parser
    # The subparsers are stored in _subparsers._group_actions[0].choices
    subparsers_action = None
    for action in args.parser._subparsers._group_actions:
        if hasattr(action, "choices"):
            subparsers_action = action
            break

    if subparsers_action is None:
        raise RuntimeError("Could not find subparsers in the main parser")

    for command in subcommands:
        print(f"Running for {command}")
        if command in subparsers_action.choices:
            subparser = subparsers_action.choices[command]
            help_output = _get_help_output(subparser)
            _write_partial_file(f"docs/commands/{command}.md", help_output)
        else:
            print(f"  Warning: subcommand '{command}' not found in parser")

    # freqtrade-client still uses subprocess as requested
    print("Running for freqtrade-client")
    result_client = subprocess.run(["freqtrade-client", "--show"], capture_output=True, text=True)

    _write_partial_file("docs/commands/freqtrade-client.md", result_client.stdout)


if __name__ == "__main__":
    if sys.version_info < (3, 13):  # pragma: no cover
        sys.exit(
            "argparse output changed in Python 3.13+. "
            "To keep command partials up to date, please run this script with Python 3.13+."
        )
    extract_command_partials()
