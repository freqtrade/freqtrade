import sys
from typing import Any, Dict, Optional, Sequence, Union

from rich.console import Console
from rich.table import Table
from rich.text import Text


TextOrString = Union[str, Text]


def print_rich_table(
    tabular_data: Sequence[Union[Dict[str, Any], Sequence[TextOrString]]],
    headers: Sequence[str],
    summary: Optional[str] = None,
    *,
    table_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    table = Table(title=summary, **(table_kwargs or {}))

    for header in headers:
        table.add_column(header, justify="right")

    for row in tabular_data:
        if isinstance(row, dict):
            table.add_row(*[str(row[header]) for header in headers])
        else:
            table.add_row(*row)

    console = Console(
        width=200 if "pytest" in sys.modules else None,
    )
    console.print(table)
