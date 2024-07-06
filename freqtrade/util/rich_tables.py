import sys
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table


def print_rich_table(
    tabular_data: List[Dict[str, Any]],
    headers: List[str],
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
