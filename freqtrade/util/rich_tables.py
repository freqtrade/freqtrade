from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table


def print_rich_table(summary: str, headers: List[str], tabular_data: List[Dict[str, Any]]) -> None:
    table = Table(title=summary)

    for header in headers:
        table.add_column(header, justify="right")

    for row in tabular_data:
        table.add_row(*[str(row[header]) for header in headers])

    console = Console()
    console.print(table)
