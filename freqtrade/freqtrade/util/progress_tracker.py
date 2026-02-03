from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from freqtrade.util.rich_progress import CustomProgress


def retrieve_progress_tracker(pt: CustomProgress | None) -> CustomProgress:
    if pt is None:
        return get_progress_tracker()
    return pt


def get_progress_tracker(**kwargs) -> CustomProgress:
    """
    Get progress Bar with custom columns.
    """
    from freqtrade.loggers import error_console

    return CustomProgress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
        expand=True,
        console=error_console,
        **kwargs,
    )
