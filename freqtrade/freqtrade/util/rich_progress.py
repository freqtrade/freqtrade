from collections.abc import Callable
from typing import Any

from rich.console import ConsoleRenderable, Group, RichCast
from rich.progress import Progress, Task, TaskID


class CustomProgress(Progress):
    def __init__(
        self,
        *args,
        cust_objs: list[ConsoleRenderable] | None = None,
        cust_callables: list[Callable[[], ConsoleRenderable]] | None = None,
        ft_callback: Callable[[Task], None] | None = None,
        **kwargs,
    ) -> None:
        self._cust_objs = cust_objs or []
        self._cust_callables = cust_callables or []
        self._ft_callback = ft_callback
        if self._ft_callback:
            kwargs["disable"] = True

        super().__init__(*args, **kwargs)

    def update(
        self,
        task_id: TaskID,
        *,
        total: float | None = None,
        completed: float | None = None,
        advance: float | None = None,
        description: str | None = None,
        visible: bool | None = None,
        refresh: bool = False,
        **fields: Any,
    ) -> None:
        t = super().update(
            task_id,
            total=total,
            completed=completed,
            advance=advance,
            description=description,
            visible=visible,
            refresh=refresh,
            **fields,
        )
        if self._ft_callback:
            self._ft_callback(
                self.tasks[task_id],
            )
        return t

    def get_renderable(self) -> ConsoleRenderable | RichCast | str:
        objs = [obj for obj in self._cust_objs]
        for cust_call in self._cust_callables:
            objs.append(cust_call())
        renderable = Group(*objs, *self.get_renderables())
        return renderable
