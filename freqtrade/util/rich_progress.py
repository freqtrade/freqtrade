from typing import Callable, List, Union

from rich.console import ConsoleRenderable, Group, RichCast
from rich.progress import Progress


class CustomProgress(Progress):
    def __init__(self, *args, cust_objs=[], cust_callables: List[Callable] = [], **kwargs) -> None:
        self._cust_objs = cust_objs
        self._cust_callables = cust_callables
        super().__init__(*args, **kwargs)

    def get_renderable(self) -> Union[ConsoleRenderable, RichCast, str]:
        objs = [obj for obj in self._cust_objs]
        for cust_call in self._cust_callables:
            objs.append(cust_call())
        renderable = Group(*objs, *self.get_renderables())
        return renderable
