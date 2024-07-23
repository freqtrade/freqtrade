from typing import Union

from rich.console import ConsoleRenderable, Group, RichCast
from rich.progress import Progress


class CustomProgress(Progress):
    def __init__(self, *args, cust_objs=[], **kwargs) -> None:
        self._cust_objs = cust_objs
        super().__init__(*args, **kwargs)

    def get_renderable(self) -> Union[ConsoleRenderable, RichCast, str]:
        renderable = Group(*self._cust_objs, *self.get_renderables())
        return renderable
