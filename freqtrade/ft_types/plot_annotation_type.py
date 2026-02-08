from datetime import datetime
from typing import Literal, Required

from pydantic import TypeAdapter
from typing_extensions import TypedDict


class _BaseAnnotationType(TypedDict, total=False):
    color: str
    label: str
    z_level: int


class _Base2DAnnotationType(_BaseAnnotationType, total=False):
    start: str | datetime
    end: str | datetime
    y_start: float
    y_end: float


class AreaAnnotationType(_Base2DAnnotationType, total=False):
    type: Required[Literal["area"]]


class LineAnnotationType(_Base2DAnnotationType, total=False):
    type: Required[Literal["line"]]
    width: int
    line_style: Literal["solid", "dashed", "dotted"]


class PointAnnotationType(_BaseAnnotationType, total=False):
    type: Required[Literal["point"]]
    x: str | datetime
    y: float
    size: int
    shape: Literal["circle", "rect", "roundRect", "triangle", "pin", "arrow", "none"]
    rotate: int


AnnotationType = AreaAnnotationType | LineAnnotationType | PointAnnotationType

AnnotationTypeTA: TypeAdapter[AnnotationType] = TypeAdapter(AnnotationType)
