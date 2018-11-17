from typing import Any, Dict
from collections import namedtuple


class Context(object):

    # context data structure
    context = namedtuple(
        'context',
        ['trades', 'ticker']
    )
