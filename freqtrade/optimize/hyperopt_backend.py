from typing import Any
from queue import Queue
from multiprocessing.managers import SyncManager

hyperopt: Any = None
manager: SyncManager
# stores the optimizers in multi opt mode
optimizers: Queue
# stores a list of the results to share between optimizers
# in the form of dict[tuple(Xi)] = yi
results_board: Queue
# store the results in single opt mode
results: Queue
