from typing import Any, Dict, List, Tuple
from queue import Queue
from multiprocessing.managers import SyncManager

hyperopt: Any = None
manager: SyncManager
# stores the optimizers in multi opt mode
optimizers: Queue
# stores the results to share between optimizers
# in the form of key = Tuple[Xi], value = Tuple[float, int]
# where float is the loss and int is a decreasing counter of optimizers
# that have registered the result
results_shared: Dict[Tuple, Tuple]
# in single mode the results_list is used to pass the results to the optimizer
# to fit new models
results_list: List
# results_batch stores keeps results per batch that are eventually logged and stored
results_batch: Queue
