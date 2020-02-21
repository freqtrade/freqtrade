from joblib._parallel_backends import LokyBackend

hyperopt = None


class MultiCallback:
    def __init__(self, *callbacks):
        self.callbacks = [cb for cb in callbacks if cb]

    def __call__(self, out):
        for cb in self.callbacks:
            cb(out)


class CustomImmediateResultBackend(LokyBackend):
    def callback(self, result):
        """
        Our custom completion callback. Executed in the parent process.
        Use it to run Optimizer.tell() with immediate results of the backtest()
        evaluated in the joblib worker process.
        """
        if not result.exception():
            # Fetch results from the Future object passed to us.
            # Future object is assumed to be 'done' already.
            f_val = result.result().copy()
            hyperopt.parallel_callback(f_val)

    def apply_async(self, func, callback=None):
        cbs = MultiCallback(callback, self.callback)
        return super().apply_async(func, cbs)
