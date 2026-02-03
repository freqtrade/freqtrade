from multiprocessing import get_all_start_methods, get_start_method, set_start_method


def set_mp_start_method():
    """
    Set multiprocessing start method to not be fork.
    forkserver will become the default in 3.14 - and is deprecated in 3.13
    """
    try:
        sms = get_all_start_methods()
        if "forkserver" in sms and get_start_method(True) is None:
            set_start_method("forkserver")
    except RuntimeError:
        pass
