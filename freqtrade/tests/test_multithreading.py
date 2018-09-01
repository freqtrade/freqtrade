from freqtrade import (multithreading)


def test_client_init():
    # Check if create a Dask Client
    default_conf = {}
    default_conf.update({
        'multithreading': {'use_multithreading': True, 'workers_number': 4}
    })
    client = multithreading.init(default_conf)
    assert (str(type(client))) == "<class 'distributed.client.Client'>"
