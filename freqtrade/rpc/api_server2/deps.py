from .webserver import ApiServer


def get_rpc():
    return ApiServer._rpc


def get_config():
    return ApiServer._config
