import logging
from ipaddress import IPv4Address
from typing import Any, Dict

import uvicorn
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from freqtrade.rpc.api_server.uvicorn_threaded import UvicornServer
from freqtrade.rpc.rpc import RPC, RPCException, RPCHandler


logger = logging.getLogger(__name__)


class ApiServer(RPCHandler):

    _rpc: RPC
    _has_rpc: bool = False
    _config: Dict[str, Any] = {}

    def __init__(self, rpc: RPC, config: Dict[str, Any]) -> None:
        super().__init__(rpc, config)
        self._server = None

        ApiServer._rpc = rpc
        ApiServer._has_rpc = True
        ApiServer._config = config
        api_config = self._config['api_server']

        self.app = FastAPI(title="Freqtrade API",
                           docs_url='/docs' if api_config.get('enable_openapi', False) else None,
                           redoc_url=None,
                           )
        self.configure_app(self.app, self._config)

        self.start_api()

    def cleanup(self) -> None:
        """ Cleanup pending module resources """
        if self._server:
            logger.info("Stopping API Server")
            self._server.cleanup()

    def send_msg(self, msg: Dict[str, str]) -> None:
        pass

    def handle_rpc_exception(self, request, exc):
        logger.exception(f"API Error calling: {exc}")
        return JSONResponse(
            status_code=502,
            content={'error': f"Error querying {request.url.path}: {exc.message}"}
        )

    def configure_app(self, app: FastAPI, config):
        from freqtrade.rpc.api_server.api_auth import http_basic_or_jwt_token, router_login
        from freqtrade.rpc.api_server.api_v1 import router as api_v1
        from freqtrade.rpc.api_server.api_v1 import router_public as api_v1_public
        app.include_router(api_v1_public, prefix="/api/v1")

        app.include_router(api_v1, prefix="/api/v1",
                           dependencies=[Depends(http_basic_or_jwt_token)],
                           )
        app.include_router(router_login, prefix="/api/v1", tags=["auth"])

        app.add_middleware(
            CORSMiddleware,
            allow_origins=config['api_server'].get('CORS_origins', []),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        app.add_exception_handler(RPCException, self.handle_rpc_exception)

    def start_api(self):
        """
        Start API ... should be run in thread.
        """
        rest_ip = self._config['api_server']['listen_ip_address']
        rest_port = self._config['api_server']['listen_port']

        logger.info(f'Starting HTTP Server at {rest_ip}:{rest_port}')
        if not IPv4Address(rest_ip).is_loopback:
            logger.warning("SECURITY WARNING - Local Rest Server listening to external connections")
            logger.warning("SECURITY WARNING - This is insecure please set to your loopback,"
                           "e.g 127.0.0.1 in config.json")

        if not self._config['api_server'].get('password'):
            logger.warning("SECURITY WARNING - No password for local REST Server defined. "
                           "Please make sure that this is intentional!")

        if (self._config['api_server'].get('jwt_secret_key', 'super-secret')
                in ('super-secret, somethingrandom')):
            logger.warning("SECURITY WARNING - `jwt_secret_key` seems to be default."
                           "Others may be able to log into your bot.")

        logger.info('Starting Local Rest Server.')
        verbosity = self._config['api_server'].get('verbosity', 'error')
        log_config = uvicorn.config.LOGGING_CONFIG
        # Change logging of access logs to stderr
        log_config["handlers"]["access"]["stream"] = log_config["handlers"]["default"]["stream"]
        uvconfig = uvicorn.Config(self.app,
                                  port=rest_port,
                                  host=rest_ip,
                                  use_colors=False,
                                  log_config=log_config,
                                  access_log=True if verbosity != 'error' else False,
                                  )
        try:
            self._server = UvicornServer(uvconfig)
            self._server.run_in_thread()
        except Exception:
            logger.exception("Api server failed to start.")
