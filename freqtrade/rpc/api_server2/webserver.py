from typing import Any, Dict, Optional

import uvicorn
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from freqtrade.rpc.rpc import RPC, RPCHandler

from .uvicorn_threaded import UvicornServer


class ApiServer(RPCHandler):

    _rpc: Optional[RPC] = None
    _config: Dict[str, Any] = {}

    def __init__(self, rpc: RPC, config: Dict[str, Any]) -> None:
        super().__init__(rpc, config)
        self._server = None

        ApiServer._rpc = rpc
        ApiServer._config = config

        self.app = FastAPI(title="Freqtrade API")
        self.configure_app(self.app, self._config)

        self.start_api()

    def cleanup(self) -> None:
        """ Cleanup pending module resources """
        if self._server:
            self._server.cleanup()

    def send_msg(self, msg: Dict[str, str]) -> None:
        pass

    def configure_app(self, app: FastAPI, config):
        from .api_v1 import router as api_v1
        from .api_v1 import router_public as api_v1_public
        from .api_auth import http_basic_or_jwt_token, router_login
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

    def start_api(self):
        """
        Start API ... should be run in thread.
        """
        uvconfig = uvicorn.Config(self.app,
                                  port=self._config['api_server'].get('listen_port', 8080),
                                  host=self._config['api_server'].get(
                                      'listen_ip_address', '127.0.0.1'),
                                  access_log=True)
        self._server = UvicornServer(uvconfig)

        self._server.run_in_thread()
