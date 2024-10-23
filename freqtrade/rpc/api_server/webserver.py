import logging
from ipaddress import ip_address
from typing import Any, Optional

import orjson
import uvicorn
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from freqtrade.configuration import running_in_docker
from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException
from freqtrade.rpc.api_server.uvicorn_threaded import UvicornServer
from freqtrade.rpc.api_server.ws.message_stream import MessageStream
from freqtrade.rpc.rpc import RPC, RPCException, RPCHandler
from freqtrade.rpc.rpc_types import RPCSendMsg


logger = logging.getLogger(__name__)


class FTJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        """
        Use rapidjson for responses
        Handles NaN and Inf / -Inf in a javascript way by default.
        """
        return orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY)


class ApiServer(RPCHandler):
    __instance = None
    __initialized = False

    _rpc: RPC
    _has_rpc: bool = False
    _config: Config = {}
    # websocket message stuff
    _message_stream: Optional[MessageStream] = None

    def __new__(cls, *args, **kwargs):
        """
        This class is a singleton.
        We'll only have one instance of it around.
        """
        if ApiServer.__instance is None:
            ApiServer.__instance = object.__new__(cls)
            ApiServer.__initialized = False
        return ApiServer.__instance

    def __init__(self, config: Config, standalone: bool = False) -> None:
        ApiServer._config = config
        if self.__initialized and (standalone or self._standalone):
            return
        self._standalone: bool = standalone
        self._server = None

        ApiServer.__initialized = True

        api_config = self._config["api_server"]

        self.app = FastAPI(
            title="Freqtrade API",
            docs_url="/docs" if api_config.get("enable_openapi", False) else None,
            redoc_url=None,
            default_response_class=FTJSONResponse,
        )
        self.configure_app(self.app, self._config)
        self.start_api()

    def add_rpc_handler(self, rpc: RPC):
        """
        Attach rpc handler
        """
        if not ApiServer._has_rpc:
            ApiServer._rpc = rpc
            ApiServer._has_rpc = True
        else:
            # This should not happen assuming we didn't mess up.
            raise OperationalException("RPC Handler already attached.")

    def cleanup(self) -> None:
        """Cleanup pending module resources"""
        ApiServer._has_rpc = False
        del ApiServer._rpc
        if self._server and not self._standalone:
            logger.info("Stopping API Server")
            # self._server.force_exit, self._server.should_exit = True, True
            self._server.cleanup()

    @classmethod
    def shutdown(cls):
        cls.__initialized = False
        del cls.__instance
        cls.__instance = None
        cls._has_rpc = False
        cls._rpc = None

    def send_msg(self, msg: RPCSendMsg) -> None:
        """
        Publish the message to the message stream
        """
        if ApiServer._message_stream:
            ApiServer._message_stream.publish(msg)

    def handle_rpc_exception(self, request, exc):
        logger.error(f"API Error calling: {exc}")
        return JSONResponse(
            status_code=502, content={"error": f"Error querying {request.url.path}: {exc.message}"}
        )

    def configure_app(self, app: FastAPI, config):
        from freqtrade.rpc.api_server.api_auth import http_basic_or_jwt_token, router_login
        from freqtrade.rpc.api_server.api_background_tasks import router as api_bg_tasks
        from freqtrade.rpc.api_server.api_backtest import router as api_backtest
        from freqtrade.rpc.api_server.api_pairlists import router as api_pairlists
        from freqtrade.rpc.api_server.api_v1 import router as api_v1
        from freqtrade.rpc.api_server.api_v1 import router_public as api_v1_public
        from freqtrade.rpc.api_server.api_ws import router as ws_router
        from freqtrade.rpc.api_server.deps import is_webserver_mode
        from freqtrade.rpc.api_server.web_ui import router_ui

        app.include_router(api_v1_public, prefix="/api/v1")

        app.include_router(router_login, prefix="/api/v1", tags=["auth"])
        app.include_router(
            api_v1,
            prefix="/api/v1",
            dependencies=[Depends(http_basic_or_jwt_token)],
        )
        app.include_router(
            api_backtest,
            prefix="/api/v1",
            dependencies=[Depends(http_basic_or_jwt_token), Depends(is_webserver_mode)],
        )
        app.include_router(
            api_bg_tasks,
            prefix="/api/v1",
            dependencies=[Depends(http_basic_or_jwt_token), Depends(is_webserver_mode)],
        )
        app.include_router(
            api_pairlists,
            prefix="/api/v1",
            dependencies=[Depends(http_basic_or_jwt_token), Depends(is_webserver_mode)],
        )
        app.include_router(ws_router, prefix="/api/v1")
        # UI Router MUST be last!
        app.include_router(router_ui, prefix="")

        app.add_middleware(
            CORSMiddleware,
            allow_origins=config["api_server"].get("CORS_origins", []),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        app.add_exception_handler(RPCException, self.handle_rpc_exception)
        app.add_event_handler(event_type="startup", func=self._api_startup_event)
        app.add_event_handler(event_type="shutdown", func=self._api_shutdown_event)

    async def _api_startup_event(self):
        """
        Creates the MessageStream class on startup
        so it has access to the same event loop
        as uvicorn
        """
        if not ApiServer._message_stream:
            ApiServer._message_stream = MessageStream()

    async def _api_shutdown_event(self):
        """
        Removes the MessageStream class on shutdown
        """
        if ApiServer._message_stream:
            ApiServer._message_stream = None

    def start_api(self):
        """
        Start API ... should be run in thread.
        """
        rest_ip = self._config["api_server"]["listen_ip_address"]
        rest_port = self._config["api_server"]["listen_port"]

        logger.info(f"Starting HTTP Server at {rest_ip}:{rest_port}")
        if not ip_address(rest_ip).is_loopback and not running_in_docker():
            logger.warning("SECURITY WARNING - Local Rest Server listening to external connections")
            logger.warning(
                "SECURITY WARNING - This is insecure please set to your loopback,"
                "e.g 127.0.0.1 in config.json"
            )

        if not self._config["api_server"].get("password"):
            logger.warning(
                "SECURITY WARNING - No password for local REST Server defined. "
                "Please make sure that this is intentional!"
            )

        if self._config["api_server"].get("jwt_secret_key", "super-secret") in (
            "super-secret, somethingrandom"
        ):
            logger.warning(
                "SECURITY WARNING - `jwt_secret_key` seems to be default."
                "Others may be able to log into your bot."
            )

        logger.info("Starting Local Rest Server.")
        verbosity = self._config["api_server"].get("verbosity", "error")

        uvconfig = uvicorn.Config(
            self.app,
            port=rest_port,
            host=rest_ip,
            use_colors=False,
            log_config=None,
            access_log=True if verbosity != "error" else False,
            ws_ping_interval=None,  # We do this explicitly ourselves
        )
        try:
            self._server = UvicornServer(uvconfig)
            if self._standalone:
                self._server.run()
            else:
                self._server.run_in_thread()
        except Exception:
            logger.exception("Api server failed to start.")
