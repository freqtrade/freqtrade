import logging

from freqtrade.constants import Config
from freqtrade.enums import RPCMessageType
from freqtrade.rpc import RPC
from freqtrade.rpc.webhook import Webhook


logger = logging.getLogger(__name__)


class Discord(Webhook):
    def __init__(self, rpc: "RPC", config: Config):
        self._config = config
        self.rpc = rpc
        self.strategy = config.get("strategy", "")
        self.timeframe = config.get("timeframe", "")
        self.bot_name = config.get("bot_name", "")

        self._url = config["discord"]["webhook_url"]
        self._format = "json"
        self._retries = 1
        self._retry_delay = 0.1
        self._timeout = self._config["discord"].get("timeout", 10)

    def cleanup(self) -> None:
        """
        Cleanup pending module resources.
        This will do nothing for webhooks, they will simply not be called anymore
        """
        pass

    def send_msg(self, msg) -> None:
        if fields := self._config["discord"].get(msg["type"].value):
            logger.info(f"Sending discord message: {msg}")

            msg["strategy"] = self.strategy
            msg["timeframe"] = self.timeframe
            msg["bot_name"] = self.bot_name
            color = 0x0000FF
            if msg["type"] in (RPCMessageType.EXIT, RPCMessageType.EXIT_FILL):
                profit_ratio = msg.get("profit_ratio")
                color = 0x00FF00 if profit_ratio > 0 else 0xFF0000
            title = msg["type"].value
            if "pair" in msg:
                title = f"Trade: {msg['pair']} {msg['type'].value}"
            embeds = [
                {
                    "title": title,
                    "color": color,
                    "fields": [],
                }
            ]
            for f in fields:
                for k, v in f.items():
                    v = v.format(**msg)
                    embeds[0]["fields"].append({"name": k, "value": v, "inline": True})

            # Send the message to discord channel
            payload = {"embeds": embeds}
            self._send_msg(payload)
