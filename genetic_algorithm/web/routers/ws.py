"""
WebSocket endpoint — streams EventBus events to connected browser clients.

Supports:
  - Real-time event streaming (all events or filtered by run_id)
  - Bulk history replay on connect (so late joiners catch up)
  - Periodic heartbeat to detect stale connections
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from genetic_algorithm.web.event_bus import get_event_bus

logger = logging.getLogger(__name__)
router = APIRouter()

# Default heartbeat interval (can be overridden via WebConfig on app.state)
_DEFAULT_HEARTBEAT_INTERVAL = 30.0


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint.

    On connect, sends buffered event history, then streams new events.
    Client can send JSON commands to filter by run_id.
    Server sends periodic heartbeat messages to keep the connection alive.
    """
    await websocket.accept()
    bus = get_event_bus()
    queue = bus.create_async_consumer()
    filter_run_id: str | None = None

    # Read heartbeat interval from app config if available
    heartbeat_interval = _DEFAULT_HEARTBEAT_INTERVAL
    try:
        web_config = websocket.app.state.web_config
        heartbeat_interval = getattr(web_config, "ws_heartbeat_interval", _DEFAULT_HEARTBEAT_INTERVAL)
    except Exception:
        pass

    last_heartbeat = time.monotonic()

    try:
        # Send event history (so the UI is immediately populated)
        history = bus.get_history(limit=500)
        for event in history:
            await websocket.send_json(event.to_dict())

        # Stream events + listen for client commands
        while True:
            # Wait for next event with a short timeout so we can also
            # check for incoming client messages and send heartbeats
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
                # Apply client-side filter
                if filter_run_id and event.run_id != filter_run_id:
                    continue
                await websocket.send_json(event.to_dict())
            except asyncio.TimeoutError:
                pass

            # Send heartbeat if interval elapsed
            now = time.monotonic()
            if now - last_heartbeat >= heartbeat_interval:
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": time.time(),
                })
                last_heartbeat = now

            # Non-blocking check for client commands
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(), timeout=0.01
                )
                try:
                    cmd = json.loads(raw)
                    if cmd.get("command") == "subscribe":
                        filter_run_id = cmd.get("run_id")
                    elif cmd.get("command") == "unsubscribe":
                        filter_run_id = None
                except (json.JSONDecodeError, KeyError):
                    pass
            except asyncio.TimeoutError:
                pass

    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected")
    except Exception:
        logger.exception("WebSocket error")
    finally:
        bus.remove_async_consumer(queue)
