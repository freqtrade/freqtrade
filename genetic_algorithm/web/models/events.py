"""
Pydantic models for WebSocket event payloads.

Every event published on the EventBus is serialised through these models
before being sent over the WebSocket to browser clients.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel


class WSEvent(BaseModel):
    """Envelope sent to WebSocket clients."""

    type: str  # EventType.value
    run_id: str
    data: Dict[str, Any] = {}
    timestamp: float


class WSCommand(BaseModel):
    """
    Envelope received from WebSocket clients (future: live steering).

    Commands:
        subscribe   — subscribe to events for a specific run
        unsubscribe — stop receiving events for a run
    """

    command: str
    run_id: Optional[str] = None
    data: Dict[str, Any] = {}
