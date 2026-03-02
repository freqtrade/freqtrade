/**
 * WebSocket hook — maintains a persistent connection to the GA backend,
 * reconnects on drop, and feeds events into the Zustand store.
 *
 * Also monitors server heartbeats: if no heartbeat arrives within 2× the
 * expected interval the connection is assumed stale and force-reconnected.
 */

import { useEffect, useRef, useCallback } from 'react';
import type { WSEvent } from '../types';
import { useStore } from '../store/useStore';

const WS_RECONNECT_MS = 2000;
const WS_MAX_RECONNECT_MS = 30000;
const WS_HEARTBEAT_INTERVAL = 30_000; // must match server's ws_heartbeat_interval
const WS_HEARTBEAT_TIMEOUT = WS_HEARTBEAT_INTERVAL * 2.5; // allow generous slack

function getWsUrl(): string {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${window.location.host}/ws`;
}

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectDelay = useRef(WS_RECONNECT_MS);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const heartbeatTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const { pushEvent, setConnected } = useStore();

  const resetHeartbeatWatchdog = useCallback((ws: WebSocket) => {
    clearTimeout(heartbeatTimer.current);
    heartbeatTimer.current = setTimeout(() => {
      // No heartbeat received — connection is stale
      if (ws.readyState === WebSocket.OPEN) {
        ws.close(); // will trigger onclose → reconnect
      }
    }, WS_HEARTBEAT_TIMEOUT);
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      const ws = new WebSocket(getWsUrl());
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        reconnectDelay.current = WS_RECONNECT_MS;
        resetHeartbeatWatchdog(ws);
      };

      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          // Heartbeat messages keep the watchdog alive but aren't stored
          if (data.type === 'heartbeat') {
            resetHeartbeatWatchdog(ws);
            return;
          }
          const event: WSEvent = data;
          pushEvent(event);
          // Any real message also resets the watchdog
          resetHeartbeatWatchdog(ws);
        } catch {
          // ignore malformed messages
        }
      };

      ws.onclose = () => {
        setConnected(false);
        clearTimeout(heartbeatTimer.current);
        // Exponential backoff reconnect
        reconnectTimer.current = setTimeout(() => {
          reconnectDelay.current = Math.min(
            reconnectDelay.current * 1.5,
            WS_MAX_RECONNECT_MS,
          );
          connect();
        }, reconnectDelay.current);
      };

      ws.onerror = () => {
        ws.close();
      };
    } catch {
      // Retry on connection failure
      reconnectTimer.current = setTimeout(connect, reconnectDelay.current);
    }
  }, [pushEvent, setConnected, resetHeartbeatWatchdog]);

  // Subscribe to a specific run (filter server-side)
  const subscribe = useCallback((runId: string) => {
    wsRef.current?.send(JSON.stringify({ command: 'subscribe', run_id: runId }));
  }, []);

  const unsubscribe = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ command: 'unsubscribe' }));
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      clearTimeout(heartbeatTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { subscribe, unsubscribe };
}
