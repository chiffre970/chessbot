"""WebSocket connection manager for real-time updates."""

import asyncio
from typing import Dict, List, Set

import structlog
from fastapi import WebSocket

logger = structlog.get_logger()


class ConnectionManager:
    """Manage WebSocket connections and broadcast messages."""

    def __init__(self) -> None:
        """Initialize connection manager."""
        # Active connections: {websocket: set of subscribed run_ids}
        self.active_connections: Dict[WebSocket, Set[str]] = {}
        # Reverse mapping: {run_id: set of websockets}
        self.run_subscriptions: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection.

        Args:
            websocket: WebSocket connection to accept
        """
        await websocket.accept()
        async with self._lock:
            self.active_connections[websocket] = set()
        logger.info("websocket_connected", total_connections=len(self.active_connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        """Disconnect a WebSocket and clean up subscriptions.

        Args:
            websocket: WebSocket connection to disconnect
        """
        async with self._lock:
            if websocket in self.active_connections:
                # Get all run_ids this websocket was subscribed to
                run_ids = self.active_connections[websocket]
                
                # Remove from run subscriptions
                for run_id in run_ids:
                    if run_id in self.run_subscriptions:
                        self.run_subscriptions[run_id].discard(websocket)
                        if not self.run_subscriptions[run_id]:
                            del self.run_subscriptions[run_id]
                
                # Remove connection
                del self.active_connections[websocket]
        
        logger.info("websocket_disconnected", total_connections=len(self.active_connections))

    async def subscribe(self, websocket: WebSocket, run_id: str) -> None:
        """Subscribe a websocket to updates for a specific run.

        Args:
            websocket: WebSocket connection
            run_id: Run ID to subscribe to
        """
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections[websocket].add(run_id)
                
                if run_id not in self.run_subscriptions:
                    self.run_subscriptions[run_id] = set()
                self.run_subscriptions[run_id].add(websocket)
        
        logger.debug("websocket_subscribed", run_id=run_id)

    async def unsubscribe(self, websocket: WebSocket, run_id: str) -> None:
        """Unsubscribe a websocket from updates for a specific run.

        Args:
            websocket: WebSocket connection
            run_id: Run ID to unsubscribe from
        """
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections[websocket].discard(run_id)
            
            if run_id in self.run_subscriptions:
                self.run_subscriptions[run_id].discard(websocket)
                if not self.run_subscriptions[run_id]:
                    del self.run_subscriptions[run_id]
        
        logger.debug("websocket_unsubscribed", run_id=run_id)

    async def broadcast_to_run(self, run_id: str, message: dict) -> None:
        """Broadcast a message to all subscribers of a specific run.

        Args:
            run_id: Run ID to broadcast to
            message: Message dictionary to send
        """
        async with self._lock:
            subscribers = self.run_subscriptions.get(run_id, set()).copy()
        
        if subscribers:
            # Send to all subscribers concurrently
            await asyncio.gather(
                *[self._send_json(ws, message) for ws in subscribers],
                return_exceptions=True
            )
            logger.debug(
                "message_broadcasted",
                run_id=run_id,
                message_type=message.get("type"),
                num_subscribers=len(subscribers)
            )

    async def _send_json(self, websocket: WebSocket, message: dict) -> None:
        """Send a JSON message to a websocket.

        Args:
            websocket: WebSocket to send to
            message: Message dictionary
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error("websocket_send_failed", error=str(e))
            # Remove failed connection
            await self.disconnect(websocket)


