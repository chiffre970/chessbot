"""Tests for WebSocket functionality."""

from typing import AsyncIterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import create_app
from app.database import Database


@pytest.fixture
async def app(tmp_path) -> FastAPI:
    """Create test FastAPI application."""
    from app.websocket import ConnectionManager
    
    db_path = str(tmp_path / "test.db")
    application = create_app(db_path)
    
    # Manually initialize database and websocket manager for testing
    db = Database(db_path)
    await db.initialize()
    application.state.db = db
    application.state.ws_manager = ConnectionManager()
    
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create test client."""
    return TestClient(app)


class TestWebSocketConnection:
    """Test WebSocket connection and protocol."""

    def test_websocket_connect(self, client: TestClient) -> None:
        """Test WebSocket connection."""
        with client.websocket_connect("/ws") as websocket:
            # Connection established
            assert websocket is not None

    def test_websocket_ping_pong(self, client: TestClient) -> None:
        """Test WebSocket ping/pong heartbeat."""
        with client.websocket_connect("/ws") as websocket:
            # Send ping
            websocket.send_json({"type": "ping"})
            
            # Receive pong
            response = websocket.receive_json()
            assert response["type"] == "pong"

    def test_websocket_subscribe_to_run(self, client: TestClient) -> None:
        """Test subscribing to run updates."""
        # First, create a run
        run_response = client.post("/api/runs", json={"name": "Test Run"})
        run_id = run_response.json()["id"]

        with client.websocket_connect("/ws") as websocket:
            # Subscribe to run
            websocket.send_json({"type": "subscribe", "run_id": run_id})
            
            # Receive confirmation
            response = websocket.receive_json()
            assert response["type"] == "subscribed"
            assert response["run_id"] == run_id

    def test_websocket_unsubscribe_from_run(self, client: TestClient) -> None:
        """Test unsubscribing from run updates."""
        # Create a run
        run_response = client.post("/api/runs", json={"name": "Test Run"})
        run_id = run_response.json()["id"]

        with client.websocket_connect("/ws") as websocket:
            # Subscribe first
            websocket.send_json({"type": "subscribe", "run_id": run_id})
            websocket.receive_json()  # subscribed confirmation
            
            # Unsubscribe
            websocket.send_json({"type": "unsubscribe", "run_id": run_id})
            
            # Receive confirmation
            response = websocket.receive_json()
            assert response["type"] == "unsubscribed"
            assert response["run_id"] == run_id

    def test_websocket_metric_update_broadcast(self, client: TestClient) -> None:
        """Test that metric updates are broadcasted to subscribers."""
        # Create a run
        run_response = client.post("/api/runs", json={"name": "Test Run"})
        run_id = run_response.json()["id"]

        with client.websocket_connect("/ws") as websocket:
            # Subscribe to run
            websocket.send_json({"type": "subscribe", "run_id": run_id})
            websocket.receive_json()  # subscribed confirmation
            
            # Log metrics via REST API
            client.post(
                f"/api/runs/{run_id}/metrics",
                json={"step": 100, "metrics": {"loss": 0.5}},
            )
            
            # Should receive metric update via WebSocket
            response = websocket.receive_json()
            assert response["type"] == "metric_update"
            assert response["run_id"] == run_id
            assert response["step"] == 100
            assert response["metrics"]["loss"] == 0.5

    def test_websocket_status_change_broadcast(self, client: TestClient) -> None:
        """Test that status changes are broadcasted to subscribers."""
        # Create a run
        run_response = client.post("/api/runs", json={"name": "Test Run"})
        run_id = run_response.json()["id"]

        with client.websocket_connect("/ws") as websocket:
            # Subscribe to run
            websocket.send_json({"type": "subscribe", "run_id": run_id})
            websocket.receive_json()  # subscribed confirmation
            
            # Update status via REST API
            client.put(f"/api/runs/{run_id}", json={"status": "completed"})
            
            # Should receive status change via WebSocket
            response = websocket.receive_json()
            assert response["type"] == "status_change"
            assert response["run_id"] == run_id
            assert response["new_status"] == "completed"

    def test_websocket_checkpoint_saved_broadcast(self, client: TestClient) -> None:
        """Test that checkpoint saves are broadcasted to subscribers."""
        # Create a run
        run_response = client.post("/api/runs", json={"name": "Test Run"})
        run_id = run_response.json()["id"]

        with client.websocket_connect("/ws") as websocket:
            # Subscribe to run
            websocket.send_json({"type": "subscribe", "run_id": run_id})
            websocket.receive_json()  # subscribed confirmation
            
            # Create checkpoint via REST API
            client.post(
                f"/api/runs/{run_id}/checkpoints",
                json={"filepath": "/path/to/checkpoint.pt", "metrics": {"loss": 0.42}},
            )
            
            # Should receive checkpoint notification via WebSocket
            response = websocket.receive_json()
            assert response["type"] == "checkpoint_saved"
            assert response["run_id"] == run_id
            assert "checkpoint_id" in response

    def test_websocket_multiple_subscribers(self, client: TestClient) -> None:
        """Test that updates are sent to all subscribers."""
        # Create a run
        run_response = client.post("/api/runs", json={"name": "Test Run"})
        run_id = run_response.json()["id"]

        # Create two WebSocket connections
        with client.websocket_connect("/ws") as ws1, client.websocket_connect("/ws") as ws2:
            # Both subscribe to the same run
            ws1.send_json({"type": "subscribe", "run_id": run_id})
            ws2.send_json({"type": "subscribe", "run_id": run_id})
            ws1.receive_json()  # confirmation
            ws2.receive_json()  # confirmation
            
            # Log metrics
            client.post(
                f"/api/runs/{run_id}/metrics",
                json={"step": 100, "metrics": {"loss": 0.5}},
            )
            
            # Both should receive the update
            update1 = ws1.receive_json()
            update2 = ws2.receive_json()
            
            assert update1["type"] == "metric_update"
            assert update2["type"] == "metric_update"
            assert update1["step"] == 100
            assert update2["step"] == 100

    def test_websocket_invalid_message(self, client: TestClient) -> None:
        """Test handling of invalid WebSocket messages."""
        with client.websocket_connect("/ws") as websocket:
            # Send invalid message
            websocket.send_json({"type": "invalid_type"})
            
            # Should receive error
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert "message" in response

