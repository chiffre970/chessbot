"""Tests for REST API endpoints."""

from datetime import datetime
from typing import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api import create_app
from app.database import Database
from app.models import Run, RunStatus


@pytest.fixture
async def app(tmp_path) -> FastAPI:
    """Create test FastAPI application."""
    from app.websocket import ConnectionManager
    
    db_path = str(tmp_path / "test.db")
    application = create_app(db_path)
    
    # Manually initialize database and websocket manager for testing
    # (lifespan is not called in test client)
    db = Database(db_path)
    await db.initialize()
    application.state.db = db
    application.state.ws_manager = ConnectionManager()
    
    return application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """Create test HTTP client."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


class TestHealthEndpoints:
    """Test health and version endpoints."""

    async def test_health_check(self, client: AsyncClient) -> None:
        """Test health check endpoint."""
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    async def test_version(self, client: AsyncClient) -> None:
        """Test API version endpoint."""
        response = await client.get("/api/version")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data


class TestRunEndpoints:
    """Test run management endpoints."""

    async def test_create_run(self, client: AsyncClient) -> None:
        """Test creating a new run."""
        response = await client.post(
            "/api/runs", json={"name": "Test Run", "config": {"lr": 0.001}}
        )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["name"] == "Test Run"
        assert data["status"] == "running"
        assert data["config"]["lr"] == 0.001

    async def test_create_run_minimal(self, client: AsyncClient) -> None:
        """Test creating a run with minimal data."""
        response = await client.post("/api/runs", json={"name": "Minimal Run"})

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Minimal Run"
        assert data["config"] is None

    async def test_get_run(self, client: AsyncClient) -> None:
        """Test getting a run by ID."""
        # Create a run first
        create_response = await client.post(
            "/api/runs", json={"name": "Test Run"}
        )
        run_id = create_response.json()["id"]

        # Get the run
        response = await client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == run_id

    async def test_get_run_not_found(self, client: AsyncClient) -> None:
        """Test getting a non-existent run returns 404."""
        response = await client.get("/api/runs/nonexistent")
        assert response.status_code == 404

    async def test_list_runs(self, client: AsyncClient) -> None:
        """Test listing runs."""
        # Create multiple runs
        for i in range(3):
            await client.post("/api/runs", json={"name": f"Run {i}"})

        # List runs
        response = await client.get("/api/runs")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3

    async def test_list_runs_empty(self, client: AsyncClient) -> None:
        """Test listing runs when none exist."""
        response = await client.get("/api/runs")
        assert response.status_code == 200
        assert response.json() == []

    async def test_update_run_status(self, client: AsyncClient) -> None:
        """Test updating run status."""
        # Create a run
        create_response = await client.post(
            "/api/runs", json={"name": "Test Run"}
        )
        run_id = create_response.json()["id"]

        # Update status
        response = await client.put(
            f"/api/runs/{run_id}", json={"status": "completed"}
        )
        assert response.status_code == 200

        # Verify update
        get_response = await client.get(f"/api/runs/{run_id}")
        assert get_response.json()["status"] == "completed"

    async def test_delete_run(self, client: AsyncClient) -> None:
        """Test deleting a run."""
        # Create a run
        create_response = await client.post(
            "/api/runs", json={"name": "Test Run"}
        )
        run_id = create_response.json()["id"]

        # Delete the run
        response = await client.delete(f"/api/runs/{run_id}")
        assert response.status_code == 200

        # Verify deletion
        get_response = await client.get(f"/api/runs/{run_id}")
        assert get_response.status_code == 404


class TestMetricEndpoints:
    """Test metric endpoints."""

    @pytest.fixture
    async def run_id(self, client: AsyncClient) -> str:
        """Create a test run and return its ID."""
        response = await client.post("/api/runs", json={"name": "Test Run"})
        return response.json()["id"]

    async def test_log_metrics(self, client: AsyncClient, run_id: str) -> None:
        """Test logging metrics."""
        response = await client.post(
            f"/api/runs/{run_id}/metrics",
            json={"step": 100, "metrics": {"loss": 0.5, "accuracy": 0.85}},
        )

        assert response.status_code == 200

    async def test_get_metrics(self, client: AsyncClient, run_id: str) -> None:
        """Test getting metrics."""
        # Log some metrics first
        await client.post(
            f"/api/runs/{run_id}/metrics",
            json={"step": 100, "metrics": {"loss": 0.5}},
        )

        # Get metrics
        response = await client.get(f"/api/runs/{run_id}/metrics")
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
        assert data[0]["step"] == 100

    async def test_get_metrics_empty(self, client: AsyncClient, run_id: str) -> None:
        """Test getting metrics when none exist."""
        response = await client.get(f"/api/runs/{run_id}/metrics")
        assert response.status_code == 200
        assert response.json() == []

    async def test_get_latest_metrics(
        self, client: AsyncClient, run_id: str
    ) -> None:
        """Test getting latest metrics."""
        # Log multiple metrics
        for step in [100, 200, 300]:
            await client.post(
                f"/api/runs/{run_id}/metrics",
                json={"step": step, "metrics": {"loss": 1.0 / step}},
            )

        # Get latest
        response = await client.get(f"/api/runs/{run_id}/metrics/latest")
        assert response.status_code == 200
        data = response.json()
        assert data["step"] == 300

    async def test_get_latest_metrics_none(
        self, client: AsyncClient, run_id: str
    ) -> None:
        """Test getting latest metrics when none exist."""
        response = await client.get(f"/api/runs/{run_id}/metrics/latest")
        assert response.status_code == 404


class TestCheckpointEndpoints:
    """Test checkpoint endpoints."""

    @pytest.fixture
    async def run_id(self, client: AsyncClient) -> str:
        """Create a test run and return its ID."""
        response = await client.post("/api/runs", json={"name": "Test Run"})
        return response.json()["id"]

    async def test_create_checkpoint(
        self, client: AsyncClient, run_id: str
    ) -> None:
        """Test registering a checkpoint."""
        response = await client.post(
            f"/api/runs/{run_id}/checkpoints",
            json={
                "filepath": "/path/to/checkpoint.pt",
                "metrics": {"loss": 0.42},
                "metadata": {"optimizer": "Adam"},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["filepath"] == "/path/to/checkpoint.pt"

    async def test_list_checkpoints(
        self, client: AsyncClient, run_id: str
    ) -> None:
        """Test listing checkpoints."""
        # Create multiple checkpoints
        for i in range(3):
            await client.post(
                f"/api/runs/{run_id}/checkpoints",
                json={"filepath": f"/path/to/checkpoint_{i}.pt"},
            )

        # List checkpoints
        response = await client.get(f"/api/runs/{run_id}/checkpoints")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3

    async def test_get_checkpoint(self, client: AsyncClient, run_id: str) -> None:
        """Test getting a checkpoint by ID."""
        # Create checkpoint
        create_response = await client.post(
            f"/api/runs/{run_id}/checkpoints",
            json={"filepath": "/path/to/checkpoint.pt"},
        )
        checkpoint_id = create_response.json()["id"]

        # Get checkpoint
        response = await client.get(f"/api/checkpoints/{checkpoint_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == checkpoint_id

    async def test_delete_checkpoint(
        self, client: AsyncClient, run_id: str
    ) -> None:
        """Test deleting a checkpoint."""
        # Create checkpoint
        create_response = await client.post(
            f"/api/runs/{run_id}/checkpoints",
            json={"filepath": "/path/to/checkpoint.pt"},
        )
        checkpoint_id = create_response.json()["id"]

        # Delete checkpoint
        response = await client.delete(f"/api/checkpoints/{checkpoint_id}")
        assert response.status_code == 200

        # Verify deletion
        get_response = await client.get(f"/api/checkpoints/{checkpoint_id}")
        assert get_response.status_code == 404

    async def test_mark_checkpoint_as_best(
        self, client: AsyncClient, run_id: str
    ) -> None:
        """Test marking a checkpoint as best."""
        # Create two checkpoints
        ckpt1_response = await client.post(
            f"/api/runs/{run_id}/checkpoints",
            json={"filepath": "/path/1.pt"},
        )
        ckpt2_response = await client.post(
            f"/api/runs/{run_id}/checkpoints",
            json={"filepath": "/path/2.pt"},
        )

        checkpoint_id = ckpt2_response.json()["id"]

        # Mark as best
        response = await client.post(
            f"/api/checkpoints/{checkpoint_id}/mark_best"
        )
        assert response.status_code == 200

        # Verify
        get_response = await client.get(f"/api/checkpoints/{checkpoint_id}")
        assert get_response.json()["is_best"] is True


class TestSystemMetrics:
    """Test system metrics endpoints."""

    async def test_get_system_status(self, client: AsyncClient) -> None:
        """Test getting current system status."""
        response = await client.get("/api/system/status")
        assert response.status_code == 200
        data = response.json()
        assert "cpu_percent" in data
        assert "memory_percent" in data

    async def test_get_system_history(self, client: AsyncClient) -> None:
        """Test getting system metrics history."""
        response = await client.get("/api/system/history")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

