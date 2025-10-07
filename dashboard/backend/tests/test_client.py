"""Tests for TrainingMonitor client SDK."""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.client import TrainingMonitor


class TestTrainingMonitor:
    """Test TrainingMonitor client SDK."""

    def test_init_creates_run(self) -> None:
        """Test that initialization creates a run."""
        with patch("app.client.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.post.return_value.json.return_value = {
                "id": "run_123",
                "name": "Test Run",
                "status": "running",
            }
            
            monitor = TrainingMonitor(
                run_name="Test Run",
                config={"lr": 0.001},
                dashboard_url="http://localhost:8000"
            )
            
            # Verify run was created
            mock_client.post.assert_called_once()
            assert monitor.run_id == "run_123"
            assert monitor.run_name == "Test Run"

    def test_log_metrics(self) -> None:
        """Test logging metrics."""
        with patch("app.client.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.post.return_value.json.return_value = {
                "id": "run_123",
                "name": "Test Run",
                "status": "running",
            }
            
            monitor = TrainingMonitor(
                run_name="Test Run",
                dashboard_url="http://localhost:8000",
                batch_size=1  # Send immediately
            )
            
            # Log metrics
            monitor.log_metrics({"loss": 0.5, "accuracy": 0.85}, step=100)
            
            # Verify API call
            calls = mock_client.post.call_args_list
            # First call is create_run, second is log_metrics
            assert len(calls) == 2
            assert "/metrics" in str(calls[1])

    def test_save_checkpoint(self) -> None:
        """Test saving checkpoint."""
        with patch("app.client.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.post.return_value.json.return_value = {
                "id": "run_123",
                "name": "Test Run",
                "status": "running",
            }
            
            monitor = TrainingMonitor(
                run_name="Test Run",
                dashboard_url="http://localhost:8000"
            )
            
            # Save checkpoint
            monitor.save_checkpoint(
                filepath="/path/to/checkpoint.pt",
                metrics={"loss": 0.42}
            )
            
            # Verify API call
            calls = mock_client.post.call_args_list
            assert len(calls) == 2  # create_run + save_checkpoint
            assert "/checkpoints" in str(calls[1])

    def test_set_status(self) -> None:
        """Test updating run status."""
        with patch("app.client.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.post.return_value.json.return_value = {
                "id": "run_123",
                "name": "Test Run",
                "status": "running",
            }
            
            monitor = TrainingMonitor(
                run_name="Test Run",
                dashboard_url="http://localhost:8000"
            )
            
            # Update status
            monitor.set_status("completed")
            
            # Verify API call
            mock_client.put.assert_called_once()
            assert "/runs/run_123" in str(mock_client.put.call_args)

    def test_context_manager(self) -> None:
        """Test context manager support."""
        with patch("app.client.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.post.return_value.json.return_value = {
                "id": "run_123",
                "name": "Test Run",
                "status": "running",
            }
            
            with TrainingMonitor(
                run_name="Test Run",
                dashboard_url="http://localhost:8000"
            ) as monitor:
                assert monitor.run_id == "run_123"
            
            # Verify status was updated to completed on exit
            mock_client.put.assert_called_once()

    def test_connection_failure_fallback(self) -> None:
        """Test that monitor continues working if dashboard is unavailable."""
        with patch("app.client.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.post.side_effect = Exception("Connection refused")
            
            # Should not raise exception
            monitor = TrainingMonitor(
                run_name="Test Run",
                dashboard_url="http://localhost:8000"
            )
            
            # Should be able to log metrics (buffered)
            monitor.log_metrics({"loss": 0.5}, step=100)
            
            # Verify monitor is in offline mode
            assert monitor._offline_mode is True

    def test_metric_batching(self) -> None:
        """Test that metrics are batched before sending."""
        with patch("app.client.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.post.return_value.json.return_value = {
                "id": "run_123",
                "name": "Test Run",
                "status": "running",
            }
            
            monitor = TrainingMonitor(
                run_name="Test Run",
                dashboard_url="http://localhost:8000",
                batch_size=3  # Send every 3 metrics
            )
            
            # Log 2 metrics (should be buffered)
            monitor.log_metrics({"loss": 0.5}, step=1)
            monitor.log_metrics({"loss": 0.4}, step=2)
            
            # Only create_run call so far
            assert len(mock_client.post.call_args_list) == 1
            
            # Log 3rd metric (should trigger batch send)
            monitor.log_metrics({"loss": 0.3}, step=3)
            
            # Now should have 4 calls: create_run + 3 individual metrics
            # (Each metric is sent separately in current implementation)
            assert len(mock_client.post.call_args_list) == 4
            assert all("/metrics" in str(call) for call in mock_client.post.call_args_list[1:])

