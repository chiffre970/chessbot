"""Training Monitor SDK for logging to the dashboard."""

import atexit
import threading
import time
from queue import Queue
from typing import Any, Dict, List, Optional

import httpx
import structlog

from app.models import RunStatus

logger = structlog.get_logger()


class TrainingMonitor:
    """Client SDK for logging training metrics to the dashboard.

    Example usage:
        ```python
        monitor = TrainingMonitor(
            run_name="my_experiment",
            config={"lr": 0.001, "batch_size": 32}
        )

        for step in range(1000):
            loss = train_step()
            monitor.log_metrics({"loss": loss}, step=step)

        monitor.set_status("completed")
        ```

    Context manager usage:
        ```python
        with TrainingMonitor(run_name="experiment") as monitor:
            for step in range(1000):
                loss = train_step()
                monitor.log_metrics({"loss": loss}, step=step)
            # Automatically marks as completed
        ```
    """

    def __init__(
        self,
        run_name: str,
        config: Optional[Dict[str, Any]] = None,
        dashboard_url: str = "http://localhost:8000",
        batch_size: int = 10,
        flush_interval: float = 1.0,
    ) -> None:
        """Initialize the training monitor.

        Args:
            run_name: Name of the training run
            config: Optional configuration dictionary
            dashboard_url: URL of the dashboard backend
            batch_size: Number of metrics to batch before sending
            flush_interval: Interval in seconds to flush buffered metrics
        """
        self.run_name = run_name
        self.config = config
        self.dashboard_url = dashboard_url.rstrip("/")
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        self.run_id: Optional[str] = None
        self._offline_mode = False
        self._metric_buffer: List[Dict[str, Any]] = []
        self._metric_lock = threading.Lock()
        
        # HTTP client
        self._client = httpx.Client(timeout=5.0)

        # Background flush thread
        self._flush_thread: Optional[threading.Thread] = None
        self._stop_flush = threading.Event()

        # Create run
        self._create_run()

        # Start background metrics flusher
        if not self._offline_mode:
            self._start_flush_thread()

        # Register cleanup on exit
        atexit.register(self._cleanup)

    def _create_run(self) -> None:
        """Create a new training run in the dashboard."""
        try:
            response = self._client.post(
                f"{self.dashboard_url}/api/runs",
                json={"name": self.run_name, "config": self.config},
            )
            response.raise_for_status()
            data = response.json()
            self.run_id = data["id"]
            logger.info("training_run_created", run_id=self.run_id, name=self.run_name)
        except Exception as e:
            logger.warning(
                "failed_to_connect_to_dashboard",
                error=str(e),
                message="Continuing in offline mode",
            )
            self._offline_mode = True
            self.run_id = f"offline_{int(time.time())}"

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics for the current step.

        Args:
            metrics: Dictionary of metric names to values
            step: Training step number
        """
        if self._offline_mode:
            return

        with self._metric_lock:
            self._metric_buffer.append({"step": step, "metrics": metrics})

            # Send immediately if buffer is full
            if len(self._metric_buffer) >= self.batch_size:
                self._flush_metrics()

    def _flush_metrics(self) -> None:
        """Flush buffered metrics to the dashboard."""
        if not self._metric_buffer or self._offline_mode:
            return

        metrics_to_send = self._metric_buffer.copy()
        self._metric_buffer.clear()

        try:
            # Send each metric (could be optimized to batch endpoint later)
            for metric_data in metrics_to_send:
                self._client.post(
                    f"{self.dashboard_url}/api/runs/{self.run_id}/metrics",
                    json=metric_data,
                )
            logger.debug("metrics_flushed", count=len(metrics_to_send))
        except Exception as e:
            logger.error("failed_to_send_metrics", error=str(e))
            # Re-add to buffer for retry (limited to avoid memory issues)
            with self._metric_lock:
                self._metric_buffer = metrics_to_send[-100:] + self._metric_buffer

    def _start_flush_thread(self) -> None:
        """Start background thread for periodic metric flushing."""

        def flush_loop() -> None:
            while not self._stop_flush.is_set():
                time.sleep(self.flush_interval)
                with self._metric_lock:
                    if self._metric_buffer:
                        self._flush_metrics()

        self._flush_thread = threading.Thread(target=flush_loop, daemon=True)
        self._flush_thread.start()

    def save_checkpoint(
        self,
        filepath: str,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a checkpoint with the dashboard.

        Args:
            filepath: Path to the checkpoint file
            metrics: Optional metrics at this checkpoint
            metadata: Optional metadata (optimizer state, etc.)
        """
        if self._offline_mode:
            return

        try:
            self._client.post(
                f"{self.dashboard_url}/api/runs/{self.run_id}/checkpoints",
                json={
                    "filepath": filepath,
                    "metrics": metrics,
                    "metadata": metadata,
                },
            )
            logger.info("checkpoint_registered", filepath=filepath)
        except Exception as e:
            logger.error("failed_to_register_checkpoint", error=str(e))

    def set_status(self, status: str) -> None:
        """Update the run status.

        Args:
            status: New status ("running", "completed", "failed", "interrupted")
        """
        if self._offline_mode:
            return

        try:
            self._client.put(
                f"{self.dashboard_url}/api/runs/{self.run_id}",
                json={"status": status},
            )
            logger.info("run_status_updated", status=status)
        except Exception as e:
            logger.error("failed_to_update_status", error=str(e))

    def _cleanup(self) -> None:
        """Clean up resources."""
        # Stop flush thread
        if self._flush_thread and self._flush_thread.is_alive():
            self._stop_flush.set()
            self._flush_thread.join(timeout=2.0)

        # Flush remaining metrics
        with self._metric_lock:
            if self._metric_buffer:
                self._flush_metrics()

        # Close HTTP client
        self._client.close()

    def __enter__(self) -> "TrainingMonitor":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        # Update status based on whether an exception occurred
        if exc_type is None:
            self.set_status("completed")
        else:
            self.set_status("failed")

        self._cleanup()


