"""Database operations for the training dashboard."""

import json
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import aiosqlite

from app.models import Checkpoint, MetricPoint, Run, RunStatus, SystemMetrics


class Database:
    """SQLite database for training dashboard."""

    def __init__(self, db_path: str) -> None:
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Ensure directory exists
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def _get_connection(self) -> AsyncIterator[aiosqlite.Connection]:
        """Get a database connection with proper configuration."""
        async with aiosqlite.connect(self.db_path) as conn:
            # Enable WAL mode for better concurrency
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA synchronous=NORMAL")
            # Enable foreign keys
            await conn.execute("PRAGMA foreign_keys=ON")
            # Return row factory for dict-like access
            conn.row_factory = aiosqlite.Row
            yield conn

    async def initialize(self) -> None:
        """Create database schema."""
        async with self._get_connection() as conn:
            # Runs table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config TEXT,
                    start_time DATETIME,
                    end_time DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Metrics table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    step INTEGER NOT NULL,
                    metrics TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
                )
            """)

            # Create index for efficient metric queries
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_run_step 
                ON metrics(run_id, step)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                ON metrics(timestamp DESC)
            """)

            # Checkpoints table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    metrics TEXT,
                    metadata TEXT,
                    is_best BOOLEAN DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
                )
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_checkpoints_run 
                ON checkpoints(run_id, created_at DESC)
            """)

            # System metrics table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    cpu_percent REAL,
                    memory_percent REAL,
                    gpu_memory_used INTEGER,
                    gpu_memory_total INTEGER,
                    gpu_utilization REAL
                )
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp 
                ON system_metrics(timestamp DESC)
            """)

            await conn.commit()

    # Run operations

    async def create_run(self, run: Run) -> None:
        """Create a new training run."""
        async with self._get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO runs (id, name, status, config, start_time, end_time, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.name,
                    run.status.value,
                    json.dumps(run.config) if run.config else None,
                    run.start_time.isoformat() if run.start_time else None,
                    run.end_time.isoformat() if run.end_time else None,
                    run.created_at.isoformat(),
                ),
            )
            await conn.commit()

    async def get_run(self, run_id: str) -> Optional[Run]:
        """Get a run by ID."""
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            )
            row = await cursor.fetchone()

            if row is None:
                return None

            return self._row_to_run(dict(row))

    async def list_runs(self, limit: int = 100) -> List[Run]:
        """List all runs, most recent first."""
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            )
            rows = await cursor.fetchall()
            return [self._row_to_run(dict(row)) for row in rows]

    async def update_run_status(self, run_id: str, status: RunStatus) -> None:
        """Update run status."""
        async with self._get_connection() as conn:
            await conn.execute(
                "UPDATE runs SET status = ? WHERE id = ?",
                (status.value, run_id),
            )
            await conn.commit()

    async def update_run(
        self, run_id: str, end_time: Optional[datetime] = None
    ) -> None:
        """Update run fields."""
        async with self._get_connection() as conn:
            if end_time:
                await conn.execute(
                    "UPDATE runs SET end_time = ? WHERE id = ?",
                    (end_time.isoformat(), run_id),
                )
            await conn.commit()

    async def delete_run(self, run_id: str) -> None:
        """Delete a run and all associated data."""
        async with self._get_connection() as conn:
            await conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            await conn.commit()

    # Metric operations

    async def log_metrics(self, metric: MetricPoint) -> None:
        """Log metrics for a run."""
        async with self._get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO metrics (run_id, step, metrics, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (
                    metric.run_id,
                    metric.step,
                    json.dumps(metric.metrics),
                    metric.timestamp.isoformat(),
                ),
            )
            await conn.commit()

    async def get_metrics(
        self, run_id: str, limit: Optional[int] = None
    ) -> List[MetricPoint]:
        """Get metrics for a run."""
        query = "SELECT * FROM metrics WHERE run_id = ? ORDER BY step ASC"
        params: tuple[Any, ...] = (run_id,)

        if limit:
            query += " LIMIT ?"
            params = (run_id, limit)

        async with self._get_connection() as conn:
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()
            return [self._row_to_metric(dict(row)) for row in rows]

    async def get_latest_metrics(self, run_id: str) -> Optional[MetricPoint]:
        """Get the latest metrics for a run."""
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT * FROM metrics 
                WHERE run_id = ? 
                ORDER BY step DESC 
                LIMIT 1
                """,
                (run_id,),
            )
            row = await cursor.fetchone()

            if row is None:
                return None

            return self._row_to_metric(dict(row))

    # Checkpoint operations

    async def create_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Register a new checkpoint."""
        async with self._get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO checkpoints 
                (id, run_id, filepath, metrics, metadata, is_best, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint.id,
                    checkpoint.run_id,
                    checkpoint.filepath,
                    json.dumps(checkpoint.metrics) if checkpoint.metrics else None,
                    json.dumps(checkpoint.metadata) if checkpoint.metadata else None,
                    checkpoint.is_best,
                    checkpoint.created_at.isoformat(),
                ),
            )
            await conn.commit()

    async def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get a checkpoint by ID."""
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM checkpoints WHERE id = ?", (checkpoint_id,)
            )
            row = await cursor.fetchone()

            if row is None:
                return None

            return self._row_to_checkpoint(dict(row))

    async def list_checkpoints(self, run_id: str) -> List[Checkpoint]:
        """List all checkpoints for a run."""
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT * FROM checkpoints 
                WHERE run_id = ? 
                ORDER BY created_at DESC
                """,
                (run_id,),
            )
            rows = await cursor.fetchall()
            return [self._row_to_checkpoint(dict(row)) for row in rows]

    async def mark_checkpoint_as_best(self, checkpoint_id: str, run_id: str) -> None:
        """Mark a checkpoint as best, unmarking all others for the run."""
        async with self._get_connection() as conn:
            # Unmark all checkpoints for this run
            await conn.execute(
                "UPDATE checkpoints SET is_best = 0 WHERE run_id = ?", (run_id,)
            )
            # Mark the specified checkpoint
            await conn.execute(
                "UPDATE checkpoints SET is_best = 1 WHERE id = ?", (checkpoint_id,)
            )
            await conn.commit()

    async def delete_checkpoint(self, checkpoint_id: str) -> None:
        """Delete a checkpoint."""
        async with self._get_connection() as conn:
            await conn.execute("DELETE FROM checkpoints WHERE id = ?", (checkpoint_id,))
            await conn.commit()

    # System metrics operations

    async def log_system_metrics(self, metrics: SystemMetrics) -> None:
        """Log system resource metrics."""
        async with self._get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO system_metrics 
                (cpu_percent, memory_percent, gpu_memory_used, gpu_memory_total, 
                 gpu_utilization, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    metrics.cpu_percent,
                    metrics.memory_percent,
                    metrics.gpu_memory_used,
                    metrics.gpu_memory_total,
                    metrics.gpu_utilization,
                    metrics.timestamp.isoformat(),
                ),
            )
            await conn.commit()

    async def get_latest_system_metrics(self) -> Optional[SystemMetrics]:
        """Get the latest system metrics."""
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM system_metrics ORDER BY timestamp DESC LIMIT 1"
            )
            row = await cursor.fetchone()

            if row is None:
                return None

            return self._row_to_system_metrics(dict(row))

    async def get_system_metrics_history(
        self, limit: int = 100
    ) -> List[SystemMetrics]:
        """Get historical system metrics."""
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM system_metrics ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
            return [self._row_to_system_metrics(dict(row)) for row in rows]

    # Helper methods for row conversion

    def _row_to_run(self, row: Dict[str, Any]) -> Run:
        """Convert database row to Run model."""
        return Run(
            id=row["id"],
            name=row["name"],
            status=RunStatus(row["status"]),
            config=json.loads(row["config"]) if row["config"] else None,
            start_time=datetime.fromisoformat(row["start_time"])
            if row["start_time"]
            else None,
            end_time=datetime.fromisoformat(row["end_time"])
            if row["end_time"]
            else None,
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _row_to_metric(self, row: Dict[str, Any]) -> MetricPoint:
        """Convert database row to MetricPoint model."""
        return MetricPoint(
            run_id=row["run_id"],
            step=row["step"],
            metrics=json.loads(row["metrics"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )

    def _row_to_checkpoint(self, row: Dict[str, Any]) -> Checkpoint:
        """Convert database row to Checkpoint model."""
        return Checkpoint(
            id=row["id"],
            run_id=row["run_id"],
            filepath=row["filepath"],
            metrics=json.loads(row["metrics"]) if row["metrics"] else None,
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            is_best=bool(row["is_best"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _row_to_system_metrics(self, row: Dict[str, Any]) -> SystemMetrics:
        """Convert database row to SystemMetrics model."""
        return SystemMetrics(
            cpu_percent=row["cpu_percent"],
            memory_percent=row["memory_percent"],
            gpu_memory_used=row["gpu_memory_used"],
            gpu_memory_total=row["gpu_memory_total"],
            gpu_utilization=row["gpu_utilization"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )

