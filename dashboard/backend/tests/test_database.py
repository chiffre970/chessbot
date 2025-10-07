"""Tests for database operations."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from app.database import Database
from app.models import Checkpoint, MetricPoint, Run, RunStatus


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    """Create a test database."""
    db_path = tmp_path / "test.db"
    database = Database(str(db_path))
    await database.initialize()
    return database


@pytest.fixture
async def sample_run(db: Database) -> Run:
    """Create a sample run for testing."""
    run = Run(
        id="test_run_001",
        name="Test Run",
        status=RunStatus.RUNNING,
        config={"lr": 0.001, "batch_size": 32},
        start_time=datetime.utcnow(),
    )
    await db.create_run(run)
    return run


class TestDatabase:
    """Test database operations."""

    async def test_initialize_creates_tables(self, tmp_path: Path) -> None:
        """Test that initialize creates all required tables."""
        db_path = tmp_path / "test.db"
        db = Database(str(db_path))
        await db.initialize()

        # Verify database file exists
        assert db_path.exists()

        # Verify tables exist
        async with db._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in await cursor.fetchall()]

        assert "runs" in tables
        assert "metrics" in tables
        assert "checkpoints" in tables
        assert "system_metrics" in tables


class TestRunOperations:
    """Test run CRUD operations."""

    async def test_create_run(self, db: Database) -> None:
        """Test creating a new run."""
        run = Run(
            id="run_001",
            name="Test Run",
            status=RunStatus.RUNNING,
            config={"lr": 0.001},
            start_time=datetime.utcnow(),
        )

        await db.create_run(run)

        # Verify run was created
        retrieved = await db.get_run(run.id)
        assert retrieved is not None
        assert retrieved.id == run.id
        assert retrieved.name == run.name
        assert retrieved.status == RunStatus.RUNNING

    async def test_get_run_not_found(self, db: Database) -> None:
        """Test getting a non-existent run returns None."""
        result = await db.get_run("nonexistent")
        assert result is None

    async def test_list_runs_empty(self, db: Database) -> None:
        """Test listing runs when database is empty."""
        runs = await db.list_runs()
        assert len(runs) == 0

    async def test_list_runs(self, db: Database) -> None:
        """Test listing multiple runs."""
        # Create multiple runs
        for i in range(3):
            run = Run(
                id=f"run_{i:03d}",
                name=f"Run {i}",
                status=RunStatus.RUNNING,
            )
            await db.create_run(run)

        runs = await db.list_runs()
        assert len(runs) == 3
        assert all(isinstance(r, Run) for r in runs)

    async def test_update_run_status(self, db: Database, sample_run: Run) -> None:
        """Test updating run status."""
        await db.update_run_status(sample_run.id, RunStatus.COMPLETED)

        updated = await db.get_run(sample_run.id)
        assert updated is not None
        assert updated.status == RunStatus.COMPLETED

    async def test_update_run_end_time(self, db: Database, sample_run: Run) -> None:
        """Test updating run end time."""
        end_time = datetime.utcnow()
        await db.update_run(sample_run.id, end_time=end_time)

        updated = await db.get_run(sample_run.id)
        assert updated is not None
        assert updated.end_time is not None

    async def test_delete_run(self, db: Database, sample_run: Run) -> None:
        """Test deleting a run."""
        await db.delete_run(sample_run.id)

        deleted = await db.get_run(sample_run.id)
        assert deleted is None


class TestMetricOperations:
    """Test metric operations."""

    async def test_log_metrics(self, db: Database, sample_run: Run) -> None:
        """Test logging metrics."""
        metric = MetricPoint(
            run_id=sample_run.id, step=100, metrics={"loss": 0.5, "accuracy": 0.85}
        )

        await db.log_metrics(metric)

        # Verify metrics were logged
        metrics = await db.get_metrics(sample_run.id)
        assert len(metrics) == 1
        assert metrics[0].step == 100
        assert metrics[0].metrics["loss"] == 0.5

    async def test_get_metrics_empty(self, db: Database, sample_run: Run) -> None:
        """Test getting metrics when none exist."""
        metrics = await db.get_metrics(sample_run.id)
        assert len(metrics) == 0

    async def test_get_latest_metrics(self, db: Database, sample_run: Run) -> None:
        """Test getting latest metrics."""
        # Log multiple metrics
        for step in [100, 200, 300]:
            metric = MetricPoint(
                run_id=sample_run.id, step=step, metrics={"loss": 1.0 / step}
            )
            await db.log_metrics(metric)

        latest = await db.get_latest_metrics(sample_run.id)
        assert latest is not None
        assert latest.step == 300

    async def test_get_metrics_with_limit(self, db: Database, sample_run: Run) -> None:
        """Test getting metrics with limit."""
        # Log 10 metrics
        for step in range(10):
            metric = MetricPoint(
                run_id=sample_run.id, step=step, metrics={"loss": 1.0}
            )
            await db.log_metrics(metric)

        # Get only last 5
        metrics = await db.get_metrics(sample_run.id, limit=5)
        assert len(metrics) == 5


class TestCheckpointOperations:
    """Test checkpoint operations."""

    async def test_create_checkpoint(self, db: Database, sample_run: Run) -> None:
        """Test creating a checkpoint."""
        checkpoint = Checkpoint(
            id="ckpt_001",
            run_id=sample_run.id,
            filepath="/path/to/checkpoint.pt",
            metrics={"loss": 0.42},
            metadata={"optimizer": "Adam"},
        )

        await db.create_checkpoint(checkpoint)

        # Verify checkpoint was created
        retrieved = await db.get_checkpoint(checkpoint.id)
        assert retrieved is not None
        assert retrieved.id == checkpoint.id
        assert retrieved.filepath == checkpoint.filepath

    async def test_list_checkpoints(self, db: Database, sample_run: Run) -> None:
        """Test listing checkpoints for a run."""
        # Create multiple checkpoints
        for i in range(3):
            checkpoint = Checkpoint(
                id=f"ckpt_{i:03d}",
                run_id=sample_run.id,
                filepath=f"/path/to/checkpoint_{i}.pt",
            )
            await db.create_checkpoint(checkpoint)

        checkpoints = await db.list_checkpoints(sample_run.id)
        assert len(checkpoints) == 3

    async def test_mark_checkpoint_as_best(self, db: Database, sample_run: Run) -> None:
        """Test marking a checkpoint as best."""
        # Create two checkpoints
        ckpt1 = Checkpoint(id="ckpt_001", run_id=sample_run.id, filepath="/path/1.pt")
        ckpt2 = Checkpoint(id="ckpt_002", run_id=sample_run.id, filepath="/path/2.pt")
        await db.create_checkpoint(ckpt1)
        await db.create_checkpoint(ckpt2)

        # Mark second as best
        await db.mark_checkpoint_as_best(ckpt2.id, sample_run.id)

        # Verify only ckpt2 is marked as best
        retrieved1 = await db.get_checkpoint(ckpt1.id)
        retrieved2 = await db.get_checkpoint(ckpt2.id)
        assert retrieved1 is not None and not retrieved1.is_best
        assert retrieved2 is not None and retrieved2.is_best

    async def test_delete_checkpoint(self, db: Database, sample_run: Run) -> None:
        """Test deleting a checkpoint."""
        checkpoint = Checkpoint(
            id="ckpt_001", run_id=sample_run.id, filepath="/path/to/checkpoint.pt"
        )
        await db.create_checkpoint(checkpoint)

        await db.delete_checkpoint(checkpoint.id)

        deleted = await db.get_checkpoint(checkpoint.id)
        assert deleted is None


class TestSystemMetrics:
    """Test system metrics operations."""

    async def test_log_system_metrics(self, db: Database) -> None:
        """Test logging system metrics."""
        from app.models import SystemMetrics

        metrics = SystemMetrics(
            cpu_percent=45.5,
            memory_percent=60.2,
            gpu_memory_used=8000,
            gpu_memory_total=16000,
            gpu_utilization=75.0,
        )

        await db.log_system_metrics(metrics)

        # Verify metrics were logged
        history = await db.get_system_metrics_history(limit=10)
        assert len(history) == 1
        assert history[0].cpu_percent == 45.5

    async def test_get_latest_system_metrics(self, db: Database) -> None:
        """Test getting latest system metrics."""
        from app.models import SystemMetrics

        # Log multiple metrics
        for i in range(3):
            metrics = SystemMetrics(cpu_percent=float(i * 10), memory_percent=50.0)
            await db.log_system_metrics(metrics)

        latest = await db.get_latest_system_metrics()
        assert latest is not None
        assert latest.cpu_percent == 20.0


