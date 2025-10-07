"""FastAPI application for training dashboard."""

import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator, List, Optional

import psutil
import structlog
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.database import Database
from app.websocket import ConnectionManager
from app.models import (
    Checkpoint,
    CreateCheckpointRequest,
    CreateRunRequest,
    LogMetricsRequest,
    MetricPoint,
    Run,
    RunStatus,
    SystemMetrics,
    UpdateRunRequest,
)

logger = structlog.get_logger()


def create_app(db_path: str = "data/dashboard.db") -> FastAPI:
    """Create and configure FastAPI application.

    Args:
        db_path: Path to SQLite database

    Returns:
        Configured FastAPI application
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Application lifespan manager."""
        # Initialize database
        app.state.db = Database(db_path)
        await app.state.db.initialize()
        logger.info("database_initialized", db_path=db_path)
        
        # Initialize WebSocket manager
        app.state.ws_manager = ConnectionManager()
        logger.info("websocket_manager_initialized")
        
        yield
        logger.info("shutdown")

    app = FastAPI(
        title="Training Dashboard API",
        version=__version__,
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, restrict this
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health endpoints

    @app.get("/health")
    async def health_check() -> dict:
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.get("/api/version")
    async def get_version() -> dict:
        """Get API version."""
        return {"version": __version__}

    # WebSocket endpoint

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        """WebSocket endpoint for real-time updates."""
        manager: ConnectionManager = app.state.ws_manager
        await manager.connect(websocket)
        
        try:
            while True:
                # Receive message from client
                data = await websocket.receive_json()
                message_type = data.get("type")
                
                if message_type == "ping":
                    # Respond to heartbeat
                    await websocket.send_json({"type": "pong"})
                
                elif message_type == "subscribe":
                    # Subscribe to run updates
                    run_id = data.get("run_id")
                    if run_id:
                        await manager.subscribe(websocket, run_id)
                        await websocket.send_json({
                            "type": "subscribed",
                            "run_id": run_id
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Missing run_id"
                        })
                
                elif message_type == "unsubscribe":
                    # Unsubscribe from run updates
                    run_id = data.get("run_id")
                    if run_id:
                        await manager.unsubscribe(websocket, run_id)
                        await websocket.send_json({
                            "type": "unsubscribed",
                            "run_id": run_id
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Missing run_id"
                        })
                
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}"
                    })
        
        except WebSocketDisconnect:
            await manager.disconnect(websocket)

    # Run management endpoints

    @app.post("/api/runs", response_model=Run)
    async def create_run(request: CreateRunRequest) -> Run:
        """Create a new training run."""
        run = Run(
            id=f"run_{uuid.uuid4().hex[:12]}",
            name=request.name,
            status=RunStatus.RUNNING,
            config=request.config,
            start_time=datetime.utcnow(),
        )

        await app.state.db.create_run(run)
        logger.info("run_created", run_id=run.id, name=run.name)
        return run

    @app.get("/api/runs", response_model=List[Run])
    async def list_runs() -> List[Run]:
        """List all training runs."""
        runs = await app.state.db.list_runs()
        return runs

    @app.get("/api/runs/{run_id}", response_model=Run)
    async def get_run(run_id: str) -> Run:
        """Get a specific training run."""
        run = await app.state.db.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    @app.put("/api/runs/{run_id}")
    async def update_run(run_id: str, request: UpdateRunRequest) -> dict:
        """Update a training run."""
        run = await app.state.db.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        old_status = run.status

        if request.status is not None:
            await app.state.db.update_run_status(run_id, request.status)
            logger.info("run_status_updated", run_id=run_id, status=request.status)
            
            # Broadcast status change
            await app.state.ws_manager.broadcast_to_run(run_id, {
                "type": "status_change",
                "run_id": run_id,
                "old_status": old_status.value,
                "new_status": request.status.value
            })

        if request.end_time is not None:
            await app.state.db.update_run(run_id, end_time=request.end_time)

        return {"message": "Run updated"}

    @app.delete("/api/runs/{run_id}")
    async def delete_run(run_id: str) -> dict:
        """Delete a training run."""
        run = await app.state.db.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        await app.state.db.delete_run(run_id)
        logger.info("run_deleted", run_id=run_id)
        return {"message": "Run deleted"}

    # Metrics endpoints

    @app.post("/api/runs/{run_id}/metrics")
    async def log_metrics(run_id: str, request: LogMetricsRequest) -> dict:
        """Log metrics for a run."""
        run = await app.state.db.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        metric = MetricPoint(
            run_id=run_id, step=request.step, metrics=request.metrics
        )
        await app.state.db.log_metrics(metric)

        logger.debug(
            "metrics_logged", run_id=run_id, step=request.step, metrics=request.metrics
        )
        
        # Broadcast metric update
        await app.state.ws_manager.broadcast_to_run(run_id, {
            "type": "metric_update",
            "run_id": run_id,
            "step": request.step,
            "metrics": request.metrics
        })
        
        return {"message": "Metrics logged"}

    @app.get("/api/runs/{run_id}/metrics", response_model=List[MetricPoint])
    async def get_metrics(run_id: str, limit: Optional[int] = None) -> List[MetricPoint]:
        """Get metrics for a run."""
        metrics = await app.state.db.get_metrics(run_id, limit=limit)
        return metrics

    @app.get("/api/runs/{run_id}/metrics/latest", response_model=MetricPoint)
    async def get_latest_metrics(run_id: str) -> MetricPoint:
        """Get the latest metrics for a run."""
        metric = await app.state.db.get_latest_metrics(run_id)
        if metric is None:
            raise HTTPException(status_code=404, detail="No metrics found")
        return metric

    # Checkpoint endpoints

    @app.post("/api/runs/{run_id}/checkpoints", response_model=Checkpoint)
    async def create_checkpoint(
        run_id: str, request: CreateCheckpointRequest
    ) -> Checkpoint:
        """Register a new checkpoint."""
        run = await app.state.db.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        checkpoint = Checkpoint(
            id=f"ckpt_{uuid.uuid4().hex[:12]}",
            run_id=run_id,
            filepath=request.filepath,
            metrics=request.metrics,
            metadata=request.metadata,
        )

        await app.state.db.create_checkpoint(checkpoint)
        logger.info(
            "checkpoint_created", checkpoint_id=checkpoint.id, filepath=request.filepath
        )
        
        # Broadcast checkpoint saved
        await app.state.ws_manager.broadcast_to_run(run_id, {
            "type": "checkpoint_saved",
            "run_id": run_id,
            "checkpoint_id": checkpoint.id,
            "metrics": request.metrics
        })
        
        return checkpoint

    @app.get("/api/runs/{run_id}/checkpoints", response_model=List[Checkpoint])
    async def list_checkpoints(run_id: str) -> List[Checkpoint]:
        """List all checkpoints for a run."""
        checkpoints = await app.state.db.list_checkpoints(run_id)
        return checkpoints

    @app.get("/api/checkpoints/{checkpoint_id}", response_model=Checkpoint)
    async def get_checkpoint(checkpoint_id: str) -> Checkpoint:
        """Get a specific checkpoint."""
        checkpoint = await app.state.db.get_checkpoint(checkpoint_id)
        if checkpoint is None:
            raise HTTPException(status_code=404, detail="Checkpoint not found")
        return checkpoint

    @app.delete("/api/checkpoints/{checkpoint_id}")
    async def delete_checkpoint(checkpoint_id: str) -> dict:
        """Delete a checkpoint."""
        checkpoint = await app.state.db.get_checkpoint(checkpoint_id)
        if checkpoint is None:
            raise HTTPException(status_code=404, detail="Checkpoint not found")

        await app.state.db.delete_checkpoint(checkpoint_id)
        logger.info("checkpoint_deleted", checkpoint_id=checkpoint_id)
        return {"message": "Checkpoint deleted"}

    @app.post("/api/checkpoints/{checkpoint_id}/mark_best")
    async def mark_checkpoint_as_best(checkpoint_id: str) -> dict:
        """Mark a checkpoint as the best for its run."""
        checkpoint = await app.state.db.get_checkpoint(checkpoint_id)
        if checkpoint is None:
            raise HTTPException(status_code=404, detail="Checkpoint not found")

        await app.state.db.mark_checkpoint_as_best(checkpoint_id, checkpoint.run_id)
        logger.info("checkpoint_marked_best", checkpoint_id=checkpoint_id)
        return {"message": "Checkpoint marked as best"}

    # System metrics endpoints

    @app.get("/api/system/status", response_model=SystemMetrics)
    async def get_system_status() -> SystemMetrics:
        """Get current system resource usage."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        # Try to get GPU metrics (will be None if not available)
        gpu_memory_used = None
        gpu_memory_total = None
        gpu_utilization = None

        metrics = SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            gpu_utilization=gpu_utilization,
        )

        # Log system metrics
        await app.state.db.log_system_metrics(metrics)

        return metrics

    @app.get("/api/system/history", response_model=List[SystemMetrics])
    async def get_system_history(limit: int = 100) -> List[SystemMetrics]:
        """Get historical system metrics."""
        history = await app.state.db.get_system_metrics_history(limit=limit)
        return history

    return app

