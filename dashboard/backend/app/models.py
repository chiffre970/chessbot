"""Data models for the training dashboard."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    """Training run status."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class Run(BaseModel):
    """Training run model."""

    id: str
    name: str
    status: RunStatus
    config: Optional[Dict[str, Any]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MetricPoint(BaseModel):
    """Single metric measurement."""

    run_id: str
    step: int
    metrics: Dict[str, float]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Checkpoint(BaseModel):
    """Model checkpoint metadata."""

    id: str
    run_id: str
    filepath: str
    metrics: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None
    is_best: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SystemMetrics(BaseModel):
    """System resource metrics."""

    cpu_percent: float
    memory_percent: float
    gpu_memory_used: Optional[int] = None
    gpu_memory_total: Optional[int] = None
    gpu_utilization: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# API Request/Response models


class CreateRunRequest(BaseModel):
    """Request to create a new training run."""

    name: str
    config: Optional[Dict[str, Any]] = None


class UpdateRunRequest(BaseModel):
    """Request to update a run."""

    status: Optional[RunStatus] = None
    end_time: Optional[datetime] = None


class LogMetricsRequest(BaseModel):
    """Request to log metrics."""

    step: int
    metrics: Dict[str, float]


class CreateCheckpointRequest(BaseModel):
    """Request to register a checkpoint."""

    filepath: str
    metrics: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


# WebSocket message models


class WSMessage(BaseModel):
    """Base WebSocket message."""

    type: str


class WSSubscribeMessage(WSMessage):
    """Subscribe to run updates."""

    type: str = "subscribe"
    run_id: str


class WSUnsubscribeMessage(WSMessage):
    """Unsubscribe from run updates."""

    type: str = "unsubscribe"
    run_id: str


class WSMetricUpdateMessage(WSMessage):
    """Metric update notification."""

    type: str = "metric_update"
    run_id: str
    step: int
    metrics: Dict[str, float]


class WSStatusChangeMessage(WSMessage):
    """Status change notification."""

    type: str = "status_change"
    run_id: str
    old_status: RunStatus
    new_status: RunStatus


class WSCheckpointSavedMessage(WSMessage):
    """Checkpoint saved notification."""

    type: str = "checkpoint_saved"
    run_id: str
    checkpoint_id: str
    metrics: Optional[Dict[str, float]] = None


