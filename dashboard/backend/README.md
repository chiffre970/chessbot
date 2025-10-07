# Training Dashboard - Backend

A FastAPI-based backend for real-time ML training monitoring with WebSocket support.

## Features

- ✅ **REST API** for runs, metrics, and checkpoints management
- ✅ **WebSocket** for real-time updates
- ✅ **SQLite** database with WAL mode for concurrency
- ✅ **Python SDK** (TrainingMonitor) for easy integration
- ✅ **System metrics** monitoring (CPU, memory, GPU)
- ✅ **Crash recovery** and offline mode support
- ✅ **91% test coverage** with TDD approach

## Quick Start

### Installation

```bash
cd dashboard/backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### Running the Server

```bash
python -m app.main
```

Server will start on `http://localhost:8000`.

API docs available at: `http://localhost:8000/docs`

### Running Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=app --cov-report=html
```

## Using the TrainingMonitor SDK

### Basic Usage

```python
from app.client import TrainingMonitor

# Initialize monitor
monitor = TrainingMonitor(
    run_name="my_experiment",
    config={"lr": 0.001, "batch_size": 32},
    dashboard_url="http://localhost:8000"
)

# Log metrics
for step in range(1000):
    loss = train_step()
    monitor.log_metrics({"loss": loss}, step=step)

# Save checkpoint
monitor.save_checkpoint(
    filepath="./checkpoints/model_1000.pt",
    metrics={"loss": 0.42}
)

# Update status
monitor.set_status("completed")
```

### Context Manager

```python
from app.client import TrainingMonitor

with TrainingMonitor(run_name="experiment") as monitor:
    for step in range(1000):
        loss = train_step()
        monitor.log_metrics({"loss": loss}, step=step)
    # Automatically marks as completed
```

### Offline Mode

The client automatically switches to offline mode if the dashboard is unavailable:

```python
monitor = TrainingMonitor(run_name="experiment")
# If dashboard is down, training continues without errors
monitor.log_metrics({"loss": 0.5}, step=1)
```

## API Endpoints

### Health

- `GET /health` - Health check
- `GET /api/version` - API version

### Runs

- `GET /api/runs` - List all runs
- `GET /api/runs/{run_id}` - Get run details
- `POST /api/runs` - Create new run
- `PUT /api/runs/{run_id}` - Update run
- `DELETE /api/runs/{run_id}` - Delete run

### Metrics

- `GET /api/runs/{run_id}/metrics` - Get metrics
- `GET /api/runs/{run_id}/metrics/latest` - Get latest metrics
- `POST /api/runs/{run_id}/metrics` - Log metrics

### Checkpoints

- `GET /api/runs/{run_id}/checkpoints` - List checkpoints
- `GET /api/checkpoints/{checkpoint_id}` - Get checkpoint
- `POST /api/runs/{run_id}/checkpoints` - Register checkpoint
- `DELETE /api/checkpoints/{checkpoint_id}` - Delete checkpoint
- `POST /api/checkpoints/{checkpoint_id}/mark_best` - Mark as best

### System

- `GET /api/system/status` - Current system metrics
- `GET /api/system/history` - Historical system metrics

### WebSocket

- `WS /ws` - WebSocket endpoint for real-time updates

WebSocket message types:
- `subscribe` - Subscribe to run updates
- `unsubscribe` - Unsubscribe from run updates
- `ping` - Heartbeat
- Server sends: `metric_update`, `status_change`, `checkpoint_saved`

## Project Structure

```
dashboard/backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # Entry point
│   ├── api.py               # FastAPI application
│   ├── database.py          # Database operations
│   ├── models.py            # Pydantic models
│   ├── websocket.py         # WebSocket manager
│   └── client.py            # TrainingMonitor SDK
├── tests/
│   ├── test_api.py          # API tests
│   ├── test_database.py     # Database tests
│   ├── test_websocket.py    # WebSocket tests
│   └── test_client.py       # Client SDK tests
├── requirements.txt
├── setup.py
├── pyproject.toml           # Config (pytest, black, ruff, mypy)
└── README.md                # This file
```

## Development

### Code Quality

Format code:
```bash
black app tests
```

Lint code:
```bash
ruff check app tests
```

Type check:
```bash
mypy app
```

### Running with Auto-Reload

```bash
python -m app.main --reload
```

### Database

Database is stored in `data/dashboard.db` by default.

To use a custom path:
```bash
python -m app.main --db-path /path/to/database.db
```

## Architecture

### Database Schema

- **runs**: Training run metadata
- **metrics**: Time-series metrics
- **checkpoints**: Model checkpoint registry
- **system_metrics**: System resource usage

See `app/database.py` for full schema.

### WebSocket Protocol

The WebSocket connection manager (`app/websocket.py`) handles:
- Connection management
- Run subscriptions
- Broadcasting updates to subscribers

Messages are broadcasted when:
- Metrics are logged
- Run status changes
- Checkpoints are saved

## Configuration

Environment variables:
- `DASHBOARD_HOST` - Host to bind to (default: `0.0.0.0`)
- `DASHBOARD_PORT` - Port to bind to (default: `8000`)
- `DASHBOARD_DB_PATH` - Database path (default: `data/dashboard.db`)

## Performance

- **SQLite** with WAL mode for concurrent reads/writes
- **Async** operations throughout
- **Connection pooling** for database
- **Metric batching** in SDK (configurable)
- **Background threads** for metric flushing

Target: 1000 metrics/second (well beyond single training run needs)

## Testing

Total: **56 tests** with **91% coverage**

- Database operations: 18 tests
- API endpoints: 22 tests  
- WebSocket: 9 tests
- Client SDK: 7 tests

Run specific test file:
```bash
pytest tests/test_api.py -v
```

## License

MIT


