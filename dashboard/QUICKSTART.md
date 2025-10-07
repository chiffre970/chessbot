# Training Dashboard - Quick Start Guide

## 🚀 Current Status

✅ **Backend is RUNNING** on `http://localhost:8000`

- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- Database: `dashboard/backend/data/dashboard.db`

## 📊 Test the Dashboard

### Run the Demo Training Script

Open a **new terminal** and run:

```bash
cd /Users/rmh/Code/chessbot
python3 examples/demo_training.py
```

This will:
- Create a training run called "Demo Training Run"
- Simulate 100 training steps with realistic metrics
- Log metrics in real-time to the dashboard
- Save checkpoints every 25 steps
- Show progress in the terminal

### Monitor with API

While the demo is running, open another terminal to see real-time updates:

```bash
# List all runs
curl -s http://localhost:8000/api/runs | python3 -m json.tool

# Get metrics for a run (replace RUN_ID)
curl -s "http://localhost:8000/api/runs/RUN_ID/metrics?limit=10" | python3 -m json.tool

# Get latest metrics
curl -s http://localhost:8000/api/runs/RUN_ID/metrics/latest | python3 -m json.tool

# List checkpoints
curl -s http://localhost:8000/api/runs/RUN_ID/checkpoints | python3 -m json.tool
```

### Test WebSocket (Real-time Updates)

```bash
# Install websocat if needed: brew install websocat

# Connect to WebSocket and subscribe to a run
websocat ws://localhost:8000/ws

# Then send (replace RUN_ID):
{"type": "subscribe", "run_id": "RUN_ID"}

# You'll receive real-time updates like:
# {"type": "metric_update", "run_id": "...", "step": 10, "metrics": {...}}
```

## 🛠️ Available Endpoints

### Runs
- `GET /api/runs` - List all runs
- `POST /api/runs` - Create new run
- `GET /api/runs/{run_id}` - Get run details
- `PUT /api/runs/{run_id}` - Update run status
- `DELETE /api/runs/{run_id}` - Delete run

### Metrics
- `POST /api/runs/{run_id}/metrics` - Log metrics
- `GET /api/runs/{run_id}/metrics` - Get all metrics
- `GET /api/runs/{run_id}/metrics/latest` - Get latest metrics

### Checkpoints
- `POST /api/runs/{run_id}/checkpoints` - Register checkpoint
- `GET /api/runs/{run_id}/checkpoints` - List checkpoints
- `GET /api/checkpoints/{checkpoint_id}` - Get checkpoint details

### System
- `GET /api/system/status` - Current CPU/memory/GPU usage
- `GET /api/system/history` - Historical system metrics

### WebSocket
- `WS /ws` - Real-time updates
  - Send: `{"type": "subscribe", "run_id": "..."}`
  - Receive: `metric_update`, `status_change`, `checkpoint_saved`

## 📝 Use in Your Own Training Script

```python
from app.client import TrainingMonitor

# Option 1: Manual control
monitor = TrainingMonitor(
    run_name="my_experiment",
    config={"lr": 0.001, "batch_size": 32}
)

for step in range(1000):
    loss = train_step()
    monitor.log_metrics({"loss": loss}, step=step)

monitor.set_status("completed")

# Option 2: Context manager (auto-completes)
with TrainingMonitor(run_name="my_experiment") as monitor:
    for step in range(1000):
        loss = train_step()
        monitor.log_metrics({"loss": loss}, step=step)
```

## 🔧 Server Management

### Stop the Server
```bash
# Find the process
ps aux | grep "app.main"

# Kill it
kill <PID>
```

### Restart the Server
```bash
cd dashboard/backend
source venv/bin/activate
python -m app.main
```

### View Server Logs
Server logs are printed to stdout. To save them:
```bash
python -m app.main > server.log 2>&1 &
```

## 📊 Database

The SQLite database is located at:
```
dashboard/backend/data/dashboard.db
```

You can inspect it with:
```bash
sqlite3 dashboard/backend/data/dashboard.db

# Inside sqlite3:
.tables
.schema runs
SELECT * FROM runs;
```

## ⚡ Performance

- **Metrics ingestion**: Target 1000/second (well above needs)
- **WebSocket latency**: <1 second
- **Database**: WAL mode for concurrent access
- **Thread-safe**: Client SDK uses locks

## 🧪 Testing

```bash
cd dashboard/backend
source venv/bin/activate

# Run all tests
pytest

# Run specific test file
pytest tests/test_api.py -v

# With coverage
pytest --cov=app --cov-report=html
```

**Current Status: 56 tests passing, 91% coverage** ✅

## 🎯 Next Steps

1. ✅ Backend is complete and tested
2. ⏳ Frontend (React + Vite with Bloomberg Terminal styling)
3. ⏳ E2E integration tests

## 📚 Documentation

- Backend README: `dashboard/backend/README.md`
- API Docs (interactive): http://localhost:8000/docs
- Main Plan: `PLAN.md`
- Dashboard Plan: `DASHBOARD_PLAN.md`

---

**Server is running!** Try the demo script now! 🚀


