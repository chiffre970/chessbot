# Training Dashboard - Implementation Plan

*Detailed implementation specifications for the ML training dashboard component.*

---

## Overview

A Bloomberg Terminal-style training dashboard with real-time monitoring, checkpoint management, and experiment tracking. Designed to be generic and potentially spin out as standalone software.

**Key Requirement**: Zero chess-specific logic. Must work with any ML training script.

---

## Architecture

### System Components

```
┌─────────────────┐         WebSocket          ┌─────────────────┐
│  Training       │◄──────────────────────────►│   Dashboard     │
│  Script         │         (metrics)           │   Backend       │
│  (any ML)       │                             │   (FastAPI)     │
└─────────────────┘                             └────────┬────────┘
        │                                                │
        │ TrainingMonitor SDK                            │
        │                                                │
        └────────────────────────────────────────────────┘
                                                         │
                                                         │ SQLite
                                                         ▼
                                              ┌──────────────────┐
                                              │   Database       │
                                              │   - runs         │
                                              │   - metrics      │
                                              │   - checkpoints  │
                                              └──────────────────┘
                                                         │
                                                         │ REST + WS
                                                         ▼
                                              ┌──────────────────┐
                                              │   Dashboard      │
                                              │   Frontend       │
                                              │   (React)        │
                                              └──────────────────┘
```

---

## Backend (FastAPI)

### Database Schema (SQLite)

```sql
-- Runs table
CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL,  -- running, completed, failed, interrupted
    config TEXT,           -- JSON config
    start_time DATETIME,
    end_time DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Metrics table
CREATE TABLE metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    step INTEGER,
    metrics TEXT NOT NULL,  -- JSON blob
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

-- Checkpoints table
CREATE TABLE checkpoints (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    filepath TEXT NOT NULL,
    metrics TEXT,           -- JSON blob
    metadata TEXT,          -- JSON blob (optimizer state, etc.)
    is_best BOOLEAN DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

-- System metrics table
CREATE TABLE system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    cpu_percent REAL,
    memory_percent REAL,
    gpu_memory_used INTEGER,
    gpu_memory_total INTEGER,
    gpu_utilization REAL
);
```

### REST API Endpoints

```python
# Run management
GET     /api/runs                           # List all runs
GET     /api/runs/{run_id}                  # Get run details
POST    /api/runs                           # Create new run
PUT     /api/runs/{run_id}                  # Update run
DELETE  /api/runs/{run_id}                  # Delete run

# Metrics
GET     /api/runs/{run_id}/metrics          # Get metrics history
GET     /api/runs/{run_id}/metrics/latest   # Get latest metrics
POST    /api/runs/{run_id}/metrics          # Add metrics (batch)

# Checkpoints
GET     /api/runs/{run_id}/checkpoints      # List checkpoints
GET     /api/checkpoints/{checkpoint_id}    # Get checkpoint details
POST    /api/runs/{run_id}/checkpoints      # Register checkpoint
DELETE  /api/checkpoints/{checkpoint_id}    # Delete checkpoint
POST    /api/checkpoints/{checkpoint_id}/load  # Load checkpoint

# System
GET     /api/system/status                  # Current system resources
GET     /api/system/history                 # Historical system metrics

# Health
GET     /health                             # Health check
GET     /api/version                        # API version
```

### WebSocket Protocol

```javascript
// Client → Server messages
{
  "type": "subscribe",
  "run_id": "run_123"
}

{
  "type": "unsubscribe",
  "run_id": "run_123"
}

{
  "type": "ping"  // Heartbeat
}

// Server → Client messages
{
  "type": "metric_update",
  "run_id": "run_123",
  "step": 1000,
  "metrics": {
    "loss": 0.42,
    "learning_rate": 0.001
  }
}

{
  "type": "status_change",
  "run_id": "run_123",
  "old_status": "running",
  "new_status": "completed"
}

{
  "type": "checkpoint_saved",
  "run_id": "run_123",
  "checkpoint_id": "ckpt_456",
  "metrics": {...}
}

{
  "type": "error",
  "message": "Error description"
}

{
  "type": "pong"  // Heartbeat response
}
```

### Python SDK (TrainingMonitor)

```python
from dashboard.client import TrainingMonitor

# Initialize
monitor = TrainingMonitor(
    run_name="my_experiment",
    config={"lr": 0.001, "batch_size": 128},
    dashboard_url="ws://localhost:8000"
)

# Log metrics
monitor.log_metrics({
    "loss": 0.42,
    "accuracy": 0.85
}, step=1000)

# Log system metrics (automatic background thread)
# CPU, memory, GPU usage collected automatically

# Save checkpoint
monitor.save_checkpoint(
    filepath="./checkpoints/model_1000.pt",
    metrics={"loss": 0.42},
    metadata={"optimizer_state": "..."}
)

# Mark checkpoint as best
monitor.mark_best_checkpoint(checkpoint_id="ckpt_123")

# Update status
monitor.set_status("completed")  # or "failed", "interrupted"

# Context manager support
with TrainingMonitor(run_name="experiment") as monitor:
    for step in range(1000):
        # training code
        monitor.log_metrics({"loss": loss}, step=step)
    # automatically marks as completed
```

### Error Handling & Recovery

#### Crash Recovery
```python
# Training script checks for existing run
monitor = TrainingMonitor.resume_or_create(
    run_name="my_experiment",
    checkpoint_dir="./checkpoints"
)

if monitor.is_resumed:
    last_checkpoint = monitor.get_last_checkpoint()
    model.load_state_dict(last_checkpoint['model'])
    optimizer.load_state_dict(last_checkpoint['optimizer'])
    start_step = last_checkpoint['step']
else:
    start_step = 0
```

#### WebSocket Reconnection
- Automatic exponential backoff (1s, 2s, 4s, 8s, max 30s)
- Buffer metrics during disconnection (max 1000 entries)
- Flush buffer on reconnect
- Training continues even if dashboard unreachable

#### Checkpoint Validation
```python
def validate_checkpoint(filepath: str, expected_metadata: dict) -> bool:
    """
    Verify checkpoint integrity before marking as valid.
    - File exists and is readable
    - Can be loaded by torch.load()
    - Metadata matches expected structure
    """
    try:
        checkpoint = torch.load(filepath, map_location='cpu')
        assert 'model' in checkpoint
        assert 'step' in checkpoint
        return True
    except Exception as e:
        log_error(f"Checkpoint validation failed: {e}")
        return False
```

---

## Frontend (React + Vite)

### Component Structure

```
src/
├── components/
│   ├── layout/
│   │   ├── Dashboard.tsx       # Main layout grid
│   │   ├── Panel.tsx           # Reusable panel container
│   │   └── Header.tsx          # Top bar
│   ├── runs/
│   │   ├── RunList.tsx         # Active runs list
│   │   ├── RunDetails.tsx      # Run configuration
│   │   └── RunStatus.tsx       # Status badge
│   ├── metrics/
│   │   ├── MetricsChart.tsx    # Plotly chart wrapper
│   │   ├── MetricsTable.tsx    # Latest metrics table
│   │   └── MetricsHistory.tsx  # Historical view
│   ├── checkpoints/
│   │   ├── CheckpointList.tsx  # Sortable table
│   │   ├── CheckpointCard.tsx  # Individual checkpoint
│   │   └── CheckpointActions.tsx  # Load/delete buttons
│   ├── system/
│   │   ├── SystemStatus.tsx    # CPU/GPU/Memory
│   │   └── ResourceChart.tsx   # Resource history
│   └── events/
│       └── EventLog.tsx        # Real-time event stream
├── services/
│   ├── api.ts                  # REST API client (axios)
│   ├── websocket.ts            # WebSocket client
│   └── types.ts                # TypeScript types
├── hooks/
│   ├── useRuns.ts              # Runs data hook
│   ├── useMetrics.ts           # Metrics data hook
│   ├── useWebSocket.ts         # WebSocket connection hook
│   └── useSystemStatus.ts      # System metrics hook
├── styles/
│   ├── theme.ts                # Bloomberg theme colors
│   └── global.css              # Global styles
└── App.tsx                     # Root component
```

### UI/UX Design

#### Bloomberg Terminal Aesthetic

**Color Palette**:
```css
:root {
  /* Background */
  --bg-primary: #000000;
  --bg-secondary: #0a0a0a;
  --bg-panel: #1a1a1a;
  
  /* Text */
  --text-primary: #FFD700;    /* Amber */
  --text-secondary: #FFA500;  /* Orange */
  --text-muted: #808080;      /* Gray */
  
  /* Status colors */
  --status-good: #00FF00;     /* Green */
  --status-warning: #FFA500;  /* Orange */
  --status-error: #FF0000;    /* Red */
  --status-info: #00FFFF;     /* Cyan */
  
  /* Borders */
  --border-color: #333333;
  
  /* Fonts */
  --font-mono: 'JetBrains Mono', 'Fira Code', 'Monaco', monospace;
}
```

**Layout Grid** (3×3 panels):
```
┌──────────────┬──────────────┬──────────────┐
│  RUN LIST    │  SYSTEM      │  EVENT LOG   │
│              │  STATUS      │              │
├──────────────┼──────────────┼──────────────┤
│  LOSS PLOT   │  LR PLOT     │  RUN CONFIG  │
│              │              │              │
├──────────────┴──────────────┼──────────────┤
│  CHECKPOINTS                │  ACTIONS     │
│  (sortable table)           │              │
└─────────────────────────────┴──────────────┘
```

**Typography**:
- All text in monospace font
- Dense layout (minimal padding)
- High information density
- Clear visual hierarchy with color

**Interactive Elements**:
- Hover: Subtle highlight (#2a2a2a)
- Active: Brighter border (--status-info)
- Disabled: Muted text (--text-muted)
- Buttons: Outlined style, no fills

#### Key UI Components

**Run Status Badge**:
```tsx
<RunStatus status="running" />
// → [● RUNNING] in green
<RunStatus status="failed" />
// → [✗ FAILED] in red
<RunStatus status="completed" />
// → [✓ COMPLETED] in green
```

**Metrics Chart** (Plotly.js):
- Black background
- Amber/cyan/green lines
- Auto-scaling Y-axis
- Real-time updates (every 1s)
- Configurable time window (last 100, 1000, all)

**Checkpoint Table**:
```
ID          | STEP   | LOSS   | ACCURACY | TIMESTAMP      | ACTIONS
─────────────────────────────────────────────────────────────────────
ckpt_003 ★  | 15000  | 0.0234 | 0.9823   | 2025-10-07 14:32 | [LOAD]
ckpt_002    | 10000  | 0.0456 | 0.9654   | 2025-10-07 12:15 | [LOAD]
ckpt_001    | 5000   | 0.0789 | 0.9321   | 2025-10-07 10:05 | [LOAD]
```
- Sortable by any column
- ★ indicates best checkpoint
- Click row to view details

---

## Data Persistence

### Storage Strategy

**SQLite for metadata**:
- Runs, metrics, checkpoints metadata
- ACID compliant (no corruption)
- Single file database
- No server process needed

**Filesystem for checkpoints**:
```
data/
├── dashboard.db              # SQLite database
├── checkpoints/
│   ├── run_abc/
│   │   ├── checkpoint_001.pt
│   │   ├── checkpoint_002.pt
│   │   └── checkpoint_003.pt
│   └── run_xyz/
│       └── checkpoint_001.pt
└── logs/
    ├── run_abc.log
    └── run_xyz.log
```

### Metrics Storage Optimization

**Challenge**: Storing millions of metric points efficiently

**Solution**: Time-series aggregation
- Store all metrics for last 24 hours (high resolution)
- Aggregate older metrics to 1-minute buckets
- Archive metrics older than 30 days to separate table

```python
# Automatic aggregation (background task)
async def aggregate_metrics():
    """Run hourly to aggregate old metrics."""
    cutoff = datetime.now() - timedelta(hours=24)
    
    # Aggregate to 1-minute buckets
    conn.execute("""
        INSERT INTO metrics_aggregated
        SELECT 
            run_id,
            datetime(timestamp, 'start of minute') as bucket,
            AVG(metric_value) as avg_value,
            MIN(metric_value) as min_value,
            MAX(metric_value) as max_value
        FROM metrics
        WHERE timestamp < ?
        GROUP BY run_id, bucket
    """, (cutoff,))
    
    # Delete raw metrics
    conn.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff,))
```

---

## Testing Strategy

### Unit Tests (pytest)

**Backend**:
```python
# test_api.py
def test_create_run(client):
    response = client.post("/api/runs", json={
        "name": "test_run",
        "config": {"lr": 0.001}
    })
    assert response.status_code == 200
    assert "id" in response.json()

def test_log_metrics(client, run_id):
    response = client.post(f"/api/runs/{run_id}/metrics", json={
        "step": 100,
        "metrics": {"loss": 0.5}
    })
    assert response.status_code == 200

# test_websocket.py
async def test_websocket_subscribe(websocket):
    await websocket.send_json({"type": "subscribe", "run_id": "test"})
    response = await websocket.receive_json()
    assert response["type"] == "subscribed"
```

**SDK**:
```python
# test_monitor.py
def test_monitor_log_metrics(mock_server):
    monitor = TrainingMonitor("test", dashboard_url=mock_server.url)
    monitor.log_metrics({"loss": 0.5}, step=100)
    
    # Verify metric sent to server
    assert mock_server.received_metrics[0]["loss"] == 0.5
```

### Integration Tests

```python
# test_integration.py
async def test_end_to_end_flow(client, websocket):
    # Create run
    run = client.post("/api/runs", json={"name": "test"}).json()
    
    # Subscribe via websocket
    await websocket.send_json({"type": "subscribe", "run_id": run["id"]})
    
    # Log metrics via REST
    client.post(f"/api/runs/{run['id']}/metrics", json={
        "step": 1,
        "metrics": {"loss": 0.5}
    })
    
    # Receive via websocket
    msg = await websocket.receive_json()
    assert msg["type"] == "metric_update"
    assert msg["metrics"]["loss"] == 0.5
```

### E2E Tests (Playwright)

```typescript
// tests/e2e/dashboard.spec.ts
test('displays real-time metrics', async ({ page }) => {
  await page.goto('http://localhost:3000');
  
  // Start mock training
  const run = await startMockTraining();
  
  // Verify run appears in list
  await expect(page.locator(`[data-run-id="${run.id}"]`)).toBeVisible();
  
  // Verify metrics update
  await page.waitForSelector('.metrics-chart');
  const chartData = await page.locator('.metrics-chart').getAttribute('data-points');
  expect(JSON.parse(chartData).length).toBeGreaterThan(0);
});
```

---

## Deployment & Operations

### Development Setup

```bash
# Backend
cd dashboard/backend
python -m venv venv
source venv/bin/activate
pip install -e .
uvicorn app.main:app --reload --port 8000

# Frontend
cd dashboard/frontend
npm install
npm run dev  # Starts on http://localhost:3000
```

### Production Deployment

```bash
# Backend (systemd service)
[Unit]
Description=Training Dashboard Backend
After=network.target

[Service]
Type=simple
User=dashboard
WorkingDirectory=/opt/dashboard/backend
Environment="PATH=/opt/dashboard/backend/venv/bin"
ExecStart=/opt/dashboard/backend/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target

# Frontend (nginx)
server {
    listen 80;
    server_name dashboard.example.com;
    
    root /opt/dashboard/frontend/dist;
    index index.html;
    
    location / {
        try_files $uri $uri/ /index.html;
    }
    
    location /api {
        proxy_pass http://localhost:8000;
    }
    
    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Monitoring & Logging

**Structured Logging**:
```python
import structlog

logger = structlog.get_logger()

logger.info("metric_logged",
    run_id=run_id,
    step=step,
    metric_count=len(metrics)
)

logger.error("checkpoint_validation_failed",
    checkpoint_id=checkpoint_id,
    error=str(e)
)
```

**Log Rotation**:
```bash
# /etc/logrotate.d/dashboard
/var/log/dashboard/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 dashboard dashboard
    sharedscripts
    postrotate
        systemctl reload dashboard
    endscript
}
```

---

## Performance Considerations

### Metrics Ingestion
- **Target**: 1000 metrics/second (well beyond single training run needs)
- **Batching**: SDK batches metrics every 1 second
- **Async writes**: Non-blocking SQLite writes
- **Connection pooling**: Reuse DB connections

### WebSocket Scaling
- **Single run**: No scaling needed
- **Future (multiple runs)**: Redis pub/sub for message broadcasting
- **Heartbeat**: 30-second ping/pong to detect disconnections

### Database Optimization
```sql
-- Indexes for fast queries
CREATE INDEX idx_metrics_run_step ON metrics(run_id, step);
CREATE INDEX idx_checkpoints_run ON checkpoints(run_id, created_at DESC);
CREATE INDEX idx_metrics_timestamp ON metrics(timestamp);

-- Vacuum regularly (via cron)
PRAGMA auto_vacuum = INCREMENTAL;
```

---

## Open Questions

1. Should we support exporting metrics to CSV/TensorBoard format?
2. Panel layout customization for users?
3. Alert thresholds (e.g., notify if loss > X)?
4. Support for remote dashboard (train on server, view locally)?

---

## Future Enhancements

- Multiple concurrent runs support
- Experiment comparison (side-by-side plots)
- Custom dashboards (user-configurable panels)
- Alerts (Slack/email/desktop notifications)
- Distributed training support (multi-GPU/multi-node)
- PostgreSQL backend option (for larger scale)
- Authentication & multi-user support
- Mobile-responsive UI
- Plugin system for custom metrics/visualizations

