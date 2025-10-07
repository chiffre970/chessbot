# ChessBot Training Platform - Project Plan

## Overview
A two-part project: (1) A Bloomberg Terminal-style ML training dashboard with real-time monitoring, checkpoint management, and experiment tracking, and (2) An AlphaZero-style chess engine to test the platform.

**Primary Goal**: Build a signal-dense, military/high-finance aesthetic training dashboard  
**Secondary Goal**: Implement and train an AlphaZero chess bot as proof-of-concept  
**Future Consideration**: Potentially spin out the training dashboard as standalone software

---

## Tech Stack Summary

### Training Dashboard
- **Backend**: FastAPI + WebSockets + SQLite
- **Frontend**: React (Vite) + Plotly.js
- **Style**: Bloomberg Terminal aesthetic (black/amber/green/red)

### Chess Bot
- **Framework**: PyTorch (Metal GPU on M4)
- **Architecture**: ResNet-based policy-value network
- **Algorithm**: MCTS + Self-Play (AlphaZero approach)

### Target Hardware
- M4 MacBook Pro 14" (24GB RAM)
- Single machine, no distributed training
- Metal GPU acceleration via PyTorch MPS backend

---

## Project Structure

```
chessbot/
├── dashboard/                  # Training dashboard (potentially spinout)
│   ├── backend/               # FastAPI server
│   └── frontend/              # React + Vite
├── chessbot/                  # AlphaZero chess implementation
│   ├── engine/               # Chess logic wrapper
│   ├── model/                # Neural network architecture
│   ├── mcts/                 # Monte Carlo Tree Search
│   ├── selfplay/             # Self-play game generation
│   ├── training/             # Training loop
│   └── evaluation/           # Testing & playing interface
├── data/                      # Training data & storage
│   ├── runs/                 # Training run metadata
│   ├── checkpoints/          # Model checkpoints
│   ├── games/                # PGN files from self-play
│   └── experiments/          # Experiment configs
├── PLAN.md                    # This file (high-level overview)
├── DASHBOARD_PLAN.md          # Detailed dashboard implementation
├── CHESS_PLAN.md              # Detailed chess bot implementation
├── README.md
└── requirements.txt
```

---

## Core Components

### 1. Training Dashboard
See [DASHBOARD_PLAN.md](./DASHBOARD_PLAN.md) for detailed implementation.

**Key Features**:
- Real-time monitoring with Bloomberg Terminal aesthetic
- Run management (registry, history, status tracking)
- Checkpoint management (browser, sorting, validation)
- SQLite storage with ACID compliance
- WebSocket-driven updates (< 1 second latency)
- Crash recovery and error handling

**Deliverable**: Generic ML training dashboard that works with any training script

### 2. AlphaZero Chess Bot
See [CHESS_PLAN.md](./CHESS_PLAN.md) for detailed implementation.

**Key Components**:
- ResNet-based policy-value network (5-10M parameters)
- Monte Carlo Tree Search (400-1200 simulations/move)
- Self-play game generation with replay buffer
- Training loop with evaluation every 5-10 iterations
- Dashboard integration for real-time monitoring

**Deliverable**: Chess bot that plays reasonable chess (>800 ELO target)

---

## Development Phases

### Phase 0: Benchmarking & Baseline (2-3 days)
**Goal**: Understand M4 hardware limits

- Hardware profiling (Metal backend, memory bandwidth)
- ResNet sizing benchmarks (5M, 10M, 20M params)
- MCTS performance baseline (simulations/second)
- Self-play throughput estimates
- Testing infrastructure setup (pytest)

**Deliverable**: Performance report + testing foundation

### Phase 1: Dashboard Foundation (1-2 weeks)
**Goal**: Basic monitoring infrastructure

- Backend: FastAPI + WebSockets + SQLite
- Frontend: React + Bloomberg theming
- SDK: Python client library (TrainingMonitor)
- Test with dummy data

**Deliverable**: Working dashboard with test suite

### Phase 2: Dashboard Features (2-3 weeks)
**Goal**: Production-ready dashboard

- Enhanced visualization (Plotly.js)
- Run management (registry, history)
- Checkpoint management (browser, validation)
- Error handling (crash recovery, reconnection)
- UI polish + E2E tests

**Deliverable**: Production-ready dashboard

### Phase 3: Chess Bot - Core Components (1-2 weeks)
**Goal**: Foundational pieces

- Chess engine wrapper (board encoding)
- Neural network (ResNet architecture)
- MCTS implementation
- All with test coverage

**Deliverable**: Network + MCTS working independently

### Phase 4: Chess Bot - Training Pipeline (2-3 weeks)
**Goal**: End-to-end training

- Self-play (game generation + replay buffer)
- Training loop (loss, optimizer, scheduler)
- Evaluation system (model vs model)
- Dashboard integration
- Initial test run (5 iterations)

**Deliverable**: Complete training pipeline

### Phase 5: Chess Bot - Full Training (2-6+ weeks)
**Goal**: Train competent bot

- Hyperparameter tuning
- Long training run (50-100 iterations)
- Evaluation tools (CLI interface, visualization)
- Optimization (if needed)

**Deliverable**: Trained bot (>800 ELO)

### Phase 6: Polish & Documentation (1 week)
**Goal**: Production-ready

- Code cleanup (refactoring, type hints, docstrings)
- Documentation (README, API docs, guides)
- Final testing pass (>80% coverage)
- Spinout preparation

**Deliverable**: Polished, documented system

---

## Timeline Estimate
- **Phase 0**: 2-3 days
- **Phase 1**: 1-2 weeks  
- **Phase 2**: 2-3 weeks
- **Phase 3**: 1-2 weeks
- **Phase 4**: 2-3 weeks
- **Phase 5**: 2-6+ weeks (mostly training time)
- **Phase 6**: 1 week

**Total**: 10-18 weeks for complete system

**Notes**:
- Dashboard and chess bot can be parallelized after Phase 0
- Timeline assumes single developer, full-time work
- Expect 20-30% buffer for debugging

---

## Success Metrics

### Dashboard
- [ ] Monitor single training run with full visibility
- [ ] Real-time updates (< 1 second latency)
- [ ] Crash recovery works
- [ ] Bloomberg aesthetic achieved
- [ ] Generic (works with non-chess ML projects)

### Chess Bot
- [ ] Network trains on M4 Metal
- [ ] Self-play generates 100% valid games
- [ ] Bot improves over iterations (ELO increases)
- [ ] Plays reasonable chess (>800 ELO)

### Integration
- [ ] Metrics flow from training → dashboard
- [ ] Checkpoint management works end-to-end
- [ ] System handles M4 resource constraints

---

## Technical Constraints

### M4 MacBook (24GB RAM)
- **Available for training**: ~8-10GB (after OS, browser, dashboard)
- **Compute**: ~1/5 to 1/10 speed of A100
- **Storage**: ~3-6GB for full training run

### Key Trade-offs
- Small network (5-10M params vs AlphaZero's 100M+)
- Fewer MCTS sims (400-1200 vs 800-1600)
- Fewer iterations (50-100 vs 700k)
- Result: Weaker bot, but validates approach

---

## Dependencies

See detailed plans for full dependency lists.

**Core**:
- Dashboard: FastAPI, React, Plotly.js, SQLite
- Chess: PyTorch (Metal), python-chess, numpy
- Testing: pytest, playwright
- Dev: black, ruff, mypy

---

## Future Enhancements (Post-MVP)

### Dashboard
- Multiple concurrent runs
- Experiment comparison
- Alerts/notifications
- Distributed training support
- Authentication

### Chess Bot
- Transformer architecture experimentation
- Larger network (cloud GPUs)
- UCI protocol (play on Lichess)
- Web-based play interface

---

## Design Principles

- **KISS**: Keep it simple, especially for MVP
- **Modularity**: Dashboard has zero chess-specific logic
- **Instrumentation**: Over-log rather than under-log
- **Testing**: Test as you build, not at the end
- **Recovery**: Always recoverable from crash

---

## Related Documents

- **[DASHBOARD_PLAN.md](./DASHBOARD_PLAN.md)**: Detailed dashboard architecture, API design, UI/UX specs
- **[CHESS_PLAN.md](./CHESS_PLAN.md)**: Detailed chess bot architecture, neural network, MCTS, training pipeline
- **[README.md](./README.md)**: Setup instructions and quickstart guide

---

*This plan will evolve as we build. Expect adjustments based on what we learn during implementation.*
