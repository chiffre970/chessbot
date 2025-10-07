# Chess Bot - Implementation Plan

*Detailed implementation specifications for the AlphaZero-style chess engine.*

---

## Overview

An AlphaZero-inspired chess bot trained entirely through self-play reinforcement learning. No human games, no opening books, no handcrafted evaluation features—purely learned from scratch.

**Goal**: Demonstrate the training dashboard with a real ML project while building a competent chess player (>800 ELO target).

---

## Architecture

### High-Level Training Loop

```
┌─────────────────────────────────────────────────────┐
│                  Training Iteration                  │
│                                                      │
│  1. Self-Play (400-1200 MCTS sims per move)        │
│     ├─ Generate 100-500 games                       │
│     ├─ Store positions, policies, outcomes          │
│     └─ Save to replay buffer (SQLite)               │
│                                                      │
│  2. Training (1000-5000 steps)                      │
│     ├─ Sample batches from replay buffer            │
│     ├─ Compute loss (policy + value + L2)           │
│     ├─ Backpropagate and update network             │
│     └─ Log metrics to dashboard                     │
│                                                      │
│  3. Evaluation (every 5-10 iterations)              │
│     ├─ New model vs old model (100 games)           │
│     ├─ Calculate ELO rating                         │
│     └─ Mark best checkpoint                         │
│                                                      │
│  4. Checkpoint                                       │
│     ├─ Save model, optimizer, RNG state             │
│     └─ Export sample games to PGN                   │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## Component 1: Chess Engine Wrapper

### Board Representation (Input to Neural Network)

**119-plane encoding** (AlphaZero style):

```python
# Piece planes (12 planes)
P1_PAWN, P1_KNIGHT, P1_BISHOP, P1_ROOK, P1_QUEEN, P1_KING    # 6 planes
P2_PAWN, P2_KNIGHT, P2_BISHOP, P2_ROOK, P2_QUEEN, P2_KING    # 6 planes

# Historical positions (112 planes = 8 timesteps × 14 planes)
# For each of the last 8 positions:
#   - 12 piece planes
#   - 2 repetition count planes (once, twice)

# Auxiliary planes (7 planes)
COLOR            # 1 if current player is white
TOTAL_MOVE_COUNT # Move number / 100 (normalized)
P1_CASTLING_K    # Can castle kingside
P1_CASTLING_Q    # Can castle queenside
P2_CASTLING_K    # Opponent can castle kingside
P2_CASTLING_Q    # Opponent can castle queenside
NO_PROGRESS      # Moves since last capture or pawn move / 100

# Total: 12 + 112 + 7 = 131 planes
# (will optimize to 119 by reducing history to 7 timesteps)
```

### Move Representation (Output from Neural Network)

**1968-move encoding** (from-to squares):

```python
# Each move encoded as: from_square (64) × to_square (64) = 4096
# But only ~1968 are legal in chess:
#   - Normal moves: from → to
#   - Promotions: from → to + piece type (N, B, R, Q)
#   - Castling: encoded as king moves
#   - En passant: encoded as pawn captures

def encode_move(move: chess.Move) -> int:
    """Convert chess.Move to index in [0, 1967]."""
    from_sq = move.from_square
    to_sq = move.to_square
    
    if move.promotion:
        # Knight=0, Bishop=1, Rook=2, Queen=3
        promo_offset = [chess.KNIGHT, chess.BISHOP, 
                        chess.ROOK, chess.QUEEN].index(move.promotion)
        return from_sq * 64 + to_sq + promo_offset * 4096
    else:
        return from_sq * 64 + to_sq

def decode_move(index: int, board: chess.Board) -> chess.Move:
    """Convert index back to chess.Move."""
    # ... inverse of encode_move
```

### Legal Move Masking

```python
def get_legal_move_mask(board: chess.Board) -> np.ndarray:
    """
    Return binary mask of shape (1968,) where 1 = legal, 0 = illegal.
    Used to mask network policy output before softmax.
    """
    mask = np.zeros(1968, dtype=np.float32)
    for move in board.legal_moves:
        mask[encode_move(move)] = 1.0
    return mask
```

### Implementation

```python
# chessbot/engine/board.py
class ChessBoardWrapper:
    """Wrapper around python-chess with AlphaZero encoding."""
    
    def __init__(self):
        self.board = chess.Board()
        self.history = []  # Last 8 positions
    
    def get_state(self) -> np.ndarray:
        """Return 8×8×119 board representation."""
        planes = []
        
        # Piece planes (12)
        for piece_type in chess.PIECE_TYPES:
            for color in [chess.WHITE, chess.BLACK]:
                plane = self._get_piece_plane(piece_type, color)
                planes.append(plane)
        
        # Historical planes (112)
        for hist_board in self.history[-8:]:
            planes.extend(self._get_historical_planes(hist_board))
        
        # Pad if not enough history
        while len(planes) < 12 + 112:
            planes.append(np.zeros((8, 8)))
        
        # Auxiliary planes (7)
        planes.extend(self._get_auxiliary_planes())
        
        return np.stack(planes, axis=0)  # Shape: (119, 8, 8)
    
    def get_legal_moves(self) -> List[chess.Move]:
        return list(self.board.legal_moves)
    
    def get_legal_move_mask(self) -> np.ndarray:
        mask = np.zeros(1968)
        for move in self.board.legal_moves:
            mask[encode_move(move)] = 1.0
        return mask
    
    def make_move(self, move: chess.Move):
        self.history.append(self.board.copy())
        self.board.push(move)
    
    def is_game_over(self) -> bool:
        return self.board.is_game_over()
    
    def get_result(self) -> float:
        """Return game outcome from current player's perspective."""
        if self.board.is_checkmate():
            return -1.0 if self.board.turn else 1.0
        else:
            return 0.0  # Draw
```

---

## Component 2: Neural Network

### ResNet Architecture

```python
# chessbot/model/network.py
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Residual block: Conv → BN → ReLU → Conv → BN → Add → ReLU"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class ChessNet(nn.Module):
    """
    AlphaZero-style ResNet for chess.
    
    Architecture:
      - Input: 8×8×119 board representation
      - Initial conv: 3×3, 256 filters
      - Residual tower: 10-20 blocks (based on Phase 0 benchmarks)
      - Policy head: 1968 move probabilities
      - Value head: scalar win probability
    """
    
    def __init__(self, num_blocks: int = 10, num_channels: int = 256):
        super().__init__()
        
        # Initial convolution
        self.conv_input = nn.Conv2d(119, num_channels, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # Residual tower
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 1968)
        
        # Value head
        self.value_conv = nn.Conv2d(num_channels, 32, 1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        """
        Args:
            x: (batch, 119, 8, 8) board representation
        
        Returns:
            policy: (batch, 1968) move probabilities (pre-softmax)
            value: (batch, 1) win probability (tanh output)
        """
        # Initial conv
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual tower
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy = self.policy_fc(p)  # (batch, 1968)
        
        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))  # (batch, 1)
        
        return policy, value
    
    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predict policy and value for a single state.
        
        Args:
            state: (119, 8, 8) board representation
        
        Returns:
            policy: (1968,) move probabilities
            value: scalar win probability
        """
        self.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # (1, 119, 8, 8)
            policy_logits, value = self.forward(state_tensor)
            policy = F.softmax(policy_logits, dim=1)[0].cpu().numpy()
            value = value.item()
        return policy, value

# Model size estimation
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Example: 10 blocks, 256 channels
model = ChessNet(num_blocks=10, num_channels=256)
print(f"Parameters: {count_parameters(model):,}")  # ~8-10M
```

### Metal (MPS) Backend Support

```python
# chessbot/model/device.py
def get_device():
    """
    Get best available device.
    Priority: Metal (MPS) > CUDA > CPU
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Metal (MPS) backend")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA backend")
    else:
        device = torch.device("cpu")
        print("Using CPU backend")
    
    return device

# Handle MPS fallback
def safe_to_device(tensor, device):
    """Move tensor to device with MPS error handling."""
    try:
        return tensor.to(device)
    except RuntimeError as e:
        if "MPS" in str(e):
            print(f"MPS error: {e}, falling back to CPU")
            return tensor.to("cpu")
        raise
```

---

## Component 3: Monte Carlo Tree Search (MCTS)

### Tree Node Structure

```python
# chessbot/mcts/node.py
class MCTSNode:
    """Node in the MCTS tree."""
    
    def __init__(self, board: ChessBoardWrapper, parent=None, prior: float = 0.0):
        self.board = board
        self.parent = parent
        self.prior = prior  # P(s, a) from network
        
        self.children = {}  # {move: MCTSNode}
        self.visit_count = 0  # N(s, a)
        self.total_value = 0.0  # W(s, a)
        self.mean_value = 0.0  # Q(s, a) = W / N
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def is_expanded(self) -> bool:
        return len(self.children) > 0
    
    def select_child(self, c_puct: float) -> Tuple[chess.Move, 'MCTSNode']:
        """
        Select child with highest UCB score.
        UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        best_score = -float('inf')
        best_move = None
        best_child = None
        
        sqrt_parent_visits = math.sqrt(self.visit_count)
        
        for move, child in self.children.items():
            # UCB formula
            exploitation = child.mean_value
            exploration = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)
            score = exploitation + exploration
            
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        
        return best_move, best_child
    
    def expand(self, policy: np.ndarray):
        """
        Expand node by creating children for all legal moves.
        
        Args:
            policy: (1968,) move probabilities from network
        """
        legal_mask = self.board.get_legal_move_mask()
        legal_policy = policy * legal_mask
        legal_policy /= legal_policy.sum()  # Renormalize
        
        for move in self.board.get_legal_moves():
            move_idx = encode_move(move)
            prior = legal_policy[move_idx]
            
            # Create child node
            child_board = self.board.copy()
            child_board.make_move(move)
            self.children[move] = MCTSNode(child_board, parent=self, prior=prior)
    
    def update(self, value: float):
        """
        Backpropagate value up the tree.
        Value is from perspective of player who just moved.
        """
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count
```

### MCTS Search

```python
# chessbot/mcts/search.py
class MCTS:
    """Monte Carlo Tree Search with neural network guidance."""
    
    def __init__(self, network: ChessNet, num_simulations: int = 800, 
                 c_puct: float = 1.0, temperature: float = 1.0):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
    
    def search(self, board: ChessBoardWrapper) -> Tuple[chess.Move, np.ndarray]:
        """
        Run MCTS from given board position.
        
        Returns:
            best_move: Move to play
            policy: (1968,) MCTS policy (visit counts)
        """
        root = MCTSNode(board.copy())
        
        # Add Dirichlet noise to root for exploration
        self._add_dirichlet_noise(root)
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # 1. Selection: traverse tree using UCB
            while not node.is_leaf() and not node.board.is_game_over():
                move, node = node.select_child(self.c_puct)
                search_path.append(node)
            
            # 2. Expansion: expand leaf node
            if not node.board.is_game_over():
                policy, value = self.network.predict(node.board.get_state())
                node.expand(policy)
            else:
                # Terminal node
                value = node.board.get_result()
            
            # 3. Backpropagation: update all nodes in search path
            for node in reversed(search_path):
                node.update(value)
                value = -value  # Flip value for opponent
        
        # Extract policy from visit counts
        policy = self._get_policy_from_root(root)
        
        # Select move based on temperature
        if self.temperature == 0:
            # Deterministic: pick most visited
            best_move = max(root.children.items(), 
                          key=lambda x: x[1].visit_count)[0]
        else:
            # Stochastic: sample proportional to visits^(1/temp)
            moves = list(root.children.keys())
            visits = np.array([root.children[m].visit_count for m in moves])
            probs = visits ** (1.0 / self.temperature)
            probs /= probs.sum()
            best_move = np.random.choice(moves, p=probs)
        
        return best_move, policy
    
    def _add_dirichlet_noise(self, root: MCTSNode, alpha: float = 0.3, 
                             epsilon: float = 0.25):
        """Add Dirichlet noise to root node priors for exploration."""
        # First expand root
        policy, _ = self.network.predict(root.board.get_state())
        root.expand(policy)
        
        # Add noise
        noise = np.random.dirichlet([alpha] * len(root.children))
        for i, (move, child) in enumerate(root.children.items()):
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]
    
    def _get_policy_from_root(self, root: MCTSNode) -> np.ndarray:
        """Extract MCTS policy from root visit counts."""
        policy = np.zeros(1968)
        total_visits = sum(child.visit_count for child in root.children.values())
        
        for move, child in root.children.items():
            move_idx = encode_move(move)
            policy[move_idx] = child.visit_count / total_visits
        
        return policy
```

---

## Component 4: Self-Play

### Game Generation

```python
# chessbot/selfplay/generator.py
class SelfPlayGenerator:
    """Generate self-play games for training."""
    
    def __init__(self, network: ChessNet, num_simulations: int = 800):
        self.network = network
        self.num_simulations = num_simulations
    
    def generate_game(self) -> List[Experience]:
        """
        Play one self-play game.
        
        Returns:
            experiences: List of (state, policy, value) tuples
        """
        board = ChessBoardWrapper()
        experiences = []
        move_count = 0
        
        while not board.is_game_over():
            # Temperature schedule
            temperature = 1.0 if move_count < 30 else 0.1
            
            # MCTS search
            mcts = MCTS(self.network, self.num_simulations, temperature=temperature)
            move, policy = mcts.search(board)
            
            # Store experience (state, policy target, value will be filled later)
            experiences.append({
                'state': board.get_state().copy(),
                'policy': policy,
                'value': None,  # Will be filled with game outcome
                'player': board.board.turn
            })
            
            # Make move
            board.make_move(move)
            move_count += 1
        
        # Fill in game outcome for all experiences
        result = board.get_result()
        for i, exp in enumerate(experiences):
            # Value from perspective of player who made the move
            if exp['player'] == board.board.turn:
                exp['value'] = result
            else:
                exp['value'] = -result
        
        return experiences
    
    def generate_games(self, num_games: int) -> List[Experience]:
        """Generate multiple self-play games."""
        all_experiences = []
        
        for i in tqdm(range(num_games), desc="Self-play"):
            game_experiences = self.generate_game()
            all_experiences.extend(game_experiences)
            
            # Log to dashboard
            if i % 10 == 0:
                self.log_progress(i, num_games)
        
        return all_experiences
```

### Replay Buffer

```python
# chessbot/selfplay/replay_buffer.py
class ReplayBuffer:
    """SQLite-based replay buffer for training data."""
    
    def __init__(self, db_path: str, max_size: int = 500_000):
        self.db_path = db_path
        self.max_size = max_size
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
    
    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state BLOB NOT NULL,
                policy BLOB NOT NULL,
                value REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_created 
            ON experiences(created_at DESC)
        """)
    
    def add_experiences(self, experiences: List[Experience]):
        """Add new experiences to buffer."""
        for exp in experiences:
            state_bytes = exp['state'].tobytes()
            policy_bytes = exp['policy'].tobytes()
            
            self.conn.execute("""
                INSERT INTO experiences (state, policy, value)
                VALUES (?, ?, ?)
            """, (state_bytes, policy_bytes, exp['value']))
        
        self.conn.commit()
        
        # Enforce max size (FIFO)
        self._enforce_max_size()
    
    def _enforce_max_size(self):
        """Remove oldest experiences if over capacity."""
        count = self.conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
        if count > self.max_size:
            to_remove = count - self.max_size
            self.conn.execute("""
                DELETE FROM experiences
                WHERE id IN (
                    SELECT id FROM experiences
                    ORDER BY created_at ASC
                    LIMIT ?
                )
            """, (to_remove,))
            self.conn.commit()
    
    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample random batch from buffer."""
        rows = self.conn.execute("""
            SELECT state, policy, value
            FROM experiences
            ORDER BY RANDOM()
            LIMIT ?
        """, (batch_size,)).fetchall()
        
        states = np.array([np.frombuffer(r[0]).reshape(119, 8, 8) for r in rows])
        policies = np.array([np.frombuffer(r[1]) for r in rows])
        values = np.array([r[2] for r in rows])
        
        return states, policies, values
    
    def size(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
```

### 8-Way Augmentation

```python
def augment_position(state: np.ndarray, policy: np.ndarray) -> List[Tuple]:
    """
    Apply 8-way symmetry augmentation.
    Returns 8 copies: original + 3 rotations + 4 flips
    """
    augmented = []
    
    for rotation in range(4):
        # Rotate state
        rotated_state = np.rot90(state, k=rotation, axes=(1, 2))
        rotated_policy = rotate_policy(policy, rotation)
        augmented.append((rotated_state, rotated_policy))
        
        # Horizontal flip
        flipped_state = np.flip(rotated_state, axis=2)
        flipped_policy = flip_policy(rotated_policy)
        augmented.append((flipped_state, flipped_policy))
    
    return augmented
```

---

## Component 5: Training Loop

### Loss Function

```python
# chessbot/training/loss.py
def compute_loss(policy_logits, policy_target, value_pred, value_target, 
                 model_params, l2_weight=1e-4):
    """
    AlphaZero loss function.
    
    Loss = MSE(z, v) + CrossEntropy(π, p) + λ·L2(θ)
    
    Args:
        policy_logits: (batch, 1968) network output (pre-softmax)
        policy_target: (batch, 1968) MCTS policy
        value_pred: (batch, 1) network value
        value_target: (batch,) game outcome
        model_params: model parameters for L2 reg
        l2_weight: L2 regularization weight
    """
    # Value loss (MSE)
    value_loss = F.mse_loss(value_pred.squeeze(), value_target)
    
    # Policy loss (cross-entropy)
    log_policy = F.log_softmax(policy_logits, dim=1)
    policy_loss = -torch.sum(policy_target * log_policy, dim=1).mean()
    
    # L2 regularization
    l2_reg = sum(p.pow(2).sum() for p in model_params)
    
    # Total loss
    total_loss = value_loss + policy_loss + l2_weight * l2_reg
    
    return total_loss, value_loss, policy_loss
```

### Training Step

```python
# chessbot/training/trainer.py
class Trainer:
    """Main training coordinator."""
    
    def __init__(self, model: ChessNet, device, monitor: TrainingMonitor):
        self.model = model.to(device)
        self.device = device
        self.monitor = monitor
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1000, gamma=0.9
        )
    
    def train_step(self, states, policies, values):
        """Single training step."""
        self.model.train()
        
        # Move to device
        states = torch.FloatTensor(states).to(self.device)
        policies = torch.FloatTensor(policies).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        
        # Forward pass
        policy_logits, value_pred = self.model(states)
        
        # Compute loss
        loss, value_loss, policy_loss = compute_loss(
            policy_logits, policies, value_pred, values,
            self.model.parameters()
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            'total_loss': loss.item(),
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def train_iteration(self, replay_buffer: ReplayBuffer, 
                       num_steps: int, batch_size: int):
        """Train for one iteration."""
        for step in range(num_steps):
            # Sample batch
            states, policies, values = replay_buffer.sample_batch(batch_size)
            
            # Training step
            metrics = self.train_step(states, policies, values)
            
            # Log to dashboard
            if step % 10 == 0:
                self.monitor.log_metrics(metrics, step=step)
```

---

## Component 6: Evaluation

### Model vs Model Arena

```python
# chessbot/evaluation/arena.py
class Arena:
    """Evaluate two models against each other."""
    
    def __init__(self, num_games: int = 100, num_simulations: int = 400):
        self.num_games = num_games
        self.num_simulations = num_simulations
    
    def compete(self, model1: ChessNet, model2: ChessNet) -> Dict:
        """
        Play model1 vs model2.
        
        Returns:
            results: {'wins': int, 'losses': int, 'draws': int, 'elo_diff': float}
        """
        wins = 0
        losses = 0
        draws = 0
        
        for game_num in range(self.num_games):
            # Alternate colors
            if game_num % 2 == 0:
                result = self._play_game(model1, model2)
            else:
                result = -self._play_game(model2, model1)  # Flip perspective
            
            if result > 0:
                wins += 1
            elif result < 0:
                losses += 1
            else:
                draws += 1
        
        # Calculate ELO difference
        score = (wins + 0.5 * draws) / self.num_games
        elo_diff = self._score_to_elo(score)
        
        return {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': score,
            'elo_diff': elo_diff
        }
    
    def _play_game(self, white_model, black_model) -> float:
        """Play one game, return result from white's perspective."""
        board = ChessBoardWrapper()
        
        while not board.is_game_over():
            model = white_model if board.board.turn == chess.WHITE else black_model
            mcts = MCTS(model, self.num_simulations, temperature=0.1)  # Low temp
            move, _ = mcts.search(board)
            board.make_move(move)
        
        return board.get_result()  # From white's perspective
    
    def _score_to_elo(self, score: float) -> float:
        """Convert win rate to ELO difference."""
        if score <= 0 or score >= 1:
            return 0.0
        return 400 * math.log10(score / (1 - score))
```

---

## Integration with Dashboard

```python
# chessbot/training/main.py
from dashboard.client import TrainingMonitor

def main():
    # Initialize dashboard monitor
    monitor = TrainingMonitor(
        run_name="alphazero_chess_run_001",
        config={
            "network": {"blocks": 10, "channels": 256},
            "mcts": {"simulations": 800},
            "training": {"batch_size": 256, "lr": 0.001}
        }
    )
    
    # Load or create model
    model = ChessNet(num_blocks=10, num_channels=256)
    device = get_device()
    
    # Initialize components
    replay_buffer = ReplayBuffer("data/replay.db", max_size=500_000)
    trainer = Trainer(model, device, monitor)
    
    # Training loop
    for iteration in range(100):
        monitor.log_metrics({"iteration": iteration}, step=iteration)
        
        # 1. Self-play
        generator = SelfPlayGenerator(model, num_simulations=800)
        experiences = generator.generate_games(num_games=500)
        replay_buffer.add_experiences(experiences)
        
        monitor.log_metrics({
            "replay_buffer_size": replay_buffer.size(),
            "games_generated": 500
        }, step=iteration)
        
        # 2. Training
        trainer.train_iteration(replay_buffer, num_steps=1000, batch_size=256)
        
        # 3. Evaluation (every 5 iterations)
        if iteration % 5 == 0:
            arena = Arena(num_games=100)
            results = arena.compete(model, old_model)
            
            monitor.log_metrics({
                "eval_wins": results['wins'],
                "eval_elo_diff": results['elo_diff']
            }, step=iteration)
        
        # 4. Checkpoint
        checkpoint_path = f"checkpoints/model_{iteration}.pt"
        torch.save({
            'model': model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
            'iteration': iteration
        }, checkpoint_path)
        
        monitor.save_checkpoint(checkpoint_path, metrics={
            "loss": trainer.last_loss,
            "elo": current_elo
        })
    
    monitor.set_status("completed")
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_board.py
def test_board_encoding():
    board = ChessBoardWrapper()
    state = board.get_state()
    assert state.shape == (119, 8, 8)

def test_move_encoding():
    move = chess.Move.from_uci("e2e4")
    idx = encode_move(move)
    assert 0 <= idx < 1968
    decoded = decode_move(idx, chess.Board())
    assert decoded == move

# tests/test_network.py
def test_network_forward():
    model = ChessNet(num_blocks=2)
    state = torch.randn(4, 119, 8, 8)
    policy, value = model(state)
    assert policy.shape == (4, 1968)
    assert value.shape == (4, 1)

# tests/test_mcts.py
def test_mcts_search():
    model = ChessNet(num_blocks=2)
    board = ChessBoardWrapper()
    mcts = MCTS(model, num_simulations=10)
    move, policy = mcts.search(board)
    assert move in board.get_legal_moves()
    assert policy.sum() == pytest.approx(1.0)
```

---

## Hardware Optimization (M4 Specific)

### Memory Management

```python
# Monitor memory usage
import psutil

def check_memory():
    memory = psutil.virtual_memory()
    if memory.percent > 85:
        print(f"WARNING: Memory usage at {memory.percent}%")
        # Reduce batch size or MCTS simulations

# Gradient accumulation for large batches
def train_with_accumulation(model, batches, accumulation_steps=4):
    model.zero_grad()
    for i, batch in enumerate(batches):
        loss = compute_loss(batch) / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    policy, value = model(states)
    loss = compute_loss(policy, value, ...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## Open Questions

1. Exact network size (10, 15, or 20 blocks) - determined in Phase 0
2. MCTS simulation count trade-off (speed vs strength)
3. Replay buffer: Keep all positions or sample recent?
4. Should we implement parallel MCTS (multiple games)?
5. Opening book for faster early training?

---

## Future Enhancements

- Transformer architecture experimentation
- Larger network (cloud GPUs)
- UCI protocol support (play on Lichess)
- Web-based play interface
- Analysis mode with move suggestions
- Parallel MCTS (virtual loss)
- Opening book integration
- Endgame tablebase integration

