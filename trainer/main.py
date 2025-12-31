import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import chess
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
import random

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
CHECKPOINT_DIR = Path(os.getenv("TRAINER_CHECKPOINT_DIR", "/models"))
DATA_DIR = Path(os.getenv("TRAINER_DATA_DIR", "/data"))
BATCH_SIZE = int(os.getenv("TRAINER_BATCH_SIZE", "256"))
ITERATIONS = int(os.getenv("TRAINER_ITERATIONS", "1000"))
SELF_PLAY_GAMES = int(os.getenv("TRAINER_SELF_PLAY_GAMES", "100"))
MCTS_SIMULATIONS = int(os.getenv("TRAINER_MCTS_SIMULATIONS", "800"))
CPUCT = float(os.getenv("TRAINER_CPUCT", "1.5"))

# Create directories
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "games").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "metrics").mkdir(parents=True, exist_ok=True)

logger.info(f"CUDA Available: {CUDA_AVAILABLE}")
logger.info(f"Device: {DEVICE}")
if CUDA_AVAILABLE:
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# Neural Network Architecture
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class ChessNet(nn.Module):
    """AlphaZero-style Policy + Value Network"""
    def __init__(self, num_res_blocks=10, channels=256):
        super().__init__()
        
        # Input: 8x8x18 (board representation)
        self.conv_input = nn.Conv2d(18, channels, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(channels)
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_res_blocks)
        ])
        
        # Policy head
        self.conv_policy = nn.Conv2d(channels, 2, 1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.fc_policy = nn.Linear(2 * 8 * 8, 4096)  # All possible moves
        
        # Value head
        self.conv_value = nn.Conv2d(channels, 1, 1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear(8 * 8, 256)
        self.fc_value2 = nn.Linear(256, 1)
    
    def forward(self, x):
        # Input block
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.bn_policy(self.conv_policy(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.fc_policy(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # Value head
        value = F.relu(self.bn_value(self.conv_value(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.fc_value1(value))
        value = torch.tanh(self.fc_value2(value))
        
        return policy, value

# Board representation
def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Convert chess board to tensor representation (8x8x18)"""
    tensor = np.zeros((18, 8, 8), dtype=np.float32)
    
    piece_map = board.piece_map()
    
    for square, piece in piece_map.items():
        rank, file = divmod(square, 8)
        
        # Piece type (6 types x 2 colors = 12 planes)
        piece_type = piece.piece_type - 1  # 0-5
        color_offset = 0 if piece.color == chess.WHITE else 6
        tensor[piece_type + color_offset, rank, file] = 1
    
    # Side to move (1 plane)
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1
    
    # Castling rights (4 planes)
    tensor[13, :, :] = board.has_kingside_castling_rights(chess.WHITE)
    tensor[14, :, :] = board.has_queenside_castling_rights(chess.WHITE)
    tensor[15, :, :] = board.has_kingside_castling_rights(chess.BLACK)
    tensor[16, :, :] = board.has_queenside_castling_rights(chess.BLACK)
    
    # En passant (1 plane)
    if board.ep_square is not None:
        rank, file = divmod(board.ep_square, 8)
        tensor[17, rank, file] = 1
    
    return torch.from_numpy(tensor)

def move_to_index(move: chess.Move) -> int:
    """Convert move to index (0-4095)"""
    from_sq = move.from_square
    to_sq = move.to_square
    return from_sq * 64 + to_sq

def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Convert index back to move"""
    from_sq = index // 64
    to_sq = index % 64
    move = chess.Move(from_sq, to_sq)
    
    # Check for promotions
    if move in board.legal_moves:
        return move
    
    for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
        promo_move = chess.Move(from_sq, to_sq, promotion=promo)
        if promo_move in board.legal_moves:
            return promo_move
    
    return move

# MCTS Node
class MCTSNode:
    def __init__(self, board: chess.Board, parent=None, move=None, prior=0):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.prior = prior
        
        self.children = []
        self.visit_count = 0
        self.total_value = 0
        self.mean_value = 0
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def select_child(self, cpuct):
        """Select child with highest UCB score"""
        best_score = -float('inf')
        best_child = None
        
        for child in self.children:
            # UCB formula
            exploration = cpuct * child.prior * np.sqrt(self.visit_count) / (1 + child.visit_count)
            score = child.mean_value + exploration
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def expand(self, policy_probs):
        """Expand node with legal moves"""
        legal_moves = list(self.board.legal_moves)
        
        for move in legal_moves:
            move_idx = move_to_index(move)
            prior = policy_probs[move_idx]
            
            new_board = self.board.copy()
            new_board.push(move)
            
            child = MCTSNode(new_board, parent=self, move=move, prior=prior)
            self.children.append(child)
    
    def backup(self, value):
        """Backup value through tree"""
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count
        
        if self.parent:
            self.parent.backup(-value)  # Negate for opponent

# MCTS
class MCTS:
    def __init__(self, model, simulations=800, cpuct=1.5):
        self.model = model
        self.simulations = simulations
        self.cpuct = cpuct
    
    @torch.no_grad()
    def search(self, board: chess.Board) -> np.ndarray:
        """Run MCTS and return visit count distribution"""
        root = MCTSNode(board)
        
        # Get initial policy
        board_tensor = board_to_tensor(board).unsqueeze(0).to(DEVICE)
        with autocast(enabled=CUDA_AVAILABLE):
            log_policy, _ = self.model(board_tensor)
        policy = torch.exp(log_policy).cpu().numpy()[0]
        
        root.expand(policy)
        
        for _ in range(self.simulations):
            node = root
            
            # Selection
            while not node.is_leaf() and not node.board.is_game_over():
                node = node.select_child(self.cpuct)
            
            # Game over check
            if node.board.is_game_over():
                result = node.board.result()
                if result == "1-0":
                    value = 1 if node.board.turn == chess.BLACK else -1
                elif result == "0-1":
                    value = 1 if node.board.turn == chess.WHITE else -1
                else:
                    value = 0
                node.backup(value)
                continue
            
            # Expansion
            board_tensor = board_to_tensor(node.board).unsqueeze(0).to(DEVICE)
            with autocast(enabled=CUDA_AVAILABLE):
                log_policy, value = self.model(board_tensor)
            
            policy = torch.exp(log_policy).cpu().numpy()[0]
            value = value.item()
            
            if not node.is_leaf():
                # Already expanded
                node.backup(value)
            else:
                node.expand(policy)
                node.backup(value)
        
        # Return visit distribution
        visits = np.zeros(4096, dtype=np.float32)
        for child in root.children:
            move_idx = move_to_index(child.move)
            visits[move_idx] = child.visit_count
        
        # Normalize
        if visits.sum() > 0:
            visits = visits / visits.sum()
        
        return visits

# Self-play
def self_play_game(model, mcts: MCTS, temperature=1.0) -> List[Tuple]:
    """Play one self-play game and return training examples"""
    board = chess.Board()
    examples = []
    
    while not board.is_game_over():
        # MCTS search
        policy = mcts.search(board)
        
        # Store training example
        examples.append((
            board_to_tensor(board),
            policy,
            None  # Value will be filled in at end
        ))
        
        # Sample move with temperature
        if temperature > 0:
            move_probs = policy ** (1 / temperature)
            move_probs = move_probs / move_probs.sum()
            move_idx = np.random.choice(len(move_probs), p=move_probs)
        else:
            move_idx = np.argmax(policy)
        
        # Make move
        legal_moves = list(board.legal_moves)
        # Find corresponding legal move
        for move in legal_moves:
            if move_to_index(move) == move_idx:
                board.push(move)
                break
        else:
            # Fallback if index doesn't match
            board.push(random.choice(legal_moves))
    
    # Determine game result
    result = board.result()
    if result == "1-0":
        game_value = 1
    elif result == "0-1":
        game_value = -1
    else:
        game_value = 0
    
    # Fill in values (from perspective of player who made the move)
    for i in range(len(examples)):
        state, policy, _ = examples[i]
        # Alternate value sign
        value = game_value if i % 2 == (len(examples) - 1) % 2 else -game_value
        examples[i] = (state, policy, value)
    
    return examples, result

# Training
def train_network(model, optimizer, scaler, examples, batch_size=256):
    """Train network on examples"""
    model.train()
    
    random.shuffle(examples)
    total_policy_loss = 0
    total_value_loss = 0
    num_batches = 0
    
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i+batch_size]
        
        states = torch.stack([ex[0] for ex in batch]).to(DEVICE)
        target_policies = torch.tensor([ex[1] for ex in batch], dtype=torch.float32).to(DEVICE)
        target_values = torch.tensor([[ex[2]] for ex in batch], dtype=torch.float32).to(DEVICE)
        
        optimizer.zero_grad()
        
        with autocast(enabled=CUDA_AVAILABLE):
            log_policies, values = model(states)
            
            # Policy loss (cross entropy)
            policy_loss = -torch.sum(target_policies * log_policies) / len(batch)
            
            # Value loss (MSE)
            value_loss = F.mse_loss(values, target_values)
            
            # Total loss
            loss = policy_loss + value_loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        num_batches += 1
    
    return total_policy_loss / num_batches, total_value_loss / num_batches

# Event writer for live updates
def write_training_event(event_type: str, data: dict):
    """Write training event to file for backend to stream"""
    try:
        event_file = DATA_DIR / "metrics" / "latest_event.json"
        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        with open(event_file, "w") as f:
            json.dump(event, f)
    except Exception as e:
        logger.error(f"Failed to write event: {e}")

# Main training loop
def main():
    logger.info("Starting AutoGrandmaster Trainer")
    logger.info(f"Configuration: Batch={BATCH_SIZE}, Iterations={ITERATIONS}, Self-play games={SELF_PLAY_GAMES}")
    logger.info(f"Writing events to: {DATA_DIR / 'metrics' / 'latest_event.json'}")
    
    # Write startup event
    write_training_event("log", {"message": "Trainer starting up..."})
    
    # Initialize model
    model = ChessNet(num_res_blocks=5, channels=128).to(DEVICE)  # Smaller for faster training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler(enabled=CUDA_AVAILABLE)
    
    # Load checkpoint if exists
    latest_checkpoint = None
    if CHECKPOINT_DIR.exists():
        checkpoints = sorted(CHECKPOINT_DIR.glob("model_*.pt"))
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            logger.info(f"Loading checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            iteration_offset = checkpoint.get('iteration', 0)
        else:
            iteration_offset = 0
    else:
        iteration_offset = 0
    
    # Save initial model
    if iteration_offset == 0:
        checkpoint_path = CHECKPOINT_DIR / "model_00001.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': 0,
        }, checkpoint_path)
        logger.info(f"Saved initial model: {checkpoint_path}")
        write_training_event("log", {"message": f"Initial model saved: {checkpoint_path.name}"})
    
    mcts = MCTS(model, simulations=MCTS_SIMULATIONS, cpuct=CPUCT)
    
    # Training loop
    for iteration in range(iteration_offset, ITERATIONS):
        logger.info(f"\n=== Iteration {iteration + 1}/{ITERATIONS} ===")
        write_training_event("log", {"message": f"Starting iteration {iteration + 1}/{ITERATIONS}"})
        
        # Self-play
        logger.info(f"Starting self-play: {SELF_PLAY_GAMES} games")
        write_training_event("log", {"message": f"Self-play: {SELF_PLAY_GAMES} games"})
        all_examples = []
        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
        
        start_time = time.time()
        for game_num in range(SELF_PLAY_GAMES):
            examples, result = self_play_game(model, mcts, temperature=1.0)
            all_examples.extend(examples)
            results[result] += 1
            
            if (game_num + 1) % 10 == 0:
                logger.info(f"  Completed {game_num + 1}/{SELF_PLAY_GAMES} games")
                write_training_event("log", {"message": f"Completed {game_num + 1}/{SELF_PLAY_GAMES} games"})
        
        elapsed = time.time() - start_time
        logger.info(f"Self-play completed in {elapsed:.1f}s ({len(all_examples)} examples)")
        logger.info(f"Results: W={results['1-0']}, L={results['0-1']}, D={results['1/2-1/2']}")
        
        write_training_event("log", {
            "message": f"Self-play done: {elapsed:.1f}s, {len(all_examples)} examples, W/D/L={results['1-0']}/{results['1/2-1/2']}/{results['0-1']}"
        })
        
        # Training
        logger.info("Training network...")
        write_training_event("log", {"message": "Training neural network..."})
        policy_loss, value_loss = train_network(model, optimizer, scaler, all_examples, BATCH_SIZE)
        logger.info(f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
        
        write_training_event("log", {
            "message": f"Training complete - Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}"
        })
        
        # Save metrics
        metrics = {
            "iteration": iteration + 1,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "self_play_games": SELF_PLAY_GAMES,
            "examples": len(all_examples),
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        write_training_event("metrics", metrics)
        
        metrics_file = DATA_DIR / "metrics" / f"metrics_{iteration + 1:05d}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save checkpoint every 10 iterations
        if (iteration + 1) % 10 == 0:
            checkpoint_path = CHECKPOINT_DIR / f"model_{iteration + 1:05d}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iteration': iteration + 1,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            write_training_event("log", {"message": f"Checkpoint saved: model_{iteration + 1:05d}.pt"})
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
