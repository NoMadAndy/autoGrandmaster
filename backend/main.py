import os
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import chess
import asyncio
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(title="AutoGrandmaster API", version="1.0.0")

# CORS
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3001").split(",")
# Add both localhost and 127.0.0.1 variants
all_origins = cors_origins + [
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "http://localhost:8001",
    "http://127.0.0.1:8001"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=all_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connections manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New WebSocket connection. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message: {e}")

manager = ConnectionManager()

# In-memory storage (will be replaced with DB)
games_store: Dict[str, Any] = {}
models_store = {
    "active": "model_00001",
    "best": "model_00001",
    "models": [
        {"id": "model_00001", "version": 1, "elo": 1200, "created_at": datetime.utcnow().isoformat()}
    ]
}
replays_store: List[Dict[str, Any]] = []
training_state = {
    "is_running": False,
    "iteration": 0
}

# Pydantic Models
class NewGameRequest(BaseModel):
    difficulty: str = "medium"

class MakeMoveRequest(BaseModel):
    move: str
    promotion: Optional[str] = "q"  # Default to queen

class GameState(BaseModel):
    game_id: str
    fen: str
    legal_moves: List[str]
    is_game_over: bool
    result: Optional[str] = None

class ModelInfo(BaseModel):
    id: str
    version: int
    elo: int
    created_at: str

class SetActiveModelRequest(BaseModel):
    model_id: str

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# Game endpoints
@app.post("/api/game/new", response_model=GameState)
async def new_game(request: NewGameRequest):
    """Start a new game against the AI"""
    game_id = f"game_{len(games_store) + 1}_{datetime.utcnow().timestamp()}"
    board = chess.Board()
    
    games_store[game_id] = {
        "board": board,
        "difficulty": request.difficulty,
        "moves": []
    }
    
    logger.info(f"New game started: {game_id}, difficulty: {request.difficulty}")
    
    return GameState(
        game_id=game_id,
        fen=board.fen(),
        legal_moves=[move.uci() for move in board.legal_moves],
        is_game_over=board.is_game_over()
    )

@app.get("/api/game/{game_id}/state", response_model=GameState)
async def get_game_state(game_id: str):
    """Get current game state"""
    if game_id not in games_store:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = games_store[game_id]
    board = game["board"]
    
    result = None
    if board.is_game_over():
        if board.is_checkmate():
            result = "checkmate"
        elif board.is_stalemate():
            result = "stalemate"
        elif board.is_insufficient_material():
            result = "insufficient_material"
        else:
            result = "draw"
    
    return GameState(
        game_id=game_id,
        fen=board.fen(),
        legal_moves=[move.uci() for move in board.legal_moves],
        is_game_over=board.is_game_over(),
        result=result
    )

@app.post("/api/game/{game_id}/move", response_model=GameState)
async def make_move(game_id: str, request: MakeMoveRequest):
    """Make a move in the game"""
    if game_id not in games_store:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = games_store[game_id]
    board = game["board"]
    
    try:
        move = chess.Move.from_uci(request.move)
        if move not in board.legal_moves:
            raise HTTPException(status_code=400, detail="Illegal move")
        
        # Handle pawn promotion
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.PAWN:
            # Check if move is to last rank
            to_rank = chess.square_rank(move.to_square)
            if (piece.color == chess.WHITE and to_rank == 7) or (piece.color == chess.BLACK and to_rank == 0):
                # Add promotion piece
                promotion_map = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}
                promotion_piece = promotion_map.get(request.promotion, chess.QUEEN)
                move = chess.Move(move.from_square, move.to_square, promotion=promotion_piece)
        
        board.push(move)
        game["moves"].append(move.uci())
        logger.info(f"Move made in {game_id}: {move.uci()}")
        
        # AI response (dummy for now - will be replaced with actual AI)
        if not board.is_game_over():
            # Simple random legal move for now
            import random
            ai_move = random.choice(list(board.legal_moves))
            board.push(ai_move)
            game["moves"].append(ai_move.uci())
            logger.info(f"AI move in {game_id}: {ai_move.uci()}")
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid move format: {str(e)}")
    
    result = None
    if board.is_game_over():
        if board.is_checkmate():
            result = "checkmate"
        elif board.is_stalemate():
            result = "stalemate"
        elif board.is_insufficient_material():
            result = "insufficient_material"
        else:
            result = "draw"
    
    return GameState(
        game_id=game_id,
        fen=board.fen(),
        legal_moves=[move.uci() for move in board.legal_moves],
        is_game_over=board.is_game_over(),
        result=result
    )

@app.get("/api/game/{game_id}/legal")
async def get_legal_moves(game_id: str):
    """Get legal moves for current position"""
    if game_id not in games_store:
        raise HTTPException(status_code=404, detail="Game not found")
    
    board = games_store[game_id]["board"]
    return {"legal_moves": [move.uci() for move in board.legal_moves]}

# Model endpoints
@app.get("/api/models", response_model=List[ModelInfo])
async def get_models():
    """Get list of all models"""
    return models_store["models"]

@app.get("/api/models/active")
async def get_active_model():
    """Get currently active model"""
    return {
        "active": models_store["active"],
        "best": models_store["best"]
    }

@app.post("/api/models/set")
async def set_active_model(request: SetActiveModelRequest):
    """Set active model for gameplay"""
    models_store["active"] = request.model_id
    logger.info(f"Active model changed to: {request.model_id}")
    return {"success": True, "active_model": request.model_id}

# Replay endpoints
@app.get("/api/replays")
async def get_replays():
    """Get list of game replays"""
    return {"replays": replays_store[:50]}  # Return last 50

@app.get("/api/replays/{replay_id}")
async def get_replay(replay_id: str):
    """Get specific replay"""
    replay = next((r for r in replays_store if r["id"] == replay_id), None)
    if not replay:
        raise HTTPException(status_code=404, detail="Replay not found")
    return replay

# Changelog endpoint
@app.get("/api/changelog")
async def get_changelog():
    """Get changelog entries"""
    try:
        with open("/repo/CHANGELOG.md", "r") as f:
            content = f.read()
            # Parse markdown and return structured data
            entries = []
            current_entry = None
            for line in content.split("\n"):
                if line.startswith("## "):
                    if current_entry:
                        entries.append(current_entry)
                    current_entry = {"version": line[3:].strip(), "changes": []}
                elif line.startswith("- ") and current_entry:
                    current_entry["changes"].append(line[2:].strip())
            if current_entry:
                entries.append(current_entry)
            return {"entries": entries[:10]}  # Return last 10 entries
    except FileNotFoundError:
        return {"entries": []}

# Training control endpoints
@app.post("/api/training/start")
async def start_training():
    """Start training process"""
    if training_state["is_running"]:
        return {"status": "already_running", "message": "Training is already running"}
    
    training_state["is_running"] = True
    logger.info("Training started")
    
    # Broadcast to WebSocket clients
    await manager.broadcast({
        "type": "training_status",
        "data": {"is_running": True, "timestamp": datetime.utcnow().isoformat()}
    })
    
    return {"status": "started", "message": "Training started successfully"}

@app.post("/api/training/stop")
async def stop_training():
    """Stop training process"""
    if not training_state["is_running"]:
        return {"status": "not_running", "message": "Training is not running"}
    
    training_state["is_running"] = False
    logger.info("Training stopped")
    
    # Broadcast to WebSocket clients
    await manager.broadcast({
        "type": "training_status",
        "data": {"is_running": False, "timestamp": datetime.utcnow().isoformat()}
    })
    
    return {"status": "stopped", "message": "Training stopped successfully"}

@app.get("/api/training/status")
async def get_training_status():
    """Get current training status"""
    return {
        "is_running": training_state["is_running"],
        "iteration": training_state["iteration"]
    }

# WebSocket for training events
@app.websocket("/ws/training")
async def websocket_training(websocket: WebSocket):
    """WebSocket endpoint for training events streaming"""
    await manager.connect(websocket)
    
    try:
        # Send initial dummy data
        await websocket.send_json({
            "type": "init",
            "data": {
                "status": "Training system ready",
                "model": models_store["active"],
                "timestamp": datetime.utcnow().isoformat()
            }
        })
        
        # Keep connection alive and send periodic updates
        while True:
            # This will be replaced with actual training events
            await asyncio.sleep(2)
            await websocket.send_json({
                "type": "heartbeat",
                "data": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "active_games": 0,
                    "model": models_store["active"]
                }
            })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Training WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Background task simulator (will be replaced by actual trainer events)
async def simulate_training_events():
    """Read and broadcast trainer events"""
    await asyncio.sleep(5)  # Wait for startup
    
    event_file = Path("/data/metrics/latest_event.json")
    last_event = None
    
    while True:
        await asyncio.sleep(1)  # Check every second
        
        try:
            if event_file.exists():
                with open(event_file, 'r') as f:
                    event = json.load(f)
                
                # Only broadcast if it's a new event
                if event != last_event:
                    await manager.broadcast(event)
                    last_event = event
        except Exception as e:
            logger.error(f"Error reading trainer events: {e}")

@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    logger.info("Backend starting up...")
    asyncio.create_task(simulate_training_events())

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("BACKEND_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
