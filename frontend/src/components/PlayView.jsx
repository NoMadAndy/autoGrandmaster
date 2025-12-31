import React, { useState, useEffect } from 'react'
import ChessBoard3D from './ChessBoard3D'
import '../styles/PlayView.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function PlayView() {
  const [gameId, setGameId] = useState(null)
  const [fen, setFen] = useState('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
  const [legalMoves, setLegalMoves] = useState([])
  const [difficulty, setDifficulty] = useState('medium')
  const [isGameOver, setIsGameOver] = useState(false)
  const [result, setResult] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [showPromotionDialog, setShowPromotionDialog] = useState(false)
  const [pendingMove, setPendingMove] = useState(null)

  const startNewGame = async () => {
    setIsLoading(true)
    try {
      const response = await fetch(`${API_URL}/api/game/new`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ difficulty })
      })
      const data = await response.json()
      
      setGameId(data.game_id)
      setFen(data.fen)
      setLegalMoves(data.legal_moves)
      setIsGameOver(data.is_game_over)
      setResult(null)
    } catch (error) {
      console.error('Failed to start game:', error)
    }
    setIsLoading(false)
  }

  const makeMove = async (move, promotion = 'q') => {
    if (!gameId || isGameOver) return
    
    // Check if this is a pawn promotion
    const fromSquare = move.substring(0, 2)
    const toSquare = move.substring(2, 4)
    const toRank = toSquare[1]
    
    // If moving to rank 8 or 1, might be promotion
    if ((toRank === '8' || toRank === '1') && !pendingMove) {
      // Check if it's actually a pawn move by trying the move
      const isPawnMove = legalMoves.some(lm => 
        lm.startsWith(move.substring(0, 4)) && lm.length > 4
      )
      
      if (isPawnMove) {
        setPendingMove(move)
        setShowPromotionDialog(true)
        return
      }
    }
    
    try {
      const response = await fetch(`${API_URL}/api/game/${gameId}/move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ move, promotion })
      })
      const data = await response.json()
      
      setFen(data.fen)
      setLegalMoves(data.legal_moves)
      setIsGameOver(data.is_game_over)
      setResult(data.result)
      setShowPromotionDialog(false)
      setPendingMove(null)
    } catch (error) {
      console.error('Failed to make move:', error)
    }
  }

  const handlePromotion = (piece) => {
    if (pendingMove) {
      makeMove(pendingMove, piece)
    }
  }

  return (
    <div className="play-view">
      {showPromotionDialog && (
        <div className="promotion-dialog-overlay">
          <div className="promotion-dialog">
            <h3>Choose Promotion Piece</h3>
            <div className="promotion-options">
              <button className="promotion-button" onClick={() => handlePromotion('q')}>♕ Queen</button>
              <button className="promotion-button" onClick={() => handlePromotion('r')}>♖ Rook</button>
              <button className="promotion-button" onClick={() => handlePromotion('b')}>♗ Bishop</button>
              <button className="promotion-button" onClick={() => handlePromotion('n')}>♘ Knight</button>
            </div>
          </div>
        </div>
      )}
      
      <div className="play-container">
        <div className="board-section">
          <ChessBoard3D
            fen={fen}
            legalMoves={legalMoves}
            onMove={makeMove}
            isPlayerTurn={!isGameOver}
          />
        </div>
        
        <div className="controls-section">
          <div className="card">
            <h2>Game Controls</h2>
            
            {!gameId ? (
              <>
                <div className="difficulty-selector">
                  <label>Difficulty:</label>
                  <select
                    value={difficulty}
                    onChange={(e) => setDifficulty(e.target.value)}
                    className="select"
                  >
                    <option value="easy">Easy</option>
                    <option value="medium">Medium</option>
                    <option value="hard">Hard</option>
                    <option value="expert">Expert</option>
                  </select>
                </div>
                
                <button
                  className="button"
                  onClick={startNewGame}
                  disabled={isLoading}
                >
                  Start New Game
                </button>
              </>
            ) : (
              <>
                {isGameOver && (
                  <div className="game-over">
                    <h3>Game Over!</h3>
                    <p>Result: {result}</p>
                  </div>
                )}
                
                <div className="game-info">
                  <p><strong>Game ID:</strong> {gameId}</p>
                  <p><strong>Difficulty:</strong> {difficulty}</p>
                  <p><strong>Status:</strong> {isGameOver ? 'Finished' : 'In Progress'}</p>
                </div>
                
                <button
                  className="button button-secondary"
                  onClick={startNewGame}
                >
                  New Game
                </button>
              </>
            )}
          </div>
          
          <div className="card info-card">
            <h3>How to Play</h3>
            <ul>
              <li>Click on a piece to select it</li>
              <li>Click on a highlighted square to move</li>
              <li>The AI will respond automatically</li>
              <li>Try different difficulty levels!</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

export default PlayView
