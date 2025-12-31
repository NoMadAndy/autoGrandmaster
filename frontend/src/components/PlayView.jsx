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

  const makeMove = async (move) => {
    if (!gameId || isGameOver) return
    
    try {
      const response = await fetch(`${API_URL}/api/game/${gameId}/move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ move })
      })
      const data = await response.json()
      
      setFen(data.fen)
      setLegalMoves(data.legal_moves)
      setIsGameOver(data.is_game_over)
      setResult(data.result)
    } catch (error) {
      console.error('Failed to make move:', error)
    }
  }

  return (
    <div className="play-view">
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
