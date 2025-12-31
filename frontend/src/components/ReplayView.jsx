import React, { useState, useEffect } from 'react'
import '../styles/ReplayView.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function ReplayView() {
  const [replays, setReplays] = useState([])
  const [selectedReplay, setSelectedReplay] = useState(null)
  const [currentMove, setCurrentMove] = useState(0)

  useEffect(() => {
    fetchReplays()
  }, [])

  const fetchReplays = async () => {
    try {
      const response = await fetch(`${API_URL}/api/replays`)
      const data = await response.json()
      setReplays(data.replays || [])
    } catch (error) {
      console.error('Failed to fetch replays:', error)
    }
  }

  const loadReplay = async (replayId) => {
    try {
      const response = await fetch(`${API_URL}/api/replays/${replayId}`)
      const data = await response.json()
      setSelectedReplay(data)
      setCurrentMove(0)
    } catch (error) {
      console.error('Failed to load replay:', error)
    }
  }

  const handlePrevious = () => {
    if (currentMove > 0) {
      setCurrentMove(currentMove - 1)
    }
  }

  const handleNext = () => {
    if (selectedReplay && currentMove < selectedReplay.moves.length - 1) {
      setCurrentMove(currentMove + 1)
    }
  }

  return (
    <div className="replay-view">
      <h1>Game Replays</h1>
      
      <div className="replay-container">
        <div className="replays-list card">
          <h3>Available Replays</h3>
          {replays.length === 0 ? (
            <p className="no-replays">No replays available yet.</p>
          ) : (
            <div className="replay-items">
              {replays.map((replay) => (
                <div
                  key={replay.id}
                  className={`replay-item ${selectedReplay?.id === replay.id ? 'active' : ''}`}
                  onClick={() => loadReplay(replay.id)}
                >
                  <div className="replay-header">
                    <span className="replay-id">{replay.id}</span>
                    <span className="replay-result">{replay.result}</span>
                  </div>
                  <div className="replay-meta">
                    <span>{replay.moves?.length || 0} moves</span>
                    <span>{new Date(replay.timestamp).toLocaleDateString()}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        
        <div className="replay-player card">
          {selectedReplay ? (
            <>
              <h3>Replay: {selectedReplay.id}</h3>
              
              <div className="board-placeholder">
                <p>Board visualization</p>
                <p>Move {currentMove + 1} of {selectedReplay.moves.length}</p>
              </div>
              
              <div className="replay-controls">
                <button
                  className="button button-secondary"
                  onClick={handlePrevious}
                  disabled={currentMove === 0}
                >
                  ← Previous
                </button>
                
                <span className="move-counter">
                  {currentMove + 1} / {selectedReplay.moves.length}
                </span>
                
                <button
                  className="button button-secondary"
                  onClick={handleNext}
                  disabled={currentMove === selectedReplay.moves.length - 1}
                >
                  Next →
                </button>
              </div>
              
              <div className="moves-list">
                <h4>Moves</h4>
                <div className="moves-grid">
                  {selectedReplay.moves?.map((move, index) => (
                    <span
                      key={index}
                      className={`move-item ${index === currentMove ? 'current' : ''}`}
                      onClick={() => setCurrentMove(index)}
                    >
                      {index + 1}. {move}
                    </span>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div className="no-selection">
              <p>Select a replay from the list to view</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default ReplayView
