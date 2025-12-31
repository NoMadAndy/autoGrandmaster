import React, { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import '../styles/TrainingView.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001'
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8001'

function TrainingView() {
  const [isConnected, setIsConnected] = useState(false)
  const [isTraining, setIsTraining] = useState(false)
  const [metrics, setMetrics] = useState([])
  const [liveGames, setLiveGames] = useState([])
  const [currentModel, setCurrentModel] = useState('model_00001')
  const [stats, setStats] = useState({
    iteration: 0,
    policyLoss: 0,
    valueLoss: 0,
    gamesPlayed: 0
  })

  useEffect(() => {
    const ws = new WebSocket(`${WS_URL}/ws/training`)
    
    ws.onopen = () => {
      console.log('WebSocket connected')
      setIsConnected(true)
    }
    
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data)
      
      if (message.type === 'training_status') {
        setIsTraining(message.data.is_running)
      } else if (message.type === 'metrics') {
        const newMetric = {
          iteration: message.data.iteration,
          policyLoss: message.data.policy_loss,
          valueLoss: message.data.value_loss,
          timestamp: message.data.timestamp
        }
        
        setMetrics(prev => [...prev.slice(-50), newMetric])
        setStats({
          iteration: message.data.iteration,
          policyLoss: message.data.policy_loss,
          valueLoss: message.data.value_loss,
          gamesPlayed: stats.gamesPlayed + 1
        })
      } else if (message.type === 'game') {
        setLiveGames(prev => [message.data, ...prev.slice(0, 9)])
      }
    }
    
    ws.onclose = () => {
      console.log('WebSocket disconnected')
      setIsConnected(false)
    }
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }
    
    return () => {
      ws.close()
    }
  }, [])

  const startTraining = async () => {
    try {
      const response = await fetch(`${API_URL}/api/training/start`, {
        method: 'POST'
      })
      const data = await response.json()
      if (data.status === 'started' || data.status === 'already_running') {
        setIsTraining(true)
      }
    } catch (error) {
      console.error('Failed to start training:', error)
    }
  }

  const stopTraining = async () => {
    try {
      const response = await fetch(`${API_URL}/api/training/stop`, {
        method: 'POST'
      })
      const data = await response.json()
      if (data.status === 'stopped' || data.status === 'not_running') {
        setIsTraining(false)
      }
    } catch (error) {
      console.error('Failed to stop training:', error)
    }
  }

  return (
    <div className="training-view">
      <div className="training-header">
        <h1>Training Dashboard</h1>
        <div className="header-controls">
          {isTraining ? (
            <button className="button button-danger" onClick={stopTraining}>
              ⏹ Stop Training
            </button>
          ) : (
            <button className="button" onClick={startTraining}>
              ▶ Start Training
            </button>
          )}
          <div className="connection-status">
            <span className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}></span>
            {isConnected ? 'Connected' : 'Disconnected'}
          </div>
        </div>
      </div>
      
      <div className="stats-grid">
        <div className="stat-card">
          <h3>Iteration</h3>
          <p className="stat-value">{stats.iteration}</p>
        </div>
        
        <div className="stat-card">
          <h3>Policy Loss</h3>
          <p className="stat-value">{stats.policyLoss.toFixed(4)}</p>
        </div>
        
        <div className="stat-card">
          <h3>Value Loss</h3>
          <p className="stat-value">{stats.valueLoss.toFixed(4)}</p>
        </div>
        
        <div className="stat-card">
          <h3>Current Model</h3>
          <p className="stat-value">{currentModel}</p>
        </div>
      </div>
      
      <div className="charts-section">
        <div className="card chart-card">
          <h3>Training Metrics</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="iteration" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid #334155' }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="policyLoss"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
                name="Policy Loss"
              />
              <Line
                type="monotone"
                dataKey="valueLoss"
                stroke="#f59e0b"
                strokeWidth={2}
                dot={false}
                name="Value Loss"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      <div className="games-section">
        <div className="card">
          <h3>Recent Self-Play Games</h3>
          {liveGames.length === 0 ? (
            <p className="no-games">No games yet. Training will start soon...</p>
          ) : (
            <div className="games-list">
              {liveGames.map((game, index) => (
                <div key={index} className="game-item">
                  <div className="game-info">
                    <span className="game-id">Game #{game.id}</span>
                    <span className="game-result">{game.result}</span>
                  </div>
                  <div className="game-moves">
                    {game.moves} moves
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
      
      <div className="info-section">
        <div className="card">
          <h3>About Training</h3>
          <p>
            The AI is learning chess through self-play using an AlphaZero-style approach.
            It plays games against itself, learns from the outcomes, and continuously improves.
          </p>
          <ul>
            <li><strong>Policy Loss:</strong> How well the network predicts moves</li>
            <li><strong>Value Loss:</strong> How well the network evaluates positions</li>
            <li><strong>Self-Play:</strong> The AI plays against itself to generate training data</li>
            <li><strong>MCTS:</strong> Monte Carlo Tree Search guides move selection</li>
          </ul>
        </div>
      </div>
    </div>
  )
}

export default TrainingView
