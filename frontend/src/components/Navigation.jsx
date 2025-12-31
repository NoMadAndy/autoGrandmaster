import React from 'react'

function Navigation({ currentView, onNavigate }) {
  return (
    <nav className="navigation">
      <div className="nav-container">
        <a href="/" className="nav-logo">♔ AutoGrandmaster</a>
        
        <ul className="nav-links">
          <li>
            <button
              className={`nav-button ${currentView === 'play' ? 'active' : ''}`}
              onClick={() => onNavigate('play')}
            >
              Play
            </button>
          </li>
          <li>
            <button
              className={`nav-button ${currentView === 'training' ? 'active' : ''}`}
              onClick={() => onNavigate('training')}
            >
              Training
            </button>
          </li>
          <li>
            <button
              className={`nav-button ${currentView === 'replay' ? 'active' : ''}`}
              onClick={() => onNavigate('replay')}
            >
              Replay
            </button>
          </li>
        </ul>
      </div>
    </nav>
  )
}

export default Navigation
