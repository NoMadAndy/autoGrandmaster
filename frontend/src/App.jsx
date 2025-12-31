import React, { useState } from 'react'
import Navigation from './components/Navigation'
import PlayView from './components/PlayView'
import TrainingView from './components/TrainingView'
import ReplayView from './components/ReplayView'
import './App.css'

function App() {
  const [currentView, setCurrentView] = useState('play')

  return (
    <div className="app">
      <Navigation currentView={currentView} onNavigate={setCurrentView} />
      
      <main className="main-content">
        {currentView === 'play' && <PlayView />}
        {currentView === 'training' && <TrainingView />}
        {currentView === 'replay' && <ReplayView />}
      </main>
    </div>
  )
}

export default App
