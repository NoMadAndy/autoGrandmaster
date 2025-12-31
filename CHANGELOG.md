# Changelog

All notable changes to AutoGrandmaster will be documented in this file.

## [1.0.0] - 2025-12-31

### Added
- Initial release of AutoGrandmaster
- AlphaZero-style self-play chess AI with PyTorch and CUDA support
- 3D chess board visualization using Three.js
- Play mode: Human vs AI with multiple difficulty levels
- Training mode: Live dashboard showing training metrics and self-play games
- Replay mode: View and analyze past games
- FastAPI backend with REST API and WebSocket support
- GitOps service for automated deployment and continuous integration
- Docker Compose setup for one-command deployment
- Comprehensive documentation and setup guides

### Features
- **AI Training**
  - Policy + Value neural network with ResNet architecture
  - Monte Carlo Tree Search (MCTS) for move selection
  - Self-play game generation
  - Model checkpointing and versioning
  - GPU-accelerated training with mixed precision (AMP)

- **User Interface**
  - Modern, responsive design with dark theme
  - 3D chess board with interactive pieces
  - Real-time training metrics visualization
  - Live self-play game monitoring
  - Game replay with move-by-move navigation

- **Backend Services**
  - RESTful API for game management
  - WebSocket streaming for live training events
  - Model management and selection
  - Game state persistence
  - Health monitoring and status endpoints

- **DevOps**
  - Automated Git repository monitoring
  - Automatic pull and deploy on changes
  - Automated changelog updates
  - Docker Compose orchestration
  - Service health checks
  - Volume management for models and data

### Technical Details
- Python 3.11 for backend and trainer
- PyTorch 2.1.0 with CUDA 12.1
- FastAPI for REST API
- React 18 + Three.js for frontend
- Docker Compose for orchestration
- NVIDIA GPU support via nvidia-container-toolkit

### Security
- Environment-based configuration
- Secrets management via Docker secrets or environment variables
- Gitignore patterns for sensitive files
- Optional authentication support

---

## Future Releases

Planned features for upcoming versions:
- Elo rating system and ranking
- Multi-player support
- Opening book integration
- Advanced position analysis
- Tournament mode
- Mobile app
- Cloud deployment options
