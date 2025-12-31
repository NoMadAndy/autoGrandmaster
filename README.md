# 🏰 AutoGrandmaster

**A Self-Learning Chess AI with Live Training Visualization**

AutoGrandmaster is a production-ready, open-source chess AI that teaches itself to play chess through self-play using an AlphaZero-style approach. Watch the AI learn in real-time, then challenge it to a game on a beautiful 3D chess board.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.1.0-red.svg)
![React](https://img.shields.io/badge/react-18.2-blue.svg)

---

## ✨ Features

### 🤖 Self-Learning AI
- **AlphaZero-style training** with Policy + Value neural network
- **Monte Carlo Tree Search (MCTS)** for intelligent move selection
- **Self-play game generation** to create training data
- **GPU-accelerated training** with CUDA and mixed precision (AMP)
- **Model versioning and checkpointing** for tracking progress

### 🎮 Interactive Gameplay
- **3D Chess Board** rendered with Three.js
- **Play against the AI** with multiple difficulty levels
- **Real-time move validation** and legal move highlighting
- **Smooth animations** for piece movements

### 📊 Live Training Dashboard
- **Real-time metrics** visualization (policy loss, value loss)
- **Live self-play games** monitoring
- **Training progress** tracking with charts
- **Model information** and statistics

### 🎬 Game Replay
- **Replay past games** move by move
- **Navigate through moves** with controls
- **Analyze game outcomes** and patterns

### 🚀 GitOps & DevOps
- **One-command deployment** with `docker compose up -d`
- **Automated Git monitoring** for repository changes
- **Auto-rebuild and deploy** on code updates
- **Automated changelog** generation
- **Health monitoring** for all services

---

## 🏗️ Architecture

AutoGrandmaster consists of four containerized services:

1. **Frontend** (React + Three.js)
   - 3D chess board visualization
   - Interactive UI with three modes: Play, Training, Replay
   - Real-time WebSocket updates

2. **Backend** (FastAPI)
   - REST API for game management
   - WebSocket for live training events
   - Model management and selection
   - Health monitoring

3. **Trainer** (PyTorch + CUDA)
   - Neural network training
   - Self-play game generation
   - MCTS implementation
   - Model checkpointing and evaluation

4. **GitOps** (Python)
   - Repository monitoring
   - Automated deployment
   - Changelog updates
   - Git operations

---

## 🚀 Quick Start

### Prerequisites

1. **Linux Server** (Ubuntu 20.04+ recommended)
2. **NVIDIA GPU** with CUDA support
3. **Docker** and **Docker Compose**
4. **NVIDIA Container Toolkit**

### Installation

#### 1. Install NVIDIA Container Toolkit

```bash
# Configure the repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

For more details, see the [official NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

#### 2. Clone Repository

```bash
git clone https://github.com/NoMadAndy/autoGrandmaster.git
cd autoGrandmaster
```

#### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings (optional)
nano .env
```

Key environment variables:
- `CUDA_VISIBLE_DEVICES`: GPU device ID (default: 0)
- `TRAINER_ITERATIONS`: Number of training iterations
- `GITOPS_ENABLE_AUTO_PUSH`: Enable/disable auto push to Git
- `GITHUB_TOKEN`: GitHub personal access token (for auto-push)

#### 4. Start All Services

```bash
# Build and start all services
docker compose up -d

# Check service status
docker compose ps

# View logs
docker compose logs -f
```

#### 5. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

---

## 📖 Usage

### Playing Against the AI

1. Navigate to the **Play** tab
2. Select difficulty level (Easy, Medium, Hard, Expert)
3. Click **Start New Game**
4. Click on a piece to select it
5. Click on a highlighted square to move
6. The AI will respond automatically

### Monitoring Training

1. Navigate to the **Training** tab
2. Watch real-time metrics:
   - Policy Loss (how well the AI predicts moves)
   - Value Loss (how well the AI evaluates positions)
3. View live self-play games
4. Track model iterations and improvements

### Viewing Replays

1. Navigate to the **Replay** tab
2. Select a game from the list
3. Use controls to navigate through moves
4. Analyze game outcomes

---

## 🛠️ Development

### Project Structure

```
autoGrandmaster/
├── backend/              # FastAPI backend
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── frontend/             # React + Three.js frontend
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── package.json
│   └── src/
│       ├── App.jsx
│       ├── components/
│       └── styles/
├── trainer/              # PyTorch training service
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── gitops/               # GitOps automation
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── docker-compose.yml    # Service orchestration
├── .env.example          # Environment template
├── .gitignore
├── CHANGELOG.md
└── README.md
```

### Running in Development Mode

```bash
# Backend (with hot reload)
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend (with hot reload)
cd frontend
npm install
npm run dev

# Trainer (standalone)
cd trainer
pip install -r requirements.txt
python main.py
```

### Running Tests

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

---

## 🎯 Configuration

### Training Parameters

Adjust training parameters in `.env`:

```bash
# Training configuration
TRAINER_BATCH_SIZE=256           # Batch size for training
TRAINER_ITERATIONS=1000          # Number of training iterations
TRAINER_SELF_PLAY_GAMES=100      # Games per iteration
TRAINER_MCTS_SIMULATIONS=800     # MCTS simulations per move
TRAINER_CPUCT=1.5                # MCTS exploration constant
```

### GitOps Configuration

Configure automatic deployment:

```bash
# GitOps settings
GITOPS_POLL_INTERVAL=30          # Check for updates every N seconds
GITOPS_ENABLE_AUTO_PULL=true     # Auto-pull from Git
GITOPS_ENABLE_AUTO_DEPLOY=true   # Auto-rebuild and deploy
GITOPS_ENABLE_AUTO_COMMIT=true   # Auto-commit local changes
GITOPS_ENABLE_AUTO_PUSH=false    # Auto-push to remote (requires GITHUB_TOKEN)

# Git configuration
GIT_USER_NAME=AutoGrandmaster Bot
GIT_USER_EMAIL=bot@autograndmaster.local
GITHUB_TOKEN=                    # Personal access token for push
```

### GPU Configuration

Select GPU device:

```bash
# Use first GPU
CUDA_VISIBLE_DEVICES=0

# Use multiple GPUs
CUDA_VISIBLE_DEVICES=0,1

# Disable GPU (CPU only)
CUDA_VISIBLE_DEVICES=-1
```

---

## 📊 Monitoring

### Service Health

Check service health:

```bash
# Check all containers
docker compose ps

# Check individual service logs
docker compose logs backend
docker compose logs trainer
docker compose logs frontend
docker compose logs gitops

# Follow logs in real-time
docker compose logs -f trainer
```

### Training Progress

Monitor training progress:

- View live metrics in the **Training** dashboard
- Check model checkpoints in `models/` directory
- Review training logs: `docker compose logs trainer`

### System Resources

Monitor GPU usage:

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Check Docker stats
docker stats
```

---

## 🔧 Troubleshooting

### GPU Not Detected

**Problem**: Trainer can't access GPU

**Solution**:
```bash
# Verify nvidia-container-toolkit installation
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Restart Docker daemon
sudo systemctl restart docker

# Recreate containers
docker compose down
docker compose up -d
```

### Port Already in Use

**Problem**: Port 3000 or 8000 already in use

**Solution**:
```bash
# Check what's using the port
sudo lsof -i :3000
sudo lsof -i :8000

# Kill the process or change port in docker-compose.yml
```

### Training Too Slow

**Problem**: Training is taking too long

**Solutions**:
1. Reduce `TRAINER_SELF_PLAY_GAMES` in `.env`
2. Reduce `TRAINER_MCTS_SIMULATIONS` in `.env`
3. Use smaller network (edit `trainer/main.py`)
4. Ensure GPU is being used (check `nvidia-smi`)

### Frontend Not Loading

**Problem**: Frontend shows blank page

**Solution**:
```bash
# Check frontend logs
docker compose logs frontend

# Rebuild frontend
docker compose build frontend
docker compose up -d frontend

# Check nginx configuration
docker compose exec frontend cat /etc/nginx/conf.d/default.conf
```

### GitOps Service Issues

**Problem**: GitOps not detecting changes

**Solution**:
```bash
# Check GitOps logs
docker compose logs gitops

# Verify Git configuration
docker compose exec gitops git status

# Ensure Docker socket is mounted
docker compose exec gitops ls -l /var/run/docker.sock
```

---

## 🔒 Security Considerations

### Docker Socket Access

The GitOps service requires access to the Docker socket (`/var/run/docker.sock`) to manage containers. This grants significant privileges.

**Mitigation strategies**:
1. Run in isolated environment
2. Use Docker contexts for remote Docker access
3. Implement webhook-based deployment instead
4. Use Kubernetes operators in production

### Secrets Management

Never commit sensitive information:

```bash
# Always in .gitignore
.env
*.env.local

# Use Docker secrets for production
docker secret create github_token github_token.txt
```

### Network Security

For production deployment:
1. Use reverse proxy (Nginx/Traefik) with SSL
2. Implement authentication (OAuth, JWT)
3. Use firewall rules to restrict access
4. Enable CORS only for specific origins

---

## 🚢 Production Deployment

### Using Reverse Proxy

Example Nginx configuration:

```nginx
server {
    listen 443 ssl http2;
    server_name autograndmaster.example.com;

    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api/ {
        proxy_pass http://localhost:8000;
    }

    location /ws/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Scaling Considerations

For high-traffic deployments:
1. Use multiple trainer instances with different GPUs
2. Add load balancer for backend
3. Use Redis for shared state
4. Implement database for persistence (PostgreSQL)
5. Use Kubernetes for orchestration

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 for Python code
- Use ESLint for JavaScript/React code
- Write tests for new features
- Update documentation
- Keep commits atomic and descriptive

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **AlphaZero** paper by DeepMind for the self-play training approach
- **python-chess** library for chess logic
- **PyTorch** team for the deep learning framework
- **Three.js** for 3D visualization
- **FastAPI** for the modern API framework

---

## 📬 Contact

- **GitHub**: [@NoMadAndy](https://github.com/NoMadAndy)
- **Repository**: [autoGrandmaster](https://github.com/NoMadAndy/autoGrandmaster)
- **Issues**: [GitHub Issues](https://github.com/NoMadAndy/autoGrandmaster/issues)

---

## 🗺️ Roadmap

### Version 1.1
- [ ] Elo rating system
- [ ] Model evaluation and promotion
- [ ] Performance optimizations
- [ ] Advanced MCTS with batching

### Version 1.2
- [ ] Opening book integration
- [ ] Position analysis tools
- [ ] Game database
- [ ] Advanced replay features

### Version 2.0
- [ ] Multi-player support
- [ ] Tournament mode
- [ ] Cloud deployment
- [ ] Mobile app

---

**Made with ♟️ by the AutoGrandmaster Team**