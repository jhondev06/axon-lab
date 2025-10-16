# Development Setup Guide

## 🚀 Quick Start

Get AXON Neural Research Laboratory running on your local machine in minutes.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.11 or higher
- **Git**: Latest version
- **Docker**: 20.10+ (optional, for containerized development)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB free space minimum

### Operating System Support
- ✅ **Windows 10/11** (WSL2 recommended for Linux compatibility)
- ✅ **macOS 10.15+**
- ✅ **Linux** (Ubuntu 20.04+, CentOS 8+, or equivalent)

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/axon-neural-lab.git
cd axon-neural-lab
```

### 2. Environment Setup

#### Option A: Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv axon-env

# Activate virtual environment
# Windows
axon-env\Scripts\activate
# macOS/Linux
source axon-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Conda Environment

```bash
# Create conda environment
conda create -n axon python=3.11
conda activate axon

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

#### Environment Variables

Create a `.env` file in the project root:

```bash
# Copy template
cp .env.example .env

# Edit configuration
# Windows
notepad .env
# macOS/Linux
nano .env
```

#### Essential Configuration

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/axon_db

# Redis (optional, for caching)
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_HOST=localhost
API_PORT=8000
DEBUG=true

# Model Storage
MODEL_STORAGE_PATH=./models
EXPERIMENT_STORAGE_PATH=./experiments

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/axon.log
```

### 4. Database Setup

#### PostgreSQL (Recommended)

```bash
# Install PostgreSQL
# Windows: Download from postgresql.org
# macOS: brew install postgresql
# Ubuntu: sudo apt-get install postgresql postgresql-contrib

# Create database
createdb axon_db

# Run migrations
python -m axon.database.migrate
```

#### SQLite (Development Only)

```bash
# Use SQLite for quick setup
export DATABASE_URL=sqlite:///./axon.db
python -m axon.database.migrate
```

## 🏃‍♂️ Running AXON

### Development Server

```bash
# Start the API server
python -m axon.api.server

# Or using uvicorn directly
uvicorn axon.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Jupyter Notebook Environment

```bash
# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

### Background Services (Optional)

```bash
# Start Redis (if using caching)
redis-server

# Start task queue worker
python -m axon.workers.celery worker --loglevel=info
```

## 🧪 Verification

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=axon --cov-report=html

# Run specific test suite
pytest tests/unit/
pytest tests/integration/
```

### Health Check

```bash
# Check API health
curl http://localhost:8000/health

# Or using Python
python -c "import requests; print(requests.get('http://localhost:8000/health').json())"
```

### Sample Experiment

```bash
# Run a simple experiment
python examples/quick_start.py

# Or using the CLI
axon experiment run --config examples/sample_config.yaml
```

## 🐳 Docker Development

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Development Container

```bash
# Build development image
docker build -t axon-dev -f Dockerfile.dev .

# Run development container
docker run -it --rm -v $(pwd):/workspace axon-dev bash
```

## 🔧 Development Tools

### Code Quality

```bash
# Format code
black axon/
isort axon/

# Lint code
flake8 axon/
pylint axon/

# Type checking
mypy axon/
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## 📁 Project Structure

```
axon-neural-lab/
├── axon/                   # Main package
│   ├── api/               # REST API
│   ├── core/              # Core functionality
│   ├── models/            # ML models
│   ├── data/              # Data processing
│   ├── experiments/       # Experiment management
│   └── utils/             # Utilities
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── fixtures/          # Test fixtures
├── examples/              # Example scripts
├── docs/                  # Documentation
├── notebooks/             # Jupyter notebooks
├── docker/                # Docker configurations
├── scripts/               # Utility scripts
└── requirements/          # Dependencies
```

## 🚨 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in development mode
pip install -e .
```

#### Database Connection
```bash
# Check PostgreSQL is running
pg_isready -h localhost -p 5432

# Test connection
python -c "from axon.database import engine; print(engine.execute('SELECT 1').scalar())"
```

#### Port Conflicts
```bash
# Check what's using port 8000
netstat -tulpn | grep 8000

# Use different port
uvicorn axon.api.main:app --port 8001
```

### Performance Issues

#### Memory Usage
```bash
# Monitor memory usage
python -m memory_profiler examples/memory_test.py

# Use smaller batch sizes
export BATCH_SIZE=32
```

#### GPU Setup
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🔗 Next Steps

- 📖 Read the [Architecture Overview](../architecture/overview.md)
- 🎯 Try the [Quick Start Tutorial](../user-guides/quick-start.md)
- 🔍 Explore [Example Notebooks](../../notebooks/)
- 🤝 Check [Contributing Guidelines](../../CONTRIBUTING.md)

## 💬 Getting Help

- 📧 **Email**: jhondev06@gmail.com
- 💬 **Discord**: [AXON Community](https://discord.gg/axon-lab)
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-org/axon-neural-lab/issues)
- 📚 **Documentation**: [Full Documentation](../README.md)