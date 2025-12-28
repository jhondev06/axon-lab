# AXON - Production-Grade Neural Research Framework for Quantitative Finance

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Tests-116%2B%20Passing-brightgreen.svg" alt="Tests">
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED.svg" alt="Docker">
  <img src="https://img.shields.io/badge/Status-Production-orange.svg" alt="Status">
</p>

> **Enterprise-grade ML pipeline for algorithmic trading research**  
> Modular, resilient, and battle-tested framework with live trading capabilities.

---

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| **8-Step ML Pipeline** | End-to-end automation from data to deployment |
| **Multi-Model Ensemble** | XGBoost, CatBoost, LightGBM, LSTM with intelligent voting |
| **Battle Arena** | Paper & live trading with complete risk management |
| **Resilience System** | Auto-reconnection, state persistence, crash recovery |
| **Telegram Kill Switch** | Remote emergency stop and status monitoring |
| **Multi-Objective Optimization** | Optuna + NSGA-II for Sharpe/Drawdown balancing |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AXON-V3 ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌───────────┐ │
│  │   main.py   │────▶│  pipeline   │────▶│   brains    │────▶│  outputs  │ │
│  │ Orchestrator│     │   8 Steps   │     │ Intelligence│     │  Reports  │ │
│  └─────────────┘     └─────────────┘     └─────────────┘     └───────────┘ │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        DATA CONNECTORS                              │   │
│  │  Yahoo Finance │ Alpha Vantage │ Binance WebSocket │ Synthetic     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   BATTLE ARENA (TRADING ENGINE)                     │   │
│  │  Paper Trading │ Live Trading │ Resilience │ Telegram Bot          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📊 Pipeline Steps

| Step | Name | Description |
|------|------|-------------|
| 1 | **Triage** | Queue processing with priority management |
| 2 | **Dataset** | Multi-source data with intelligent caching |
| 3 | **Features** | 40+ technical indicators + market microstructure |
| 4 | **Train** | Multi-model training with early stopping |
| 5 | **Backtest** | Event-driven simulation with realistic costs |
| 6 | **Error Lens** | Regime detection and error pattern analysis |
| 7 | **Decision** | Promotion gates (Sharpe > 1.0, DD < 15%, WR > 45%) |
| 8 | **Report** | Automated Markdown + Telegram notifications |

---

## 🛡️ Production Resilience

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      RESILIENCE ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐      │
│   │   StateManager  │     │HeartbeatMonitor │     │ReconnectionHandler│    │
│   │ • Atomic writes │     │ • 30s health    │     │ • Exp backoff   │      │
│   │ • Auto backup   │     │ • WS liveness   │     │ • Jitter ±30%   │      │
│   │ • Crash recovery│     │ • Alerts        │     │ • Max 10 retry  │      │
│   └─────────────────┘     └─────────────────┘     └─────────────────┘      │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    TELEGRAM KILL SWITCH                             │  │
│   │   /stop - Emergency stop  │  /status - System status               │  │
│   │   /start - Resume trading │  /positions - Open positions           │  │
│   │   /balance - Account info │  /help - Available commands            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🐳 Docker Deployment

```bash
# Clone and start
git clone https://github.com/jhondev06/axon-lab.git
cd axon-lab

# Configure environment
cp .env.example .env
# Edit .env with your credentials

# Start with Docker
docker-compose up -d

# View logs
docker logs -f axon-trading

# Stop
docker-compose down
```

**Environment Variables:**
```env
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

---

## 💻 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (116+ tests)
python run_tests.py

# Run full pipeline
python main.py

# Run specific module
python -c "from src.battle_arena.core.paper_trader import PaperTrader; p=PaperTrader()"
```

---

## 🧠 ML Models

| Model | Type | Features |
|-------|------|----------|
| **XGBoost** | Gradient Boosting | GPU support, early stopping |
| **CatBoost** | Gradient Boosting | Categorical encoding, GPU |
| **LightGBM** | Gradient Boosting | Fast training, leaf-wise |
| **LSTM** | Deep Learning | Bidirectional, attention |
| **Ensemble** | Meta-Model | Weighted voting, stacking |

**Optimization:**
- Optuna TPE sampler
- NSGA-II multi-objective
- Time-series cross-validation
- Pruning for efficiency

---

## 📈 Risk Management

```yaml
risk:
  max_position_size_pct: 10%     # Per-trade limit
  max_total_exposure_pct: 50%    # Portfolio limit
  max_daily_loss_pct: 5%         # Daily stop
  max_drawdown_pct: 10%          # Maximum drawdown
  max_orders_per_hour: 10        # Rate limiting
```

---

## 📁 Project Structure

```
AXON-V3/
├── main.py                    # Pipeline orchestrator
├── axon.cfg.yml               # Central configuration
├── docker-compose.yml         # Docker deployment
├── requirements.txt           # Dependencies (~50 packages)
│
├── src/                       # Core implementation
│   ├── models.py              # ML models (2505 lines)
│   ├── optimization.py        # Optimization engine (2521 lines)
│   ├── backtest.py            # Backtesting (554 lines)
│   ├── features.py            # Feature engineering (679 lines)
│   ├── brains/                # Intelligence modules
│   └── battle_arena/          # Trading system (18 files)
│
├── tests/                     # Test suite (30+ files, 116+ tests)
├── docs/                      # Documentation
└── outputs/                   # Reports, artifacts, logs
```

---

## 🎯 Performance Thresholds

| Metric | Minimum | Target |
|--------|---------|--------|
| Sharpe Ratio | 1.0 | 2.0+ |
| Max Drawdown | < 15% | < 10% |
| Win Rate | > 45% | > 55% |
| Profit Factor | > 1.5 | > 2.0 |

---

## 🔒 Security

- API keys via environment variables only
- No credentials in code or config
- Telegram authorization with chat_id whitelist
- Read-only config mount in Docker

---

## 📚 Documentation

- **[DEEP_DIVE.md](DEEP_DIVE.md)** - Complete architecture documentation
- **[docs/](docs/)** - API reference and guides
- **[tests/](tests/)** - Test suite and examples

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

---

## 📄 License

**Apache License 2.0** - See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Jhon** - [@jhondev06](https://github.com/jhondev06)

> *"Building production-grade systems, one commit at a time."*

---

<p align="center">
  <strong>⭐ Star this repo if you find it useful! ⭐</strong>
</p>